import logging
from decimal import Decimal
from typing import Dict, List

import pandas as pd
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

# ✅ Import ML models
from scripts.ml_models import npc_vol, npc_trend, npc_inv


class NPCStrategy(ScriptStrategyBase):
    # ✅ Required for Hummingbot to load markets
    markets = {"binance_perpetual_testnet": {"ETH-USDT"}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.exchange = "binance_perpetual_testnet"
        self.trading_pair = "ETH-USDT"
        self.base, self.quote = self.trading_pair.split("-")
        self.price_source = PriceType.MidPrice
        self.order_amount = 0.01
        self.order_refresh_time = 15
        self.create_timestamp = 0

        self.bid_spread = 0.001
        self.ask_spread = 0.001
        self.reference_price = Decimal("1")

        # ✅ Candle config
        self.candles = CandlesFactory.get_candle(CandlesConfig(
            connector="binance_perpetual",
            trading_pair=self.trading_pair,
            interval="1m",
            max_records=1000
        ))
        self.candles.start()

    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            self.update_strategy_variables()
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def update_strategy_variables(self):
        df = self.candles.candles_df.copy()
        df.dropna(inplace=True)

        if len(df) < 30:
            self.log_with_clock(logging.WARNING, "⏳ Waiting for more candle data...")
            return

        # ✅ Call volatility model
        spread_result = npc_vol.train_and_predict(df)
        self.bid_spread = spread_result.get("bid_spread", 0.001)
        self.ask_spread = spread_result.get("ask_spread", 0.001)

        # ✅ Call trend model
        trend_shift = npc_trend.train_and_predict(df)

        # ✅ Compute inventory skew
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        mid_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)

        base_val = float(base_bal * mid_price)
        quote_val = float(quote_bal)
        total_val = base_val + quote_val
        inventory_ratio = base_val / total_val if total_val > 0 else 0.5

        inv_shift = npc_inv.train_and_predict(inventory_ratio)

        # ✅ Update reference price
        self.reference_price = Decimal(mid_price) * Decimal(1 + trend_shift) * Decimal(1 + inv_shift)

        self.log_with_clock(logging.INFO, f"📊 Spreads: bid={self.bid_spread:.4f}, ask={self.ask_spread:.4f}")
        self.log_with_clock(logging.INFO, f"📈 Trend shift: {trend_shift:.6f}, Inventory shift: {inv_shift:.6f}")
        self.log_with_clock(logging.INFO, f"📌 Ref price: {self.reference_price:.2f}")

    def create_proposal(self) -> List[OrderCandidate]:
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        buy_price = min(self.reference_price * Decimal(1 - self.bid_spread), Decimal(best_bid))
        sell_price = max(self.reference_price * Decimal(1 + self.ask_spread), Decimal(best_ask))

        return [
            OrderCandidate(self.trading_pair, True, OrderType.LIMIT, TradeType.BUY,
                           Decimal(self.order_amount), buy_price),
            OrderCandidate(self.trading_pair, True, OrderType.LIMIT, TradeType.SELL,
                           Decimal(self.order_amount), sell_price)
        ]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

    def place_orders(self, proposal: List[OrderCandidate]):
        for order in proposal:
            if order.order_side == TradeType.SELL:
                self.sell(self.exchange, order.trading_pair, order.amount, order.order_type, order.price)
            else:
                self.buy(self.exchange, order.trading_pair, order.amount, order.order_type, order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        return (
            f"\n🔄 Strategy Running...\n"
            f"📌 Ref Price: {self.reference_price:.4f}\n"
            f"📉 Bid Spread: {self.bid_spread * 10000:.2f} bps\n"
            f"📈 Ask Spread: {self.ask_spread * 10000:.2f} bps\n"
        )
