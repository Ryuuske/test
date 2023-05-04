import logging
from decimal import Decimal
from math import ceil, floor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.exchange_base cimport ExchangeBase
from hummingbot.core.clock cimport Clock
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.limit_order cimport LimitOrder
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.utils import map_df_to_str
from hummingbot.strategy.asset_price_delegate cimport AssetPriceDelegate
from hummingbot.strategy.asset_price_delegate import AssetPriceDelegate
from hummingbot.strategy.hanging_orders_tracker import CreatedPairOfOrders, HangingOrdersTracker
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.order_book_asset_price_delegate cimport OrderBookAssetPriceDelegate
from hummingbot.strategy.strategy_base import StrategyBase
from hummingbot.strategy.utils import order_age
from .data_types import PriceSize, Proposal
from .inventory_cost_price_delegate import InventoryCostPriceDelegate
from .inventory_skew_calculator cimport c_calculate_bid_ask_ratios_from_base_asset_ratio
from .inventory_skew_calculator import calculate_total_order_size
from .pure_market_making_order_tracker import PureMarketMakingOrderTracker
from .moving_price_band import MovingPriceBand


NaN = float("nan")
s_decimal_zero = Decimal(0)
s_decimal_neg_one = Decimal(-1)
pmm_logger = None


class PureScalpingStrategy(StrategyBase):
    def __init__(self,
             market_info: MarketTradingPairTuple,
             asset_price_delegate: Optional[AssetPriceDelegate] = None):
		super().__init__()
		self._market_info = market_info
		self._connector = self._market_info.trading_pair_tuple[0].exchange
		self._order_book_bid_depth_tolerance = self._market_info.market.get("order_book_bid_depth_tolerance", 0.0)
		self._order_book_ask_depth_tolerance = self._market_info.market.get("order_book_ask_depth_tolerance", 0.0)
		self._asset_price_delegate = asset_price_delegate or OrderBookAssetPriceDelegate(self._market_info)
		self._order_refresh_time = self._market_info.market.get("limit_order_refresh_time", 15.0)
		self._minimum_spread = self._market_info.market.get("minimum_spread", 0.0)
		self._order_levels = []
		self._hanging_orders_tracker = HangingOrdersTracker(self)
		self._pmm_order_book_bid_depth = 0.0
		self._pmm_order_book_ask_depth = 0.0
		self._pmm_bids = []
		self._pmm_asks = []
		self._last_state = None
		self._order_price_spread = self._market_info.market.get("order_price_spread", Decimal("0"))
		self._active_bids = []
		self._active_asks = []
		self._spread = Decimal(0)
		self._last_mid_price = Decimal(0)
		self._mid_price = Decimal(0)
		self._last_price = Decimal(0)
		self._trading_pair = self._market_info.trading_pair_tuple.trading_pair
		self._last_trade_type = TradeType.SELL
		self._last_trade_price = Decimal(0)
		self._minimum_order_size: Decimal = self._market_info.market.get("min_order_size", Decimal("0"))
		self._inventory_cost_price_delegate = InventoryCostPriceDelegate()
		self._order_book_watcher_task = None
		self._order_book_watcher_last_timestamp = 0
		self.continuous_orders = 0
		self.max_allowed_orders = 0
		self.active_orders = set()
		self.active_buys = set()
		self.active_sells = set()
		self.canceled_orders = set()
		self.order_levels = []
	
	def configure_parser(cls, parser: argparse.ArgumentParser, child_parser: argparse.ArgumentParser) -> None:
		child_parser.add_argument("--continuous_orders", type=int, required=True,
								  help="The number of continuous orders to place after the initial orders are filled.")
		child_parser.add_argument("--max_allowed_orders", type=int, required=True,
								  help="The maximum number of active orders allowed, including both initial and continuous orders.")
		child_parser.add_argument("--order_levels", type=float, nargs="+", required=True,
								  help="The spread between order levels, starting from the best bid and ask price.")
		super(PureMarketMakingStrategy, cls).configure_parser(parser, child_parser)
	
	def validate_parameters(self):
		super().validate_parameters()
		if self.min_profitability is not None and self.min_profitability <= 0:
			raise ValueError("Invalid min_profitability value. Please set it to a positive number.")
		if self._order_book_bid_depth_tolerance < 0:
			raise ValueError("Invalid order_book_bid_depth_tolerance value. Please set it to a non-negative number.")
		if self._order_book_ask_depth_tolerance < 0:
			raise ValueError("Invalid order_book_ask_depth_tolerance value. Please set it to a non-negative number.")
		if self._order_refresh_time <= 0:
			raise ValueError("Invalid order_refresh_time value. Please set it to a positive number.")
		if self._minimum_spread < 0:
			raise ValueError("Invalid minimum_spread value. Please set it to a non-negative number.")
		if self._asset_price_delegate is None:
			raise ValueError("Invalid asset_price_delegate value. Please set it to a valid object.")
		if self.continuous_orders <= 0:
			raise ValueError("Invalid continuous_orders value. Please set it to a positive number.")
		if self.max_allowed_orders <= 0:
			raise ValueError("Invalid max_allowed_orders value. Please set it to a positive number.")
		if len(self.order_levels) == 0:
			raise ValueError("Invalid order_levels value. Please set it to a list of positive numbers.")
		for level in self.order_levels:
			if level <= 0:
				raise ValueError("Invalid order_levels value. Please set it to a list of positive numbers.")
	
	def initialize_markers(self):
		self._hanging_orders_tracker.reset()
		self._active_bids = []
		self._active_asks = []
		self._pmm_bids = []
		self._pmm_asks = []
		self._last_state = None
		self._spread = Decimal(0)
		self._last_mid_price = self.mid_price
		self.order_levels = sorted(self.order_levels, reverse=True)
		self.active_orders = set()
		self.active_buys = set()
		self.active_sells = set()
		self.canceled_orders = set()
		initial_order_size = calculate_total_order_size(self.order_levels[0], self._connector.min_order_size)
		price_band = MovingPriceBand(
			order_size=initial_order_size,
			order_levels=self.order_levels,
			spread=self._order_price_spread,
			mid_price=self.mid_price,
			min_profitability=self.min_profitability,
			connector=self._connector,
			asset_price_delegate=self._asset_price_delegate
		)
		self._hanging_orders_tracker.add_price_band(price_band)
		self.add_to_active_orders(price_band.bid_limit_orders)
		self.add_to_active_orders(price_band.ask_limit_orders)
		self.add_to_active_buys(price_band.bid_limit_orders)
		self.add_to_active_sells(price_band.ask_limit_orders)
		self.place_initial_orders(price_band.bid_limit_orders + price_band.ask_limit_orders)
	
	def place_initial_orders(self, orders: List[LimitOrder]):
		self._connector.cancel_all(self._trading_pair)
		for order in orders:
			if self._connector.ready and order.is_valid:
				self._connector.submit_order(order)
				self.active_orders.add(order.client_order_id)
				if order.is_buy:
					self.active_buys.add(order.client_order_id)
				else:
					self.active_sells.add(order.client_order_id)
				self.ev_loop.create_task(self.trigger_tracking())
			else:
				self.logger().warning(
					"Failed to place order %s because Hummingbot is not ready or order is invalid.",
					order.client_order_id
				)
	
	def place_additional_orders(self, proposals: List[Proposal]):
		if len(proposals) == 0:
			return
		self.logger().info(f"Submitting {len(proposals)} new proposals...")
		new_bid_orders = []
		new_ask_orders = []
		for proposal in proposals:
			if proposal.is_buy:
				self.logger().info(f"Submitting new bid {proposal.order}")
				self._pmm_bids.append(proposal.order)
				new_bid_orders.append(proposal.order)
			else:
				self.logger().info(f"Submitting new ask {proposal.order}")
				self._pmm_asks.append(proposal.order)
				new_ask_orders.append(proposal.order)
		self.add_to_active_orders(new_bid_orders + new_ask_orders)
		self.add_to_active_buys(new_bid_orders)
		self.add_to_active_sells(new_ask_orders)
		self.ev_loop.create_task(self.trigger_tracking())
		for order in new_bid_orders + new_ask_orders:
			self._connector.submit_order(order)
			self.active_orders.add(order.client_order_id)
			if order.is_buy:
				self.active_buys.add(order.client_order_id)
			else:
				self.active_sells.add(order.client_order_id)
			
	def create_order_proposals(self, bid_price: Decimal, ask_price: Decimal, mid_price: Decimal) -> List[Proposal]:
		bid_order_size = calculate_total_order_size(self.order_levels[0], self._connector.min_order_size)
		ask_order_size = calculate_total_order_size(self.order_levels[0], self._connector.min_order_size)
		if len(self._pmm_bids) + len(self.active_buys) >= self.max_allowed_orders:
			bid_order_size = s_decimal_zero
		if len(self._pmm_asks) + len(self.active_sells) >= self.max_allowed_orders:
			ask_order_size = s_decimal_zero
		bid_orders, ask_orders = [], []
		order_book = self._connector.get_order_book(self._trading_pair, depth=self.order_book_depth_level)
		if order_book is None:
			self.logger().warning(f"Skipping tick update due to empty order book for {self._trading_pair}.")
			return []
		bid_depth = order_book.bid_entries
		ask_depth = order_book.ask_entries
		bid_base_amount = sum([order.base_asset_amount for order in self.active_buys])
		ask_base_amount = sum([order.base_asset_amount for order in self.active_sells])
		for level in self.order_levels:
			bid_proposal_size = calculate_total_order_size(level, self._connector.min_order_size)
			ask_proposal_size = calculate_total_order_size(level, self._connector.min_order_size)
			bid_proposal_size = min(bid_proposal_size, bid_order_size, self._connector.quantize_order_amount(self._trading_pair, bid_base_amount))
			ask_proposal_size = min(ask_proposal_size, ask_order_size, self._connector.quantize_order_amount(self._trading_pair, ask_base_amount))
			bid_order = self.place_bid_order(bid_price, bid_proposal_size, order_type=self.order_type)
			ask_order = self.place_ask_order(ask_price, ask_proposal_size, order_type=self.order_type)
			if bid_order is not None:
				bid_orders.append(bid_order)
			if ask_order is not None:
				ask_orders.append(ask_order)
			bid_order_size -= bid_proposal_size
			ask_order_size -= ask_proposal_size
			if bid_order_size <= 0 and ask_order_size <= 0:
				break
		return [Proposal(bid_order, True) for bid_order in bid_orders] + [Proposal(ask_order, False) for ask_order in ask_orders]
	
	def update_mid_prices(self, order_book: Dict[str, any]):
		self.mid_price = (Decimal(str(order_book["bids"][0]["price"])) + Decimal(str(order_book["asks"][0]["price"]))) / 2
		self.bid_spread = self.mid_price * (Decimal("1") - self.bid_spread_percentage / Decimal("100"))
		self.ask_spread = self.mid_price * (Decimal("1") + self.ask_spread_percentage / Decimal("100"))
		self.order_level_spread = self.mid_price * (Decimal("1") - self.order_level_spread_percentage / Decimal("100"))
	
	def update_orders(self, order_book: Dict[str, any]):
		self.hanging_orders_tracker.update_orders(
			self._pmm_bids + self._pmm_asks,
			order_book["bids"],
			order_book["asks"]
		)
		for order_id in list(self.active_orders):
			if order_id not in self.hanging_orders_tracker.tracked_order_ids:
				self.logger().info(f"Order {order_id} is not tracked anymore. Removing from tracked orders.")
				self.active_orders.remove(order_id)
				if order_id in self.active_buys:
					self.active_buys.remove(order_id)
				else:
					self.active_sells.remove(order_id)
		self._pmm_bids = self.hanging_orders_tracker.bids()
		self._pmm_asks = self.hanging_orders_tracker.asks()
	
	def should_place_order(self) -> bool:
		if len(self._pmm_bids) + len(self.active_buys) >= self.max_allowed_orders or len(self._pmm_asks) + len(self.active_sells) >= self.max_allowed_orders:
			return False
		return True
		
	def place_orders(self, proposals: List[Proposal]):
		for proposal in proposals:
			self.active_orders.append(proposal.order.client_order_id)
			if proposal.is_buy:
				self.active_buys.append(proposal.order.client_order_id)
			else:
				self.active_sells.append(proposal.order.client_order_id)
			self.logger().info(f"Sending {proposal.order.side.name.lower()} order {proposal.order.client_order_id} "
								f"for {proposal.order.quantity} {self._trading_pair.base_asset} @ "
								f"{proposal.order.price} {self._trading_pair.quote_asset}.")
			self.start_tracking_order(proposal.order)
			self._connector.submit_order(proposal.order)

	def update_strategy_state(self):
		mid_price = self.mid_price or s_decimal_zero
		current_bid_price = self._pmm_bids.best_price() or s_decimal_zero
		current_ask_price = self._pmm_asks.best_price() or s_decimal_zero
		current_bid_qty = self._pmm_bids.available_quote() or s_decimal_zero
		current_ask_qty = self._pmm_asks.available_quote() or s_decimal_zero
		current_bid_depth = self._pmm_bids.total_quote() or s_decimal_zero
		current_ask_depth = self._pmm_asks.total_quote() or s_decimal_zero
		inventory_amount = self.inventory_amount() or s_decimal_zero

		if self.current_inventory != inventory_amount:
			self.logger().info(f"Inventory changed from {self.current_inventory} to {inventory_amount}.")
			self.current_inventory = inventory_amount
		if self.mid_price != mid_price:
			self.logger().info(f"Mid price changed from {self.mid_price} to {mid_price}.")
			self.mid_price = mid_price
		if self.current_bid_price != current_bid_price:
			self.logger().info(f"Bid price changed from {self.current_bid_price} to {current_bid_price}.")
			self.current_bid_price = current_bid_price
		if self.current_ask_price != current_ask_price:
			self.logger().info(f"Ask price changed from {self.current_ask_price} to {current_ask_price}.")
			self.current_ask_price = current_ask_price
		if self.current_bid_qty != current_bid_qty:
			self.logger().info(f"Bid qty changed from {self.current_bid_qty} to {current_bid_qty}.")
			self.current_bid_qty = current_bid_qty
		if self.current_ask_qty != current_ask_qty:
			self.logger().info(f"Ask qty changed from {self.current_ask_qty} to {current_ask_qty}.")
			self.current_ask_qty = current_ask_qty
		if self.current_bid_depth != current_bid_depth:
			self.logger().info(f"Bid depth changed from {self.current_bid_depth} to {current_bid_depth}.")
			self.current_bid_depth = current_bid_depth
		if self.current_ask_depth != current_ask_depth:
			self.logger().info(f"Ask depth changed from {self.current_ask_depth} to {current_ask_depth}.")
			self.current_ask_depth = current_ask_depth

		self.hanging_orders_tracker.advance_epoch()
		self.cancel_orders()

	def should_fill_order(self, order: LimitOrder, trades: List[Dict[str, any]]) -> bool:
		order_age_seconds = order_age(order)
		if order_age_seconds > self.cancel_order_wait_time():
			self.logger().info(f"Cancelling order {order.client_order_id} due to order age {order_age_seconds} s.")
			return False
		return True

	def execute_sell(self, asset_amount: Decimal, order_id: str):
		current_price: Decimal = self.current_ask_price or s_decimal_zero
		mid_price: Decimal = self.mid_price or s_decimal_zero
		target_price: Decimal = current_price - self.order_level_spread * self._trading_pair.increment
		order_price: Decimal = max(target_price, mid_price - self.ask_spread * self._trading_pair.increment)
		order_price = self.quantize_order_price(order_price, False)

		if self._exchange.ready and self.should_place_order():
			quote_amount: Decimal = asset_amount * order_price
			if quote_amount < self._min_order_size:
				self.logger().info(f"Quote amount {quote_amount} is lower than minimum order size {self._min_order_size}. "
									f"Selling {asset_amount} {self._trading_pair.base_asset} cancelled.")
				return
			order = LimitOrder(
				client_order_id=order_id,
				trading_pair=self._trading_pair,
				is_buy_order=False,
				base_currency_amount=asset_amount,
				quote_currency_amount=quote_amount,
				order_type=OrderType.LIMIT,
				price=order_price
			)
			self.place_orders([Proposal(order, True)])
		else:
			self.logger().info("sell locked: exchange status or order limits")

	def execute_buy(self, asset_amount: Decimal, order_id: str):
		current_price: Decimal = self.current_bid_price or s_decimal_zero
		mid_price: Decimal = self.mid_price or s_decimal_zero
		target_price: Decimal = current_price + self.order_level_spread * self._trading_pair.increment
		order_price: Decimal = min(target_price, mid_price + self.bid_spread * self._trading_pair.increment)
		order_price = self.quantize_order_price(order_price, True)

		if self._exchange.ready and self.should_place_order():
			quote_amount: Decimal = asset_amount * order_price
			if quote_amount < self._min_order_size:
				self.logger().info(f"Quote amount {quote_amount} is lower than minimum order size {self._min_order_size}. "
									f"Buying {asset_amount} {self._trading_pair.base_asset} cancelled.")
				return
			order = LimitOrder(
				client_order_id=order_id,
				trading_pair=self._trading_pair,
				is_buy_order=True,
				base_currency_amount=asset_amount,
				quote_currency_amount=quote_amount,
				order_type=OrderType.LIMIT,
				price=order_price
			)
			self.place_orders([Proposal(order, True)])
		else:
			self.logger().info("buy locked: exchange status or order limits")

	def create_order_pairs(self, asset_amount: Decimal, is_buy: bool) -> List[CreatedPairOfOrders]:
		if not self.ready_for_new_orders():
			return []

		opposite_side_book: pd.DataFrame = self.order_book().opposite_side()
		if opposite_side_book is None or opposite_side_book.empty:
			return []

		current_price: Decimal = opposite_side_book.index[0]
		mid_price: Decimal = self.mid_price or s_decimal_zero
		spread: Decimal = self.ask_spread if is_buy else self.bid_spread
		new_price: Decimal = current_price + spread * self._trading_pair.increment * (-1 if is_buy else 1)
		new_price = self.quantize_order_price(new_price, is_buy)

		order_size = asset_amount / self.order_levels
		orders = []
		for i in range(self.order_levels):
			order_amount: Decimal = order_size * (i + 1)
			quote_amount: Decimal = order_amount * new_price
			if quote_amount < self._min_order_size:
				break
			order_type: OrderType = OrderType.LIMIT
			order = LimitOrder(
				client_order_id=self.get_new_order_id(),
				trading_pair=self._trading_pair,
				is_buy_order=is_buy,
				base_currency_amount=order_amount,
				quote_currency_amount=quote_amount,
				order_type=order_type,
				price=new_price
			)
			orders.append(order)

		return [CreatedPairOfOrders(
			buy_order=None if is_buy else orders,
			sell_order=orders if is_buy else None
		)]

	def get_active_orders(self) -> List[LimitOrder]:
		active_orders: List[LimitOrder] = super().get_active_orders()
		active_order_ids: List[str] = [o.client_order_id for o in active_orders]
		pure_mm_orders: List[LimitOrder] = self._order_tracker.get_orders()
		pure_mm_active_order_ids: List[str] = [o.client_order_id for o in pure_mm_orders if o.is_active]
		result: List[LimitOrder] = [o for o in active_orders if o.client_order_id in pure_mm_active_order_ids]
		result.extend([o for o in pure_mm_orders if o.client_order_id not in active_order_ids])
		return result

	def place_orders(self, proposals: List[Proposal]):
		hanging_orders: Dict[str, LimitOrder] = self._hanging_orders_tracker.hanging_orders
		open_orders: List[LimitOrder] = [o for o in self.get_active_orders() if o.client_order_id not in hanging_orders]
		cancelled_order_ids: List[str] = [o.client_order_id for o in self._exchange.cancel_all(self.get_tracking_pair())]
		cancelled_orders: List[LimitOrder] = [o for o in open_orders if o.client_order_id in cancelled_order_ids]

		all_order_sizes: Dict[Decimal, Decimal] = {}
		for proposal in proposals:
			if proposal.is_buy and proposal.order.price <= s_decimal_zero:
				continue
			trading_pair = proposal.order.trading_pair
			base_asset = trading_pair.base_asset
			quote_asset = trading_pair.quote_asset
			base_amount = proposal.order.base_currency_amount
			quote_amount = proposal.order.quote_currency_amount

			if proposal.is_buy:
				if quote_asset not in all_order_sizes:
					all_order_sizes[quote_asset] = s_decimal_zero
				all_order_sizes[quote_asset] += quote_amount
			else:
				if base_asset not in all_order_sizes:
					all_order_sizes[base_asset] = s_decimal_zero
				all_order_sizes[base_asset] += base_amount

		current_orders = open_orders + list(hanging_orders.values())
		current_order_sizes = self._order_size_estimator.get_order_sizes(current_orders)

		existing_buy_base_amount = current_order_sizes.get((True, base_asset), s_decimal_zero)
		existing_sell_quote_amount = current_order_sizes.get((False, quote_asset), s_decimal_zero)

		buy_base_amount = all_order_sizes.get(quote_asset, s_decimal_zero) / proposals[0].order.price if proposals else s_decimal_zero
		sell_quote_amount = all_order_sizes.get(base_asset, s_decimal_zero) if proposals else s_decimal_zero

		buy_target_base_amount = min(buy_base_amount, self.max_order_size - existing_buy_base_amount)
		sell_target_quote_amount = min(sell_quote_amount, self.max_order_size - existing_sell_quote_amount)

		total_buy_amount = buy_target_base_amount + existing_buy_base_amount
		total_sell_amount = sell_target_quote_amount + existing_sell_quote_amount

		if total_buy_amount < s_decimal_zero or total_sell_amount < s_decimal_zero:
			return

		buy_proposals = []
		sell_proposals = []

		if total_buy_amount > s_decimal_zero:
			buy_proposals = self.create_order_pairs(total_buy_amount, True)
			buy_proposals = self.filter_orders_too_far_from_price(buy_proposals, True)
			buy_proposals = self.filter_duplicate_orders(buy_proposals, current_orders)
			buy_proposals = self.filter_balance_orders(buy_proposals, False, base_asset, existing_buy_base_amount, max_amount=self.max_order_size)
			buy_proposals = self.filter_balance_orders(buy_proposals, True, quote_asset, existing_sell_quote_amount, max_amount=buy_target_base_amount * proposals[0].order.price)

		    if total_sell_amount > s_decimal_zero:
				sell_proposals = self.create_order_pairs(total_sell_amount, False)
				sell_proposals = self.filter_orders_too_far_from_price(sell_proposals, False)
				sell_proposals = self.filter_duplicate_orders(sell_proposals, current_orders)
				sell_proposals = self.filter_balance_orders(sell_proposals, True, base_asset, existing_buy_base_amount, max_amount=sell_target_quote_amount / proposals[0].order.price)
				sell_proposals = self.filter_balance_orders(sell_proposals, False, quote_asset, existing_sell_quote_amount, max_amount=self.max_order_size)

			if not buy_proposals and not sell_proposals:
				self.logger().info("All bids and asks are too far from the mid price. Hanging tight.")
				return

			if not proposals:
				open_bid_count = len([o for o in self.get_active_orders() if o.is_buy])
				open_ask_count = len([o for o in self.get_active_orders() if o.is_sell])
				bid_enabled = open_bid_count < self._market_info.limit_order_count_limit
				ask_enabled = open_ask_count < self._market_info.limit_order_count_limit

				if buy_proposals and bid_enabled:
					buy_order_ids = [o.client_order_id for o in open_orders if o.is_buy]
					buy_proposals = self.filter_limit_orders(buy_proposals, buy_order_ids, self.max_order_age)
					if buy_proposals:
						self.logger().info(f"Sending {len(buy_proposals)} new buy limit orders:\n{map_df_to_str(pd.DataFrame([p.order.to_dict() for p in buy_proposals]))}")
						buy_orders = [p.order for p in buy_proposals]
						for order in buy_orders:
							self._order_tracker.add_order(order)
						self._hanging_orders_tracker.add_hanging_orders(CreatedPairOfOrders.from_order_list(buy_orders))
						self._exchange.buy(self.get_tracking_pair(), buy_orders)
				else:
					self.logger().debug("max_active_orders limit reached for buy orders or no buy proposals")
				if sell_proposals and ask_enabled:
					sell_order_ids = [o.client_order_id for o in open_orders if o.is_sell]
					sell_proposals = self.filter_limit_orders(sell_proposals, sell_order_ids, self.max_order_age)
					if sell_proposals:
						self.logger().info(f"Sending {len(sell_proposals)} new sell limit orders:\n{map_df_to_str(pd.DataFrame([p.order.to_dict() for p in sell_proposals]))}")
						sell_orders = [p.order for p in sell_proposals]
						for order in sell_orders:
							self._order_tracker.add_order(order)
						self._hanging_orders_tracker.add_hanging_orders(CreatedPairOfOrders.from_order_list(sell_orders))
						self._exchange.sell(self.get_tracking_pair(), sell_orders)
				else:
					self.logger().debug("max_active_orders limit reached for sell orders or no sell proposals")
			else:
				self.logger().info(f"Skipping hanging orders as the bot is in Proposal mode.")
    
	def create_order_pairs(self, total_amount: Decimal, is_buy: bool) -> List[Proposal]:
        """
        This method creates a list of orders using the data in self._parameters and the
        total amount to be spread evenly on both sides of the order book.
        """
        base_asset = self._market_info.base_asset
        quote_asset = self._market_info.quote_asset
        mid_price = self._asset_price_delegate.get_mid_price()
        spread = self._parameters.bid_spread_percentage / Decimal('100')
        total_orders = self._parameters.order_levels if is_buy else self._parameters.order_levels_sell
        order_price = mid_price * (s_decimal_one + (spread if is_buy else -spread))
        order_amount = total_amount / total_orders
        order_size_tiers = self._parameters.order_level_amount
        tier_index = 0
        proposal_list = []
        for i in range(total_orders):
            if order_size_tiers and tier_index < len(order_size_tiers) and order_size_tiers[tier_index] <= order_amount:
                order_amount = order_size_tiers[tier_index]
                tier_index += 1
            proposal_list.append(Proposal(LimitOrder(self.get_tracking_pair(), is_buy, order_price, order_amount)))
            order_price *= (s_decimal_one + (spread * (s_decimal_neg_one if is_buy else s_decimal_one)))
        return proposal_list

    def place_orders(self) -> bool:
        """
        Creates a new set of buy and sell orders and places them with the connector.
        """
        mid_price = self._asset_price_delegate.get_mid_price()
        current_bid_price, current_bid_amount = self._order_book_bid_spread.get_price_and_liquidity_at(0)
        current_ask_price, current_ask_amount = self._order_book_ask_spread.get_price_and_liquidity_at(0)
        buy_proposals = []
        sell_proposals = []
        remaining_sell_amount = remaining_buy_amount = s_decimal_zero
        inventory_skew = self._inventory_cost_price_delegate.get_cost_price_skew(mid_price)
        target_skew = self._parameters.inventory_target_base_pct / Decimal('100')
        total_orders = self._parameters.order_levels
        total_sell_amount = total_buy_amount = s_decimal_zero

        # Calculate target base amount based on inventory target percentage.
        target_base_pct = self._parameters.inventory_target_base_pct / Decimal('100')
        target_base_asset_amount = self._market_info.total_base_asset_amount * target_base_pct

        # Calculate current base asset amount and inventory skew.
        current_base_asset_amount = self._market_info.base_asset_balance
        available_quote_asset_amount = self._market_info.quote_asset_balance
        available_quote_asset_amount -= self._hanging_orders_tracker.get_locked_quote_amount()
        available_quote_asset_amount -= self._active_order_tracker.get_locked_quote_amount()

        if current_base_asset_amount > s_decimal_zero:
            base_asset_ratio = current_base_asset_amount / self._market_info.total_balances
            current_skew = inventory_skew / base_asset_ratio
        else:
            current_skew = s_decimal_zero

        # Decide how much we need to buy/sell to reach target base asset amount.
        target_base_asset_delta = target_base_asset_amount - current_base_asset_amount

        if target_base_asset_delta > s_decimal_zero:
            # Buy base asset to reach target.
            buy_proposals = self.create_order_pairs(target_base_asset_delta, True)
            remaining_buy_amount = target_base_asset_delta
        elif target_base_asset_delta < s_decimal_zero:
            # Sell base asset to reach target.
            sell_proposals = self.create_order_pairs(-target_base_asset_delta, False)
            remaining_sell_amount = -target_base_asset_delta

        # If there are still orders left, use them for continuous orders.
        if self._active_order_tracker.has_pending_orders():
            last_buy_order = self._active_order_tracker.get_last_order(True)
            last_sell_order = self._active_order_tracker.get_last_order(False)
            remaining_buy_amount += sum(o.quantity for o in self._active_order_tracker.get_orders(True))
            remaining_sell_amount += sum(o.quantity for o in self._active_order_tracker.get_orders(False))
            buy_proposals += self.create_order_pairs(remaining_buy_amount, True)
            sell_proposals += self.create_order_pairs(remaining_sell_amount, False)

        # If there are still orders left, use them for order levels.
        if len(buy_proposals) < total_orders and remaining_buy_amount > s_decimal_zero:
            buy_proposals += self.create_order_pairs(remaining_buy_amount, True)
        if len(sell_proposals) < total_orders and remaining_sell_amount > s_decimal_zero:
            sell_proposals += self.create_order_pairs(remaining_sell_amount, False)

        # Place orders
        if total_sell_amount > s_decimal_zero:
            sell_proposals = self.create_order_pairs(total_sell_amount, False)

        if len(buy_proposals) > 0:
            self.logger().info(f"Creating {len(buy_proposals)} buy orders.")
            for bp in buy_proposals:
                order_id = self.place_order(OrderType.LIMIT, bp.price, bp.size, True)
                if order_id is not None:
                    self._active_order_tracker.add_order(LimitOrder(self._market_info.trading_pair,
                                                                     OrderType.LIMIT,
                                                                     bp.price,
                                                                     bp.size,
                                                                     self._creation_timestamp,
                                                                     order_id))
            self._hanging_orders_tracker.track_orders(HangingOrdersTracker.CONNECTION_RETRY_INTERVAL_SECONDS)
            self._last_buy_order_price = buy_proposals[0].price

        if len(sell_proposals) > 0:
            self.logger().info(f"Creating {len(sell_proposals)} sell orders.")
            for sp in sell_proposals:
                order_id = self.place_order(OrderType.LIMIT, sp.price, sp.size, False)
                if order_id is not None:
                    self._active_order_tracker.add_order(LimitOrder(self._market_info.trading_pair,
                                                                     OrderType.LIMIT,
                                                                     sp.price,
                                                                     sp.size,
                                                                     self._creation_timestamp,
                                                                     order_id))
            self._hanging_orders_tracker.track_orders(HangingOrdersTracker.CONNECTION_RETRY_INTERVAL_SECONDS)
            self._last_sell_order_price = sell_proposals[0].price

        self.logger().info(f"Active buys = {self._active_order_tracker.buy_order_count}, "
                            f"active sells = {self._active_order_tracker.sell_order_count}")
        return True

    def create_order_pairs(self, total_order_amount: Decimal, is_buy: bool) -> List[Proposal]:
        order_pairs = []
        spreads = self._buy_spreads if is_buy else self._sell_spreads
        base_prices = self._buy_levels if is_buy else self._sell_levels
        order_level_sizes = self._buy_order_level_sizes if is_buy else self._sell_order_level_sizes
        order_level_spreads = self._buy_order_level_spreads if is_buy else self._sell_order_level_spreads
        total_order_count = len(base_prices) + self._continuous_orders
        order_sizes = calculate_order_sizes(total_order_amount, total_order_count, order_level_sizes)

        if is_buy:
            order_sizes = order_sizes[::-1]

        for i, base_price in enumerate(base_prices):
            if i >= len(order_sizes):
                break
            order_size = order_sizes[i]
            order_pair = self.calculate_order_pair(base_price, order_size, spreads, order_level_spreads)
            order_pairs.append(order_pair)

        if len(order_sizes) > len(base_prices):
            continuous_order_sizes = order_sizes[len(base_prices):]
            continuous_order_spreads = self._continuous_order_spreads
            for i in range(self._continuous_orders):
                if i >= len(continuous_order_sizes):
                    break
                order_size = continuous_order_sizes[i]
                order_level_spread = continuous_order_spreads[i % len(continuous_order_spreads)]
                base_price = order_pairs[-1].price
                order_pair = self.calculate_order_pair(base_price, order_size, spreads, [order_level_spread])
                order_pairs.append(order_pair)

        return order_pairs

	def should_cancel(self, order: LimitOrder) -> bool:
		return order_age(order) > self.cancel_order_wait_time \
			   or order.traded_amount is not None and order.filled_amount / order.traded_amount >= self.cancel_order_pct_threshold

	def should_update(self, order: LimitOrder) -> bool:
		return order_age(order) > self.order_update_wait_time


    async def place_orders(self, proposals: List[Proposal]) -> List[LimitOrder]:
        # Calculate the number of orders to place based on the current order count
        current_order_count = self._order_tracker.count_orders()
        max_order_count = self._max_order_count
        available_order_count = max_order_count - current_order_count

        # Sort the proposals by their price, in ascending order for buy orders, and descending order for sell orders.
        is_buy = proposals[0].is_buy
        sorted_proposals = sorted(proposals, key=lambda p: p.price, reverse=is_buy)

        # Filter out any proposals that would cause us to exceed the available order count.
        proposals_to_place = []
        for proposal in sorted_proposals:
            if proposal.order_count > available_order_count:
                break
            proposals_to_place.append(proposal)
            available_order_count -= proposal.order_count

        # Place the orders and update the order tracker.
        orders = []
        for proposal in proposals_to_place:
            order = await self.place_order(proposal)
            if order is not None:
                orders.append(order)
                self._order_tracker.add_order(order)
        return orders

    async def cancel_orders(self, order_ids: List[str]):
        # Calculate the number of orders that will remain after cancelling.
        current_order_count = self._order_tracker.count_orders()
        max_order_count = self._max_order_count
        available_order_count = max_order_count - current_order_count
        remaining_order_count = max(available_order_count, self._min_order_count)

        # Sort the orders by their price, in ascending order for buy orders, and descending order for sell orders.
        orders = self._order_tracker.get_orders_by_id(order_ids)
        is_buy = orders[0].is_buy
        sorted_orders = sorted(orders, key=lambda o: o.price, reverse=is_buy)

        # Filter out any orders that would cause us to cancel all of our orders.
        orders_to_cancel = []
        for order in sorted_orders:
            if remaining_order_count <= 0:
                break
            if self.is_within_order_range(order):
                orders_to_cancel.append(order)
                remaining_order_count -= order.quantity

        # Cancel the orders and update the order tracker.
        for order in orders_to_cancel:
            await self.cancel_order(order)
            self._order_tracker.remove_order(order)

    async def tick(self, timestamp: float):
        # Get the current bid and ask prices.
        bid_price, ask_price = await self.get_bid_ask_prices()

        # Calculate the bid and ask spreads.
        bid_spread = self._bid_spread
        ask_spread = self._ask_spread
        order_level_spread = self._order_level_spread

        # Determine the order levels.
        if self._order_levels is None:
            order_levels = self.calculate_order_levels(bid_price, ask_price)
        else:
            order_levels = self._order_levels

        # Calculate the skew factors for the inventory.
        bid_skew, ask_skew = self.calculate_skew_factors()

        # Calculate the size of the orders to place at each order level.
        total_buy_amount = self.calculate_order_size(True)
        total_sell_amount = self.calculate_order_size(False)
        buy_proposals = []
        sell_proposals = []

        # Create the initial set of proposals.
        for i in range(len(order_levels)):
            order_level = order_levels[i]
            is_buy = i % 2 == 0
            price = order_level["price"]
            size = order_level["size"]

            # Apply the skew factor to the order size.
            if is_buy:
                size *= bid_skew
            else:
                size *= ask_skew

            # Determine the number of orders to place at this level.
            if is_buy:
                count = self._order_level_count[i // 2]
            else:
                count = self._order_level_count[(i - 1) // 2]

            # Determine the order spread for this level.
            if i == 0:
                spread = bid_spread
            elif i == len(order_levels) - 1:
                spread = ask_spread
            else:
                spread = order_level_spread

            # Calculate the order size and price for each order.
            for j in range(count):
                order_size = size / count
                order_price = self.calculate_order_price(price, is_buy, spread, j, count)

                # Create the proposal for this order.
                proposal = Proposal(is_buy, order_size, order_price)
                if is_buy:
                    buy_proposals.append(proposal)
                else:
                    sell_proposals.append(proposal)

        # Place the initial set of orders.
        if total_buy_amount > s_decimal_zero:
            buy_proposals = self.create_order_pairs(total_buy_amount, True)
            await self.place_orders(buy_proposals)

        if total_sell_amount > s_decimal_zero:
            sell_proposals = self.create_order_pairs(total_sell_amount, False)
            await self.place_orders(sell_proposals)

        # Update the hanging orders.
        self._hanging_orders_tracker.tick(self._order_tracker.orders)

        # Check if any orders need to be cancelled.
        if self.should_cancel():
            order_ids = self._hanging_orders_tracker.get_order_ids_to_cancel(self.cancel_order_wait_time)
            if len(order_ids) > 0:
                await self.cancel_orders(order_ids)

        # Check if any orders need to be updated.
        if self.should_update():
            order_ids = self._hanging_orders_tracker.get_order_ids_to_update(self.update_order_wait_time)
            if len(order_ids) > 0:
                await self.update_orders(order_ids)

    def create_order_pairs(self, amount: Decimal, is_buy: bool) -> List[CreatedPairOfOrders]:
        """
        Create a pair of orders at each order level.
        """
        order_pairs = []
        order_levels = self._order_levels

        # Determine the order levels.
        if order_levels is None:
            bid_price, ask_price = self._exchange.get_bid_ask_prices(self._market_trading_pair)
            order_levels = self.calculate_order_levels(bid_price, ask_price)

        # Calculate the skew factors for the inventory.
        bid_skew, ask_skew = self.calculate_skew_factors()

        # Calculate the size of the orders to place at each order level.
        total_amount = self.calculate_order_size(is_buy)

        # Create the set of proposals.
        proposals = []
        for i in range(len(order_levels)):
            order_level = order_levels[i]
            price = order_level["price"]
            size = order_level["size"]

            # Apply the skew factor to the order size.
            if is_buy:
                size *= bid_skew
            else:
                size *= ask_skew

            # Determine the number of orders to place at this level.
            if is_buy:
                count = self._order_level_count[i // 2]
            else:
                count = self._order_level_count[(i - 1) // 2]

            # Determine the order spread for this level.
            if i == 0:
                spread = self._bid_spread
            elif i == len(order_levels) - 1:
                spread = self._ask_spread
            else:
                spread = self._order_level_spread

            # Calculate the order size and price for each order.
            for j in range(count):
                order_size = size / count
                order_price = self.calculate_order_price(price, is_buy, spread, j, count)

                # Create the proposal for this order.
                proposal = Proposal(is_buy, order_size, order_price)
                proposals.append(proposal)

        # Calculate the total amount of the orders to place.
        total_size = sum([p.size for p in proposals])
        total_size = min(total_size, amount)
        total_amount = min(total_amount, amount)

        # Determine the fraction of the total amount to place at each order level.
        fractions = []
        for proposal in proposals:
            if total_size > s_decimal_zero:
                fraction = proposal.size / total_size
                fractions.append(fraction)
            else:
                fractions.append(0)

        # Calculate the actual amount to place at each order level.
        amounts = [total_amount * fraction for fraction in fractions]

        # Create the order pairs.
        for i in range(len(proposals)):
            proposal = proposals[i]
            amount = amounts[i]
            order_price = proposal.price
            order_size = amount / order_price
            order_type = OrderType.LIMIT

            if is_buy:
                price_type = PriceType.BEST_PRICE
            else:
                price_type = PriceType.EXACT_PRICE

            # Create the pair of orders for this proposal.
            pair = CreatedPairOfOrders()
            pair.primary_order = LimitOrder(self._market_trading_pair,
                                             order_type,
                                             order_size,
                                             order_price,
                                             self._order_time_in_force,
                                             exchange_time=self.current_timestamp)
            pair.secondary_order = LimitOrder(self._market_trading_pair,
                                               order_type,
                                               order_size,
                                               order_price,
                                               self._order_time_in_force,
                                               exchange_time=self.current_timestamp)
            pair.is_buy = is_buy
            pair.price = proposal.price
            pair.size = proposal.size
            pair.fee_asset = self._exchange.get_fee_asset(self._market_trading_pair)
            pair.fee_paid = s_decimal_zero
            pair.has_been_filled = False
            pair.time_to_live = self.time_to_live

            order_pairs.append(pair)

        return order_pairs

    def create_order(self,
                     proposal: Proposal,
                     is_buy: bool,
                     order_type: OrderType) -> LimitOrder:
        order_size = proposal.size
        order_price = proposal.price
        if is_buy:
            order_price = self.adjust_buy_price(order_price)
        else:
            order_price = self.adjust_sell_price(order_price)

        return LimitOrder(self._market_trading_pair,
                           order_type,
                           order_size,
                           order_price,
                           self._order_time_in_force,
                           exchange_time=self.current_timestamp)

    def adjust_buy_price(self, price: Decimal) -> Decimal:
        return price * (s_decimal_one + self._bid_spread)

    def adjust_sell_price(self, price: Decimal) -> Decimal:
        return price * (s_decimal_one - self._ask_spread)

    def calculate_bid_ask_spreads(self,
                                  raw_bid_price: Decimal,
                                  raw_ask_price: Decimal,
                                  skew: Optional[Decimal] = None) -> Tuple[Decimal, Decimal]:
        if skew is None:
            skew = self._inventory_skew_calculator.current_skew
        if skew > Decimal("1"):
            skew = Decimal("1")
        elif skew < Decimal("-1"):
            skew = Decimal("-1")

        bid_spread = self._bid_spread + skew * self._bid_spread * self._inventory_range_multiplier
        ask_spread = self._ask_spread - skew * self._ask_spread * self._inventory_range_multiplier

        adjusted_bid_price = self.adjust_buy_price(raw_bid_price)
        adjusted_ask_price = self.adjust_sell_price(raw_ask_price)

        min_ask_price = adjusted_bid_price * (s_decimal_one + bid_spread)
        max_bid_price = adjusted_ask_price * (s_decimal_one - ask_spread)

        bid_spread = (max_bid_price - raw_bid_price) / raw_bid_price
        ask_spread = (raw_ask_price - min_ask_price) / raw_ask_price

        return bid_spread, ask_spread

    def create_order_pairs(self,
                           size: Decimal,
                           is_buy: bool) -> List[CreatedPairOfOrders]:
        order_pairs = []
        order_levels = self._order_levels
        order_level_spread = self._order_level_spread

        if len(order_levels) == 0:
            order_pairs.append(self.create_order_pair(size, is_buy))
            return order_pairs

        remainder = size
        for i in range(len(order_levels)):
            level_size = order_levels[i] * size
            if remainder < level_size:
                order_pairs.append(self.create_order_pair(remainder, is_buy))
                remainder = s_decimal_zero
                break
            order_pairs.append(self.create_order_pair(level_size, is_buy))
            remainder -= level_size
            if remainder <= s_decimal_zero:
                break

        if remainder > s_decimal_zero:
            last_price = order_pairs[-1].orders[0].price
            last_size = order_pairs[-1].orders[0].quantity
            for i in range(len(order_levels), len(order_levels) + self._continuous_order_limit):
                price = last_price * (s_decimal_one + (i - len(order_levels) + 1) * order_level_spread)
                size = min(remainder, last_size)
                if size < self._min_order_size:
                    break
                order_pair = self.create_order_pair(size, is_buy, price)
                order_pairs.append(order_pair)
                remainder -= size
                if remainder <= s_decimal_zero:
                    break

        return order_pairs

    def create_order_pair(self,
                          size: Decimal,
                          is_buy: bool,
                          price: Optional[Decimal] = None) -> CreatedPairOfOrders:
        if price is None:
            price = self.get_price(is_buy)

        base_asset_amount = size * price
        if is_buy:
            quote_asset_amount = size
            price *= (s_decimal_one - self._bid_spread)
        else:
            quote_asset_amount = base_asset_amount
            price *= (s_decimal_one + self._ask_spread)

        base_asset_amount = Decimal(str(base_asset_amount))
        quote_asset_amount = Decimal(str(quote_asset_amount))
        orders = [
            LimitOrder(self._market_trading_pair_tuple.trading_pair,
                       OrderType.LIMIT,
                       base_asset_amount,
                       price,
                       TradeType.BUY if is_buy else TradeType.SELL),
            LimitOrder(self._market_trading_pair_tuple.trading_pair,
                       OrderType.LIMIT,
                       quote_asset_amount,
                       price,
                       TradeType.SELL if is_buy else TradeType.BUY)
        ]

        return CreatedPairOfOrders(orders, True)

    def get_price(self, is_buy: bool) -> Decimal:
        trading_pair = self._market_trading_pair_tuple.trading_pair
        mid_price = self._market_info.get_mid_price(trading_pair)
        bid, ask = self._market_info.get_price_for_order(trading_pair, is_buy)
        price = mid_price if bid is None or ask is None else (bid + ask) / 2
        return Decimal(str(price))

    def validate_market_trading_pair(self) -> None:
        self._market_trading_pair_tuple = MarketTradingPairTuple.convert_from_strings(self.market, self.trading_pair)
        base_asset, quote_asset = self._market_trading_pair_tuple.base_asset, self._market_trading_pair_tuple.quote_asset
        if base_asset == quote_asset:
            raise ValueError(f"The PureMarketMaking strategy does not support trading the same asset ({base_asset})")

        if self._market_info.trading_pairs is not None and self.trading_pair not in self._market_info.trading_pairs:
            raise ValueError(f"{self.trading_pair} is not a valid trading pair on {self.market.name}")

    def create_order_pairs(self, total_amount: Decimal, is_buy: bool) -> List[CreatedPairOfOrders]:
        current_price = self.get_price(is_buy)
        price_band = self._price_band
        buy_amount = s_decimal_zero
        sell_amount = s_decimal_zero
        buy_orders = []
        sell_orders = []
        while total_amount > s_decimal_zero:
            next_price = price_band.get_price(is_buy)
            amount_remaining = total_amount
            if is_buy:
                if next_price > current_price:
                    break
            else:
                if next_price < current_price:
                    break

            order_size = self._order_levels.pop(0)
            if order_size > amount_remaining:
                order_size = amount_remaining
            amount_remaining -= order_size

            price = max(next_price, current_price * (Decimal(1) - price_band.width))
            price = min(price, current_price * (Decimal(1) + price_band.width))

            order = LimitOrder(
                self.market_info,
                self.market,
                self._market_trading_pair_tuple.trading_pair,
                OrderType.LIMIT,
                is_buy,
                price,
                order_size,
                self._order_step_size,
                self._order_optimization_depth,
                self._expiration_seconds
            )
            if is_buy:
                buy_orders.append(order)
                buy_amount += order_size
            else:
                sell_orders.append(order)
                sell_amount += order_size

            if amount_remaining <= s_decimal_zero or not self._order_levels:
                break

            # Move price band to next level
            price_band.move(is_buy)
        self._order_levels = self._original_order_levels[:]
        self._order_levels_age += 1
        return [CreatedPairOfOrders(buy_orders, buy_amount), CreatedPairOfOrders(sell_orders, sell_amount)]

    def place_orders(self, buy_proposals: List[CreatedPairOfOrders], sell_proposals: List[CreatedPairOfOrders]):
        now = self.current_timestamp
        if len(buy_proposals) > 0:
            self.logger().info(f"Creating {len(buy_proposals)} buy limit order(s).")
        if len(sell_proposals) > 0:
            self.logger().info(f"Creating {len(sell_proposals)} sell limit order(s).")

        for buy_order in buy_proposals:
            for order in buy_order.orders:
                self.add_limit_order(order)
                self.trigger_event(self.OnBuyOrderCreated, order)

        for sell_order in sell_proposals:
            for order in sell_order.orders:
                self.add_limit_order(order)
                self.trigger_event(self.OnSellOrderCreated, order)

        if self._last_bid_price != self._last_ask_price:
            self._last_bid_price = self._last_ask_price = (buy_proposals[0].orders[0].price + sell_proposals[0].orders[0].price) / 2

        self._last_order_create_timestamp = now

    def create_order_pairs(self, total_amount: Decimal, is_buy: bool) -> List[CreatedPairOfOrders]:
        """
        Generate buy/sell order pair(s) for a given total order amount.
        """
        side = "buy" if is_buy else "sell"
        min_order_amount = self._market_info.min_order_size
        order_size = max(self._order_size_floor, min_order_amount)
        spreads = self.order_spreads
        spread = spreads[0] if is_buy else spreads[1]
        price = self._asset_price_delegate.get_price(spread * s_decimal_neg_one if is_buy else spread)

        if self._order_levels == 1:
            return [CreatedPairOfOrders([self.create_order(price, order_size, side)])]

        if total_amount < order_size * self._order_levels:
            order_size = total_amount / self._order_levels
            order_size = self.round_order_amount(order_size)

        orders = []
        remaining_amount = total_amount

        for i in range(self._order_levels):
            if remaining_amount <= s_decimal_zero:
                break

            order_amount = min(remaining_amount, order_size)
            remaining_amount -= order_amount
            order_amount = self.round_order_amount(order_amount)

            # Increment price based on order book
            order_price = self.calculate_order_price(price, i, is_buy)

            orders.append(self.create_order(order_price, order_amount, side))

        return [CreatedPairOfOrders(orders)]

    def create_order(self, price: Decimal, amount: Decimal, side: str) -> LimitOrder:
        """
        Create a limit order with the given price and amount.
        """
        order_type = OrderType.LIMIT
        order_id = self._order_id_nonce.get_next_id()
        trading_pair = self._market_info.trading_pair
        return LimitOrder(trading_pair, order_type, price, amount, self._leverage, order_id=order_id, side=side)

    def calculate_order_price(self, current_price: Decimal, order_level: int, is_buy: bool) -> Decimal:
        """
        Calculate order price for a given order level.
        """
        spreads = self.order_spreads
        spread = spreads[0] if is_buy else spreads[1]
        sign = s_decimal_neg_one if is_buy else Decimal(1)
        order_price = current_price + sign * order_level * spread
        return self.round_price(order_price)

