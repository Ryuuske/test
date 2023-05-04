#!/usr/bin/env python

from hummingbot.strategy.pure_scalping.pure_scalping import PureScalpingStrategy
from .inventory_cost_price_delegate import InventoryCostPriceDelegate

__all__ = [
    PureScalpingStrategy.__name__,
    InventoryCostPriceDelegate.__name__,
]
