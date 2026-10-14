from typing import List, Callable
from .models import OHLCV

# Functional types representing our ports
GetOhlcvFn = Callable[[str, str, int], List[OHLCV]]
GetLatestPriceFn = Callable[[str], float]
ValidateSymbolFn = Callable[[str], bool]
