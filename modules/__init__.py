"""
Quantitative Backtesting Platform Module Package

Contains the following core modules:
- Data Management Module (DataManagementModule)
- Factor Analysis Module (FactorAnalysisModule)
- Strategy Backtesting Module (BacktestingModule)
- Result Management Module (ResultManager)
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"

from .data_management import DataManagementModule
from .factor_analysis import FactorAnalysisModule
from .backtesting import BacktestingModule
from .result_manager import ResultManager

__all__ = [
    "DataManagementModule",
    "FactorAnalysisModule",
    "BacktestingModule",
    "ResultManager"
]
