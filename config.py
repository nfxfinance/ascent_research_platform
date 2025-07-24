#!/usr/bin/env python3

import os
from pathlib import Path

# System Configuration
class Config:
    """System configuration class"""

    # Application Information
    APP_NAME = "Quantitative Backtesting Platform"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Professional quantitative investment backtesting and factor analysis platform"

    # Path Configuration
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    REPORTS_DIR = BASE_DIR / "reports"
    MODULES_DIR = BASE_DIR / "modules"
    TEMP_DIR = BASE_DIR / "temp"

    # Create necessary directories
    for dir_path in [DATA_DIR, REPORTS_DIR, TEMP_DIR]:
        dir_path.mkdir(exist_ok=True)

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///quantitative_platform.db")

    # API Configuration
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
    WIND_API_KEY = os.getenv("WIND_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    # System Parameters
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB

    # Backtesting Parameters
    DEFAULT_INITIAL_CAPITAL = 1000000  # Default initial capital
    DEFAULT_COMMISSION = 0.0003  # Default commission rate
    DEFAULT_SLIPPAGE = 0.0001  # Default slippage

    # Factor Analysis Parameters
    DEFAULT_LOOKBACK_PERIOD = 252  # Default lookback period
    DEFAULT_REBALANCE_FREQ = "M"  # Default rebalance frequency

    # Risk Management Parameters
    MAX_DRAWDOWN_LIMIT = 0.2  # Maximum drawdown limit
    MAX_POSITION_LIMIT = 0.95  # Maximum position limit

    # Report Configuration
    REPORT_TEMPLATES = {
        "Factor Analysis": "factor_analysis_template.html",
        "Backtesting Analysis": "backtest_analysis_template.html",
        "Comprehensive Analysis": "comprehensive_analysis_template.html"
    }

    # Chart Configuration
    CHART_THEME = "plotly_white"
    CHART_HEIGHT = 400
    CHART_WIDTH = 800

    # Data Source Configuration
    SUPPORTED_FILE_TYPES = [".csv", ".xlsx", ".xls", ".json", ".parquet"]
    SUPPORTED_DATABASES = ["MySQL", "PostgreSQL", "SQLite", "Oracle", "SQL Server"]

    # Cache Configuration
    CACHE_TTL = 3600  # Cache expiration time (seconds)

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

    # Performance Configuration
    MAX_WORKERS = 4  # Maximum number of worker threads
    CHUNK_SIZE = 10000  # Data processing chunk size

    # Status Codes
    STATUS_CODES = {
        "SUCCESS": 200,
        "ERROR": 500,
        "NOT_FOUND": 404,
        "FORBIDDEN": 403
    }

# Constants Definition
class Constants:
    """Constants definition class"""

    # Trading Calendar
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5

    # Market Indices
    BENCHMARK_INDICES = {
        "CSI 300": "000300.SH",
        "CSI 500": "000905.SH",
        "ChiNext Index": "399006.SZ",
        "SSE 50": "000016.SH",
        "STAR 50": "000688.SH"
    }

    # Factor Types
    FACTOR_TYPES = {
        "Market Cap Factors": ["market_cap", "log_market_cap"],
        "Value Factors": ["pe_ratio", "pb_ratio", "ps_ratio"],
        "Growth Factors": ["revenue_growth", "profit_growth", "eps_growth"],
        "Profitability Factors": ["roe", "roa", "gross_margin"],
        "Quality Factors": ["debt_ratio", "current_ratio", "quick_ratio"],
        "Momentum Factors": ["momentum_1m", "momentum_3m", "momentum_6m"],
        "Volatility Factors": ["volatility_1m", "volatility_3m", "volatility_6m"]
    }

    # Technical Indicators
    TECHNICAL_INDICATORS = {
        "Moving Averages": ["SMA", "EMA", "WMA"],
        "Trend Indicators": ["MACD", "ADX", "Aroon"],
        "Oscillators": ["RSI", "Stochastic", "Williams %R"],
        "Volume Indicators": ["OBV", "VWAP", "Chaikin MF"],
        "Volatility Indicators": ["Bollinger Bands", "ATR", "Keltner Channel"]
    }

    # Risk Metrics
    RISK_METRICS = [
        "Total Return", "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
        "Maximum Drawdown", "Calmar Ratio", "Sortino Ratio", "Information Ratio",
        "VaR", "CVaR", "Beta", "Alpha", "Win Rate", "Profit/Loss Ratio"
    ]

    # Data Frequencies
    DATA_FREQUENCIES = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }

    # File Formats
    EXPORT_FORMATS = {
        "CSV": ".csv",
        "Excel": ".xlsx",
        "JSON": ".json",
        "Parquet": ".parquet",
        "HTML": ".html",
        "PDF": ".pdf"
    }

# Environment Configuration
class Environment:
    """Environment configuration class"""

    @staticmethod
    def is_development():
        """Check if it's development environment"""
        return os.getenv("ENVIRONMENT", "development") == "development"

    @staticmethod
    def is_production():
        """Check if it's production environment"""
        return os.getenv("ENVIRONMENT", "development") == "production"

    @staticmethod
    def get_env_var(key, default=None):
        """Get environment variable"""
        return os.getenv(key, default)

    @staticmethod
    def get_database_url():
        """Get database connection URL"""
        if Environment.is_production():
            return os.getenv("DATABASE_URL", Config.DATABASE_URL)
        else:
            return f"sqlite:///{Config.DATA_DIR}/dev.db"

# Global Configuration Instance
config = Config()
constants = Constants()
env = Environment()
