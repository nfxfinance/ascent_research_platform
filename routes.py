#!/usr/bin/env python3

"""
URL Routing Configuration for Quantitative Backtesting Platform
路由配置文件 - 量化回测平台
"""

# Module route mapping
# 模块路由映射
MODULE_ROUTES = {
    # Short route name: Full module name
    "files": "File Management",          # 文件管理模块
    "data": "Data Management",           # 数据管理模块
    "factor": "Factor Analysis",         # 因子分析模块
    "backtest": "Strategy Backtesting",  # 策略回测模块
    "result": "Result Management",       # 结果管理模块
    "portfolio": "Portfolio Optimization", # 投资组合优化模块 (预留)
    "usage": "Usage Guide"
}

# Reverse mapping for quick lookup
ROUTE_MODULES = {v: k for k, v in MODULE_ROUTES.items()}

# Route descriptions for documentation
ROUTE_DESCRIPTIONS = {
    "files": "File upload, management and database records",
    "data": "Data source integration, cleaning, storage and updates",
    "factor": "Factor mining, effectiveness validation, portfolio analysis",
    "backtest": "Strategy backtesting, performance analysis, risk metrics",
    "result": "Report generation, data export, result sharing",
    "portfolio": "Modern portfolio theory, risk-return analysis, asset allocation",
    "usage": "Usage guide for the platform"
}

# URL parameter names that can be used for routing
ROUTE_PARAMS = ['page', 'module', 'view']

# Default module when no route is specified
DEFAULT_MODULE = "Data Management"

# Route validation
def is_valid_route(route):
    """Check if route is valid"""
    return route.lower() in MODULE_ROUTES

def get_module_name(route):
    """Get full module name from route"""
    return MODULE_ROUTES.get(route.lower())

def get_route_name(module_name):
    """Get route name from full module name"""
    return ROUTE_MODULES.get(module_name)

# URL generation helpers
def generate_module_url(module_name, base_url="http://localhost:8501"):
    """Generate URL for specific module"""
    route = get_route_name(module_name)
    if route:
        return f"{base_url}/?page={route}"
    return base_url

def get_all_routes():
    """Get all available routes with descriptions"""
    return {
        route: {
            "module": module,
            "description": ROUTE_DESCRIPTIONS.get(route, ""),
            "url": f"/?page={route}"
        }
        for route, module in MODULE_ROUTES.items()
    }

# Route examples for documentation
ROUTE_EXAMPLES = {
    "data": [
        "/?page=data",
        "/?module=data",
        "/?view=data"
    ],
    "factor": [
        "/?page=factor",
        "/?module=factor",
        "/?view=factor"
    ],
    "backtest": [
        "/?page=backtest",
        "/?module=backtest",
        "/?view=backtest"
    ],
    "result": [
        "/?page=result",
        "/?module=result",
        "/?view=result"
    ]
}

# Route metadata
ROUTE_METADATA = {
    "files": {
        "icon": "📁",
        "color": "#9467bd",
        "priority": 0,
        "category": "Files"
    },
    "data": {
        "icon": "📊",
        "color": "#1f77b4",
        "priority": 1,
        "category": "Data"
    },
    "factor": {
        "icon": "🔍",
        "color": "#ff7f0e",
        "priority": 2,
        "category": "Analysis"
    },
    "backtest": {
        "icon": "📈",
        "color": "#2ca02c",
        "priority": 3,
        "category": "Testing"
    },
    "result": {
        "icon": "📋",
        "color": "#d62728",
        "priority": 4,
        "category": "Output"
    },
    "usage": {
        "icon": "📖",
        "color": "#9467bd",
        "priority": 5,
        "category": "Documentation"
    }
}

if __name__ == "__main__":
    # Print route configuration for testing
    print("Available Routes:")
    print("=" * 50)

    for route, info in get_all_routes().items():
        metadata = ROUTE_METADATA.get(route, {})
        print(f"{metadata.get('icon', '📄')} {route} -> {info['module']}")
        print(f"   URL: {info['url']}")
        print(f"   Description: {info['description']}")
        print(f"   Category: {metadata.get('category', 'Unknown')}")
        print()
