"""
因子分析模块 - 重构版本

这个模块被拆分为多个子模块以提高可维护性：
- parameter_parser: 参数解析和表单创建
- chart_renderer: 图表渲染和结果展示
- analysis_sections: 分析模块基类和实现
- data_processor: 数据处理和验证
- factor_analysis_module: 主模块类
"""

from .parameter_parser import ParameterParser
from .chart_renderer import ChartRenderer, create_download_mhtml_button
from .analysis_sections import AnalysisSection, AutoAnalysisSection
from .data_processor import DataProcessor
from .factor_analysis_module import FactorAnalysisModule

__all__ = [
    'ParameterParser',
    'ChartRenderer',
    'create_download_mhtml_button',
    'AnalysisSection',
    'AutoAnalysisSection',
    'DataProcessor',
    'FactorAnalysisModule'
]
