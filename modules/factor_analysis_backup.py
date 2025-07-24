#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import pickle
import base64
from io import StringIO
import logging
import traceback
import inspect
import ast
import typing
from typing import List, Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats
import io

# Set matplotlib backend for streamlit
matplotlib.use('Agg')

# --- User-Specified Module Loading ---
ROOTPATH = os.getenv("CA_QROOT")
if ROOTPATH:
    sys.path.append(ROOTPATH + '/python')
else:
    # Use relative path for lib imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ascentquant_python_path = os.path.join(project_root, 'AscentQuantMaster', 'python')
    if os.path.exists(ascentquant_python_path) and ascentquant_python_path not in sys.path:
        sys.path.insert(0, ascentquant_python_path)
    else:
        # Fallback to parent directory
        project_python_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_python_path not in sys.path:
            sys.path.insert(0, project_python_path)

from lib.lib_signal_perf import SignalPerf

import importlib
import sys

def load_ic_extensions():
    """动态加载IC扩展模块"""
    try:
        # 如果模块已经加载，重新加载它
        if 'lib.lib_signal_perf_ic_extensions' in sys.modules:
            importlib.reload(sys.modules['lib.lib_signal_perf_ic_extensions'])

        from lib.lib_signal_perf_ic_extensions import add_ic_extensions
        return add_ic_extensions
    except ImportError as e:
        logger.warning(f"无法导入IC扩展模块: {e}")
        return None

add_ic_extensions = load_ic_extensions()
from lib.utils import format_data, setup_logger

add_ic_extensions(SignalPerf)
LIB_AVAILABLE = True

setup_logger()
logger = logging.getLogger(__name__)


TYPE_MAPPING = {
    'signal_name': str,
    'signal_names': list,
    'lookfwd_day': float,
    'lookfwd_days': list,
    'sample_day': int,
    'sample_days': list,
    'sample_freq_days': int,
    'width': int,
    'height': int,
    'risk_adj': bool,
    'return_window': int,
    'max_lags': int,
    'labeling_method': str,
    'price_col': str,
    'signal_cols': list,
    'day': list,
    'window_size': int,
    'step_size': int,
    'min_periods': int,
    'quantile': float,
    'bins': int,
    'alpha': float,
    'threshold': float,
    'method': str,
    'freq': str,
    'period': int,
    'lag': int,
    'lags': list,
    'n_periods': int,
    'overlap': bool,
}


class ParameterParser:
    """智能参数解析器"""

    @staticmethod
    def infer_parameter_type(name: str, annotation: Any, default_val: Any) -> type:
        """推断参数类型"""
        # 优先使用显式类型注解
        if annotation and annotation != inspect.Parameter.empty:
            if hasattr(annotation, '__origin__'):
                # 处理泛型类型如List[int], Optional[str]等
                origin = annotation.__origin__
                if origin is list or origin is typing.List:
                    return list
                elif origin is typing.Union:
                    # Union类型，取第一个非None类型
                    for arg in annotation.__args__:
                        if arg != type(None):
                            return arg
            return annotation

        # 使用类型映射
        if name in TYPE_MAPPING:
            return TYPE_MAPPING[name]

        # 根据默认值推断
        if default_val is not None and default_val != inspect.Parameter.empty:
            return type(default_val)

        # 根据参数名模式推断
        if any(keyword in name.lower() for keyword in ['list', 'names', 'cols', 'days', 'lags', 'periods']):
            return list
        elif any(keyword in name.lower() for keyword in ['adj', 'risk', 'enable', 'show', 'plot']):
            return bool
        elif any(keyword in name.lower() for keyword in ['width', 'height', 'sample', 'window', 'step', 'bins', 'lag', 'period']):
            return int
        elif any(keyword in name.lower() for keyword in ['day', 'rate', 'threshold', 'alpha', 'quantile']):
            return float

        # 默认为字符串
        return str


    @staticmethod
    def get_function_signature(func) -> Dict[str, Any]:
        """获取函数签名信息"""
        params = inspect.signature(func).parameters
        return {
            name: {
                'annotation': param.annotation,
                'default': param.default if param.default is not inspect.Parameter.empty else None,
                'kind': param.kind
            }
            for name, param in params.items()
            if name not in ['self', 'save_plot', 'save_dir']  # 移除width和height限制，让用户可以设置
        }

    @staticmethod
    def create_form_widget(name: str, param_info: Dict, signal_list: List[str] = None,
                          default_signal: str = None) -> Any:
        """根据参数信息创建表单组件"""
        annotation = param_info['annotation']
        default_val = param_info['default']

        # 特殊处理信号名称
        if name == 'signal_name':
            if signal_list:
                idx = signal_list.index(default_signal) if default_signal in signal_list else 0
                return st.selectbox(f"📊 {name}", options=signal_list, index=idx, help="选择要分析的信号")
            else:
                return st.text_input(f"📊 {name}", value=str(default_val) if default_val else "", help="输入信号名称")

        # 特殊处理信号名称列表
        if name == 'signal_names' and signal_list:
            return st.multiselect(f"📊 {name}", options=signal_list, default=signal_list, help="选择要分析的信号列表")

        # 处理布尔类型
        if annotation is bool:
            return st.checkbox(f"✅ {name}", value=bool(default_val), help=f"是否启用{name}")

        # 处理Literal类型
        if hasattr(annotation, '__origin__') and str(annotation.__origin__) == 'typing.Literal':
            options = list(annotation.__args__)
            default_idx = options.index(default_val) if default_val in options else 0
            return st.selectbox(f"🔽 {name}", options=options, index=default_idx, help=f"选择{name}类型")

        # 处理列表类型（更智能的处理）
        if ((hasattr(annotation, '__origin__') and annotation.__origin__ in [list, typing.List])
            or isinstance(default_val, list)):
            # 特殊处理时间相关参数，添加更详细的帮助文本
            if any(keyword in name.lower() for keyword in ['days', 'day', 'sample', 'lookfwd', 'lookback']):
                help_text = f"输入{name}列表，支持格式：\n• 列表：[1,7,30]\n• 范围：range(1,31)\n• 带步长：range(1,31,2)\n• 逗号分隔：1,7,30"
            else:
                help_text = f"输入{name}列表"
            return st.text_input(f"📋 {name}", value=str(default_val), help=help_text)

        # 处理数值类型（更细致的处理）
        if annotation in [int, float] or 'int' in str(annotation) or 'float' in str(annotation):
            try:
                if 'int' in str(annotation) or annotation is int:
                    default_num = int(default_val) if default_val is not None else 0
                    # 为天数相关参数设置合理的范围和步长
                    if any(keyword in name.lower() for keyword in ['day', 'days', 'sample', 'lookfwd', 'lookback']):
                        return st.number_input(f"🔢 {name}", value=default_num, min_value=0, max_value=1000, step=1, help=f"设置{name}的值(天数)")
                    elif any(keyword in name.lower() for keyword in ['width', 'height']):
                        return st.number_input(f"🔢 {name}", value=default_num, min_value=100, max_value=2000, step=50, help=f"设置图表{name}")
                    else:
                        return st.number_input(f"🔢 {name}", value=default_num, help=f"设置{name}的值")
                else:
                    default_num = float(default_val) if default_val is not None else 0.0
                    if any(keyword in name.lower() for keyword in ['ratio', 'rate', 'alpha', 'confidence', 'threshold']):
                        return st.number_input(f"🔢 {name}", value=default_num, min_value=0.0, max_value=1.0, step=0.01, help=f"设置{name}的值(比例)")
                    else:
                        return st.number_input(f"🔢 {name}", value=default_num, help=f"设置{name}的值")
            except (TypeError, ValueError):
                return st.text_input(f"📝 {name}", value=str(default_val) if default_val else "", help=f"输入{name}的值")

        # 处理字符串类型的特殊情况
        if annotation is str or (hasattr(annotation, '__origin__') and annotation.__origin__ is str):
            # 如果参数名表明这是一个表达式（如range表达式）
            if 'range' in str(default_val) or 'np.' in str(default_val):
                return st.text_input(f"📝 {name}", value=str(default_val), help=f"输入{name}表达式，如 range(1, 31)")
            else:
                return st.text_input(f"📝 {name}", value=str(default_val) if default_val else "", help=f"输入{name}的值")

        # 处理特殊的Union类型
        if hasattr(annotation, '__origin__') and annotation.__origin__ is typing.Union:
            # 提取Union中的类型
            union_types = annotation.__args__
            type_names = [getattr(t, '__name__', str(t)) for t in union_types if t != type(None)]

            if len(type_names) == 1:
                # 如果只是Optional类型，按照单一类型处理
                return ParameterParser.create_form_widget(name, {'annotation': union_types[0], 'default': default_val}, signal_list, default_signal)
            else:
                # 多类型Union，使用文本输入
                return st.text_input(f"📝 {name}", value=str(default_val) if default_val else "", help=f"输入{name}的值 (支持类型: {', '.join(type_names)})")

        # 默认使用文本输入
        return st.text_input(f"📝 {name}", value=str(default_val) if default_val else "", help=f"输入{name}的值")

    @staticmethod
    def create_compact_parameter_form(signature: Dict[str, Any], signal_list: List[str] = None,
                                    default_signal: str = None, export_defaults: Dict = None) -> Dict[str, Any]:
        """创建紧凑的参数表单，智能分组和多列显示"""
        # 合并函数默认值和export_defaults
        merged_signature = {}
        for name, param_info in signature.items():
            merged_param_info = param_info.copy()
            if export_defaults and name in export_defaults:
                merged_param_info['default'] = export_defaults[name]
            merged_signature[name] = merged_param_info

        # 参数分组
        param_groups = {
            'primary': [],      # 主要参数：signal相关
            'time': [],         # 时间参数：day, days, window等
            'display': [],      # 显示参数：width, height
            'numeric': [],      # 数值参数：threshold, alpha等
            'boolean': [],      # 布尔参数
            'advanced': []      # 高级参数：其他
        }

        for name, param_info in merged_signature.items():
            annotation = param_info['annotation']

            if 'signal' in name.lower():
                param_groups['primary'].append((name, param_info))
            elif any(keyword in name.lower() for keyword in ['day', 'days', 'window', 'period', 'lag', 'sample', 'freq']):
                param_groups['time'].append((name, param_info))
            elif any(keyword in name.lower() for keyword in ['width', 'height']):
                param_groups['display'].append((name, param_info))
            elif annotation is bool:
                param_groups['boolean'].append((name, param_info))
            elif any(keyword in name.lower() for keyword in ['threshold', 'alpha', 'confidence', 'ratio', 'rate']):
                param_groups['numeric'].append((name, param_info))
            else:
                param_groups['advanced'].append((name, param_info))

        form_data = {}

        # 渲染主要参数（单列显示）
        if param_groups['primary']:
            st.markdown("**📊 信号参数**")
            for name, param_info in param_groups['primary']:
                form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal)

        # 渲染时间参数（2列显示）
        if param_groups['time']:
            st.markdown("**⏰ 时间参数**")
            time_params = param_groups['time']
            for i in range(0, len(time_params), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(time_params):
                        name, param_info = time_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal)

        # 渲染显示参数（2列显示）
        if param_groups['display']:
            st.markdown("**🎨 显示参数**")
            display_params = param_groups['display']
            for i in range(0, len(display_params), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(display_params):
                        name, param_info = display_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal)

        # 渲染数值参数（3列显示）
        if param_groups['numeric']:
            st.markdown("**🔢 数值参数**")
            numeric_params = param_groups['numeric']
            for i in range(0, len(numeric_params), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(numeric_params):
                        name, param_info = numeric_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal)

        # 渲染布尔参数（4列显示）
        if param_groups['boolean']:
            st.markdown("**✅ 开关参数**")
            boolean_params = param_groups['boolean']
            for i in range(0, len(boolean_params), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(boolean_params):
                        name, param_info = boolean_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal)

        # 渲染高级参数（2列显示）
        if param_groups['advanced']:
            with st.expander("🔧 高级参数", expanded=False):
                advanced_params = param_groups['advanced']
                for i in range(0, len(advanced_params), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(advanced_params):
                            name, param_info = advanced_params[i + j]
                            with col:
                                form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal)

        return form_data

    @staticmethod
    def process_form_data(form_data: Dict[str, Any], func_signature: Dict[str, Any]) -> Dict[str, Any]:
        """处理表单数据，转换为正确的类型"""
        processed = {}

        for name, value in form_data.items():
            param_info = func_signature.get(name, {})
            annotation = param_info.get('annotation', inspect.Parameter.empty)
            default_val = param_info.get('default', inspect.Parameter.empty)

            # 推断参数类型
            inferred_type = ParameterParser.infer_parameter_type(name, annotation, default_val)

            print(f"Processing {name}: value={value}, type={inferred_type}, annotation={annotation}")

            try:
                # 如果值已经是正确类型（来自预设），直接使用
                if not isinstance(value, str):
                    processed[name] = value
                    continue

                # 处理列表类型
                if inferred_type is list or (isinstance(default_val, list) and default_val != inspect.Parameter.empty):
                    if isinstance(value, str):
                        # 尝试安全评估
                        if value.strip().startswith('[') and value.strip().endswith(']'):
                            processed[name] = ast.literal_eval(value)
                        elif 'range(' in value:
                            # 处理range表达式
                            safe_globals = {"__builtins__": {}}
                            safe_locals = {"range": range, "np": np}
                            try:
                                result = eval(value, safe_globals, safe_locals)
                                processed[name] = list(result) if hasattr(result, '__iter__') else [result]
                            except:
                                st.error(f"❌ 无法解析{name}的range表达式: {value}")
                                processed[name] = default_val if default_val != inspect.Parameter.empty else []
                        else:
                            # 尝试解析逗号分隔的值
                            try:
                                processed[name] = [float(x.strip()) if '.' in x.strip() else int(x.strip())
                                                 for x in value.split(',')]
                            except:
                                processed[name] = ast.literal_eval(value)
                    else:
                        processed[name] = value

                # 处理数值类型
                elif inferred_type in [int, float]:
                    if inferred_type is int:
                        processed[name] = int(float(value))  # 先转float再转int，处理"1.0"这种情况
                    else:
                        processed[name] = float(value)

                # 处理布尔类型
                elif inferred_type is bool:
                    if isinstance(value, bool):
                        processed[name] = value
                    else:
                        processed[name] = str(value).lower() in ['true', '1', 'yes', 'on']

                # 处理字符串包含的range表达式
                elif isinstance(value, str) and ('range(' in value or 'np.' in value):
                    safe_globals = {"__builtins__": {}}
                    safe_locals = {"range": range, "np": np, "list": list}
                    try:
                        result = eval(value, safe_globals, safe_locals)
                        processed[name] = list(result) if hasattr(result, '__iter__') and not isinstance(result, str) else result
                    except Exception as e:
                        st.error(f"❌ 无法解析{name}的表达式: {value}. 错误: {e}")
                        processed[name] = default_val if default_val != inspect.Parameter.empty else value

                else:
                    processed[name] = value

            except (ValueError, SyntaxError) as e:
                st.error(f"❌ 参数 {name} 格式错误: {e}")
                processed[name] = default_val if default_val != inspect.Parameter.empty else value

        return processed


class ChartRenderer:
    """图表渲染器"""

    @staticmethod
    def render_plotly_chart(chart, key: str = None):
        """渲染Plotly图表"""
        try:
            # 验证chart是有效的Plotly图表
            if not hasattr(chart, 'to_html') or not callable(getattr(chart, 'to_html')):
                st.error("❌ 无效的Plotly图表对象")
                return

            # 确保图表有数据
            if not hasattr(chart, 'data') or len(chart.data) == 0:
                st.warning("⚠️ Plotly图表没有数据")
                return

            # 使用streamlit的plotly_chart渲染
            st.plotly_chart(chart, use_container_width=True, key=key)
            logger.info(f"Plotly图表渲染成功 (key: {key})")

        except Exception as e:
            st.error(f"❌ Plotly图表渲染失败: {e}")
            logger.error(f"Plotly chart render error: {e}\n{traceback.format_exc()}")

            # 显示调试信息
            with st.expander("🔍 Plotly图表调试信息", expanded=False):
                st.write(f"图表类型: {type(chart)}")
                if hasattr(chart, 'data'):
                    st.write(f"数据轨迹数量: {len(chart.data)}")
                if hasattr(chart, 'layout'):
                    st.write(f"布局标题: {getattr(chart.layout, 'title', 'N/A')}")

    @staticmethod
    def render_matplotlib_chart(fig, key=None):
        """渲染Matplotlib图表"""
        try:
            if hasattr(fig, '_closed') and fig._closed:
                st.warning("⚠️ Matplotlib图表已关闭，无法显示")
                return

            # 检查是否是有效的matplotlib figure
            if not hasattr(fig, 'savefig'):
                st.error("⚠️ 无效的Matplotlib图表对象")
                return

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.image(buf, use_column_width=True)
            buf.close()

        except Exception as e:
            st.error(f"渲染Matplotlib图表错误: {e}")
            logger.error(f"Matplotlib chart render error: {e}\n{traceback.format_exc()}")

    @staticmethod
    def render_dataframe(df, key=None):
        """渲染DataFrame"""
        if isinstance(df, pd.DataFrame):
            df_display = df.copy()
            numeric_cols = df_display.select_dtypes(include=[np.number]).columns
            df_display[numeric_cols] = df_display[numeric_cols].round(4)
            st.dataframe(df_display, use_container_width=True, height=400, key=key)
        else:
            st.text(str(df))

    @staticmethod
    def render_analysis_output(result, key: str = None):
        """智能渲染分析结果"""
        try:
            # 检查结果是否为None
            if result is None:
                st.warning("⚠️ 分析未返回结果")
                return

            # 如果结果是元组，取第一个元素
            if isinstance(result, tuple):
                if len(result) > 0:
                    result = result[0]
                else:
                    st.warning("⚠️ 分析返回空元组")
                    return

            # 检测plotly图表 - 更严格的检查
            if hasattr(result, '__module__') and result.__module__ and 'plotly' in result.__module__:
                # 进一步验证是否为有效的plotly图表
                if hasattr(result, 'data') and hasattr(result, 'layout'):
                    ChartRenderer.render_plotly_chart(result, key=key)
                    return
                else:
                    st.warning("⚠️ 检测到Plotly对象但数据结构不完整")
                    logger.warning(f"Invalid plotly object structure: {type(result)}")

            # 检测matplotlib图表 - 更严格的检查
            if hasattr(result, 'savefig') and callable(getattr(result, 'savefig')):
                # 验证是否为有效的matplotlib图表
                if hasattr(result, 'axes') or hasattr(result, 'get_axes'):
                    ChartRenderer.render_matplotlib_chart(result, key=key)
                    return
                else:
                    st.warning("⚠️ 检测到Matplotlib对象但结构不完整")

            # 检测pandas DataFrame
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    ChartRenderer.render_dataframe(result, key=key)
                    return
                else:
                    st.warning("⚠️ 分析返回空的DataFrame")
                    return

            # 检测字符串或其他类型
            if isinstance(result, str):
                if result.strip():  # 非空字符串
                    st.text(result)
                else:
                    st.warning("⚠️ 分析返回空字符串")
                return

            # 检测数值类型
            if isinstance(result, (int, float, complex)):
                st.metric("分析结果", result)
                return

            # 检测列表或数组
            if isinstance(result, (list, tuple, np.ndarray)):
                if len(result) > 0:
                    # 尝试转换为更友好的显示格式
                    if isinstance(result, np.ndarray):
                        if result.ndim == 1 and len(result) <= 10:
                            # 小的一维数组显示为表格
                            df_result = pd.DataFrame({'值': result})
                            st.dataframe(df_result, use_container_width=True)
                        else:
                            st.write(result)
                    else:
                        st.write(result)
                else:
                    st.warning("⚠️ 分析返回空列表/数组")
                return

            # 检测字典
            if isinstance(result, dict):
                if result:  # 非空字典
                    # 尝试转换为更好的显示格式
                    try:
                        df_result = pd.DataFrame.from_dict(result, orient='index', columns=['值'])
                        st.dataframe(df_result, use_container_width=True)
                    except:
                        st.json(result)
                else:
                    st.warning("⚠️ 分析返回空字典")
                return

            # 其他类型，尝试转换为字符串显示
            result_str = str(result)
            if result_str and result_str != 'None':
                # 检查是否包含有用信息
                if len(result_str) > 10 or any(char.isalnum() for char in result_str):
                    st.text(result_str)
                else:
                    st.warning("⚠️ 分析返回了无意义的结果")
                    with st.expander("🔍 原始结果", expanded=False):
                        st.write(f"类型: {type(result)}")
                        st.write(f"内容: {repr(result)}")
            else:
                st.warning("⚠️ 分析返回了无法显示的结果类型")
                with st.expander("🔍 调试信息", expanded=False):
                    st.write(f"结果类型: {type(result)}")
                    st.write(f"结果内容: {repr(result)}")

        except Exception as e:
            st.error(f"❌ 渲染分析结果时发生错误: {e}")
            logger.error(f"Analysis output render error: {e}\n{traceback.format_exc()}")

            # 显示详细的调试信息
            with st.expander("🔍 详细错误信息", expanded=False):
                st.code(traceback.format_exc())
                st.write("**结果信息:**")
                st.write(f"类型: {type(result)}")
                try:
                    st.write(f"内容: {repr(result)[:1000]}...")  # 限制显示长度
                except:
                    st.write("无法显示结果内容")


class AnalysisSection:
    """分析模块基类"""

    def __init__(self, name: str, icon: str, description: str, func_names: List[str]):
        self.name = name
        self.icon = icon
        self.description = description
        self.func_names = func_names if isinstance(func_names, list) else [func_names]
        self.state_key = name.lower().replace(' ', '_')

        # self.key =

    def render(self, sp: SignalPerf, signal_list: List[str], default_signal: str, export_mode: bool = False):
        """渲染分析模块"""
        with st.container():
            # 头部
            col_header, col_toggle = st.columns([0.95, 0.05])
            with col_header:
                st.header(f"{self.icon} {self.name}")
            with col_toggle:
                if not export_mode:
                    expanded = st.session_state.factor_section_states.get(self.state_key, False)
                    if st.button("🔽" if expanded else "▶️", key=f"toggle_{self.state_key}"):
                        st.session_state.factor_section_states[self.state_key] = not expanded
                        st.rerun()

            # 内容
            if st.session_state.factor_section_states.get(self.state_key, False):
                st.markdown(f"*{self.description}*")
                self.render_content(sp, signal_list, default_signal, export_mode)

        st.markdown("---")

    def render_content(self, sp: SignalPerf, signal_list: List[str], default_signal: str, export_mode: bool):
        """渲染具体内容 - 需要在子类中实现"""
        raise NotImplementedError

    def create_parameter_form(self, sp: SignalPerf, func_name: str, signal_list: List[str], default_signal: str) -> Tuple[Dict, bool]:
        """创建参数表单"""
        func = getattr(sp, func_name)  # 直接从SignalPerf实例获取方法
        signature = ParameterParser.get_function_signature(func)

        with st.form(key=f"{self.state_key}_{func_name}_form"):
            # 使用紧凑参数表单
            form_data = ParameterParser.create_compact_parameter_form(
                signature, signal_list, default_signal
            )
            submit = st.form_submit_button("🚀 运行分析")

        return form_data, submit

    def execute_analysis(self, sp: SignalPerf, func_name: str, form_data: Dict) -> Any:
        """执行分析"""
        func = getattr(sp, func_name)
        signature = ParameterParser.get_function_signature(func)
        processed_data = ParameterParser.process_form_data(form_data, signature)

        logger.info(f"执行分析函数: {func_name}, 参数: {processed_data}")

        try:
            result = func(**processed_data)

            # 缓存结果
            cache_key = f"{func_name}_{hash(str(processed_data))}"
            st.session_state.factor_analysis_results[cache_key] = result

            logger.info(f"分析函数 {func_name} 执行成功，结果类型: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"分析函数 {func_name} 执行失败: {e}\n{traceback.format_exc()}")
            raise


class AutoAnalysisSection(AnalysisSection):
    """自动分析模块 - 可以自动处理任何函数的参数"""

    def __init__(self, name: str, icon: str, description: str, func_name: str, export_defaults: Dict = None):
        super().__init__(name, icon, description, [func_name])
        self.main_func_name = func_name
        self.export_defaults = export_defaults or {}

    def create_parameter_form(self, sp: SignalPerf, func_name: str, signal_list: List[str], default_signal: str) -> Tuple[Dict, bool]:
        """创建参数表单 - 支持export_defaults作为默认值"""
        func = getattr(sp, func_name)
        signature = ParameterParser.get_function_signature(func)

        with st.form(key=f"{self.state_key}_{func_name}_form"):
            # 使用紧凑参数表单，传入export_defaults
            form_data = ParameterParser.create_compact_parameter_form(
                signature, signal_list, default_signal, self.export_defaults
            )
            submit = st.form_submit_button("🚀 运行分析")

        print("form_data", form_data)

        return form_data, submit

    def render_content(self, sp: SignalPerf, signal_list: List[str], default_signal: str, export_mode: bool):
        # 生成结果缓存key
        cache_key = f"{self.main_func_name}_{default_signal}"

        # 检查是否有缓存的结果
        has_cached_result = (
            'factor_analysis_results' in st.session_state and
            cache_key in st.session_state.factor_analysis_results
        )

        if not export_mode:
            with st.expander("⚙️ 参数设置", expanded=False):
                form_data, submit = self.create_parameter_form(
                    sp, self.main_func_name, signal_list, default_signal
                )
        else:
            submit = True
            # 获取函数签名以获取所有参数
            func = getattr(sp, self.main_func_name)
            signature = ParameterParser.get_function_signature(func)

            # 使用预设的导出默认值，并补充缺失的参数
            form_data = {}

            # 首先用函数的默认值填充所有参数
            for name, param_info in signature.items():
                form_data[name] = param_info['default']

            # 然后用export_defaults覆盖指定的参数
            if self.export_defaults:
                form_data.update(self.export_defaults)

            # 最后处理特殊参数
            if 'signal_name' in form_data and form_data['signal_name'] is None:
                form_data['signal_name'] = default_signal
            if 'signal_names' in form_data and form_data['signal_names'] is None:
                form_data['signal_names'] = signal_list

        # 如果有缓存结果，直接显示
        if has_cached_result and not (submit if 'submit' in locals() else False):
            try:
                cached_result = st.session_state.factor_analysis_results[cache_key]
                st.markdown(f"#### 📊 {self.name} 分析结果")
                ChartRenderer.render_analysis_output(
                    cached_result,
                    key=f"{self.main_func_name}_{default_signal}_cached"
                )

                # 显示缓存提示
                st.caption("💾 显示缓存结果 - 点击'运行分析'重新计算")

            except Exception as e:
                st.error(f"❌ 显示缓存结果错误: {str(e)}")
                logger.error(f"Cached result display error: {e}")

        # 如果用户点击了运行按钮或在导出模式下，执行分析
        if submit if 'submit' in locals() else False:
            try:
                with st.spinner(f"运行{self.name}..."):
                    result = self.execute_analysis(sp, self.main_func_name, form_data)

                    # 检查结果是否为空
                    if result is None:
                        st.warning(f"⚠️ {self.name} 分析未返回结果")
                        return

                    # 缓存结果
                    if 'factor_analysis_results' not in st.session_state:
                        st.session_state.factor_analysis_results = {}
                    st.session_state.factor_analysis_results[cache_key] = result

                    # 创建一个容器来显示结果
                    result_container = st.container()
                    with result_container:
                        st.markdown(f"#### 📊 {self.name} 分析结果")
                        ChartRenderer.render_analysis_output(
                            result,
                            key=f"{self.main_func_name}_{form_data.get('signal_name', default_signal)}_{hash(str(form_data))}"
                        )

            except Exception as e:
                st.error(f"❌ {self.name} 分析错误: {str(e)}")
                logger.error(f"{self.main_func_name} error: {e}\n{traceback.format_exc()}")

                # 提供详细的错误信息
                with st.expander("🔍 详细错误信息", expanded=False):
                    st.code(traceback.format_exc())
                    st.write("**参数信息:**")
                    if 'form_data' in locals():
                        st.json(form_data)


class RollingICSection(AnalysisSection):
    """滚动IC分析 - 特殊处理显示两个图表"""

    def __init__(self):
        super().__init__(
            "3. Rolling IC Analysis",
            "🔄",
            "分析滚动时间窗口的信息系数",
            ["plot_rolling_ic"]
        )

    def render_content(self, sp: SignalPerf, signal_list: List[str], default_signal: str, export_mode: bool):
        # 生成结果缓存key
        cache_key_ts = f"plot_rolling_ic_ts_{default_signal}"
        cache_key_box = f"plot_rolling_ic_box_{default_signal}"

        # 检查是否有缓存的结果
        has_cached_ts = (
            'factor_analysis_results' in st.session_state and
            cache_key_ts in st.session_state.factor_analysis_results
        )
        has_cached_box = (
            'factor_analysis_results' in st.session_state and
            cache_key_box in st.session_state.factor_analysis_results
        )

        with st.expander("⚙️ 参数设置", expanded=False):
            # 创建参数表单
            func = getattr(sp, "plot_rolling_ic")
            signature = ParameterParser.get_function_signature(func)

            with st.form(key=f"{self.state_key}_plot_rolling_ic_form"):
                # 使用紧凑参数表单
                form_data = ParameterParser.create_compact_parameter_form(
                    signature, signal_list, default_signal
                )
                submit = st.form_submit_button("🚀 运行分析")

        if 'lookfwd_day' not in form_data or form_data['lookfwd_day'] is None or form_data['lookfwd_day'] == 0:
            form_data['lookfwd_day'] = 7

        # 如果有缓存结果且用户没有点击运行按钮，显示缓存结果
        if (has_cached_ts or has_cached_box) and not submit:
            st.markdown("#### 📊 滚动IC分析结果")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 时间序列图")
                if has_cached_ts:
                    try:
                        cached_result_ts = st.session_state.factor_analysis_results[cache_key_ts]
                        ChartRenderer.render_analysis_output(
                            cached_result_ts,
                            key=f"rolling_ic_ts_{default_signal}_cached"
                        )
                        st.caption("💾 显示缓存结果")
                    except Exception as e:
                        st.error(f"❌ 显示缓存时间序列图错误: {str(e)}")
                else:
                    st.info("暂无时间序列图缓存")

            with col2:
                st.subheader("📊 箱线图")
                if has_cached_box:
                    try:
                        cached_result_box = st.session_state.factor_analysis_results[cache_key_box]
                        ChartRenderer.render_analysis_output(
                            cached_result_box,
                            key=f"rolling_ic_box_{default_signal}_cached"
                        )
                        st.caption("💾 显示缓存结果")
                    except Exception as e:
                        st.error(f"❌ 显示缓存箱线图错误: {str(e)}")
                else:
                    st.info("暂无箱线图缓存")

        if submit:
            try:
                with st.spinner("运行滚动IC分析..."):
                    # 检查信号名称
                    if not form_data.get('signal_name'):
                        st.error("❌ 请选择有效的信号名称")
                        return

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📈 时间序列图")
                        try:
                            ts_data = form_data.copy()
                            ts_data['plot_type'] = 'ts'
                            result_ts = self.execute_analysis(sp, "plot_rolling_ic", ts_data)
                            if result_ts is not None:
                                # 缓存结果
                                if 'factor_analysis_results' not in st.session_state:
                                    st.session_state.factor_analysis_results = {}
                                st.session_state.factor_analysis_results[cache_key_ts] = result_ts

                                ChartRenderer.render_analysis_output(
                                    result_ts,
                                    key=f"rolling_ic_ts_{form_data.get('signal_name', default_signal)}_{hash(str(ts_data))}"
                                )
                            else:
                                st.warning("⚠️ 时间序列图分析未返回结果")
                        except Exception as e:
                            st.error(f"❌ 时间序列图生成错误: {e}")
                            logger.error(f"Rolling IC timeseries error: {e}")

                    with col2:
                        st.subheader("📊 箱线图")
                        try:
                            box_data = form_data.copy()
                            box_data['plot_type'] = 'box'
                            result_box = self.execute_analysis(sp, "plot_rolling_ic", box_data)
                            if result_box is not None:
                                # 缓存结果
                                if 'factor_analysis_results' not in st.session_state:
                                    st.session_state.factor_analysis_results = {}
                                st.session_state.factor_analysis_results[cache_key_box] = result_box

                                ChartRenderer.render_analysis_output(
                                    result_box,
                                    key=f"rolling_ic_box_{form_data.get('signal_name', default_signal)}_{hash(str(box_data))}"
                                )
                            else:
                                st.warning("⚠️ 箱线图分析未返回结果")
                        except Exception as e:
                            st.error(f"❌ 箱线图生成错误: {e}")
                            logger.error(f"Rolling IC boxplot error: {e}")

            except Exception as e:
                st.error(f"❌ 滚动IC分析错误: {e}")
                logger.error(f"Rolling IC error: {e}\n{traceback.format_exc()}")

                # 提供详细的错误信息
                with st.expander("🔍 详细错误信息", expanded=False):
                    st.code(traceback.format_exc())
                    st.write("**参数信息:**")
                    st.json(form_data)


class FactorAnalysisModule:
    """因子分析模块 - 重构优化版本"""

    def __init__(self):
        self.name = "Factor Analysis"
        self.description = "因子挖掘、有效性验证、组合分析"
        self.initialize_state()
        self.setup_sections()

    def initialize_state(self):
        """初始化模块状态"""
        if 'factor_signal_perf' not in st.session_state:
            st.session_state.factor_signal_perf = None
        if 'factor_signal_list' not in st.session_state:
            st.session_state.factor_signal_list = []
        if 'factor_default_signal' not in st.session_state:
            st.session_state.factor_default_signal = None
        if 'factor_section_states' not in st.session_state:
            st.session_state.factor_section_states = {
                '1._signal_diagnostics_analysis': True,  # 保持第一个默认展开
                '2._ic_decay_analysis': False,
                '3._rolling_ic_analysis': False,
                '4._rolling_ic_statistics': False,
                '5._ic_distribution_analysis': False,
                '6._ic_cumulative_analysis': False,
                '7._ic_autocorrelation_analysis': False,
                '8._combined_diagnostics': False
            }
        if 'factor_export_mode' not in st.session_state:
            st.session_state.factor_export_mode = False
        # 添加分析结果缓存
        if 'factor_analysis_results' not in st.session_state:
            st.session_state.factor_analysis_results = {}

    def setup_sections(self):
        """设置分析模块"""
        self.sections = [
            # 1. 信号诊断分析 - 使用自动分析模块
            AutoAnalysisSection(
                name="1. Signal Diagnostics Analysis",
                icon="📈",
                description="Analyze the characteristics and distribution of a single factor",
                func_name="plot_signal_diagnostics_plotly",
                export_defaults={'signal_name': None, 'width': 1200, 'height': 600}
            ),

            # 2. IC衰减分析 - 使用自动分析模块
            AutoAnalysisSection(
                name="2. IC Decay Analysis",
                icon="📉",
                description="Analyze the decay of information coefficients in different time windows",
                func_name="plot_ic_decay_multi_signals_all_data",
                export_defaults={'signal_names': None, 'lookfwd_days': range(1, 31), 'width': 800, 'height': 600}
            ),

            # 3. 滚动IC分析 - 特殊处理显示两个图表
            RollingICSection(),

            # 4. 滚动IC统计 - 使用自动分析模块
            AutoAnalysisSection(
                name="4. Rolling IC Statistics",
                icon="📊",
                description="Analyze the statistics of rolling information coefficients",
                func_name="calc_rolling_ic_stats",
                export_defaults={
                    'signal_name': None,
                    'lookfwd_day': 25,
                    'sample_days': [30, 90, 180, 360, 720],
                    'sample_freq_days': 30,
                    'risk_adj': True
                }
            ),

            # 5. IC分布分析 - 独立分析模块
            AutoAnalysisSection(
                name="5. IC Distribution Analysis",
                icon="📊",
                description="IC分布直方图、Q-Q图、偏度峰度统计",
                func_name="plot_ic_distribution",
                export_defaults={
                    'signal_name': None,
                    'lookfwd_day': 25,
                    'sample_day': 30,
                    'width': 800,
                    'height': 600
                }
            ),

            # 6. IC累积分析 - 独立分析模块
            AutoAnalysisSection(
                name="6. IC Cumulative Analysis",
                icon="📈",
                description="IC时间序列、累积IC、IC信息比率",
                func_name="plot_ic_cumulative_ir",
                export_defaults={
                    'signal_name': None,
                    'lookfwd_day': 25,
                    'sample_day': 30,
                    'width': 900,
                    'height': 600
                }
            ),

            # 7. IC自相关分析 - 独立分析模块
            AutoAnalysisSection(
                name="7. IC Autocorrelation Analysis",
                icon="🔄",
                description="IC自相关系数及置信区间",
                func_name="plot_ic_autocorrelation",
                export_defaults={
                    'signal_name': None,
                    'lookfwd_day': 25,
                    'sample_day': 30,
                    'max_lags': 20,
                    'width': 800,
                    'height': 600
                }
            ),

            # 8. 综合诊断分析 - 使用自动分析模块
            AutoAnalysisSection(
                name="8. Combined Diagnostics",
                icon="🔍",
                description="因子与收益的综合诊断信息分析",
                func_name="plot_combined_diagnostics",
                export_defaults={
                    'signal_name': None,
                    'return_window': 25,
                    'width': 900,
                    'height': 600,
                    'risk_adj': True
                }
            )
        ]

    def render(self):
        """渲染因子分析模块界面"""
        st.markdown("## 🔍 Factor Analysis Module")
        st.markdown("*进行深度因子分析，包括有效性验证、IC分析、组合分析等*")

        # 数据加载区域
        self.render_data_loading()

        # 如果数据存在，显示分析界面
        if st.session_state.factor_signal_perf is not None:
            self.render_analysis_dashboard()
        else:
            self.render_welcome_message()

    def generate_test_data(self):
        """生成测试数据用于调试"""
        np.random.seed(42)
        n_samples = 1000

        # 生成时间序列
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

        # 生成价格数据（随机游走）
        price_changes = np.random.normal(0, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(price_changes))

        # 生成因子数据
        factor1 = np.random.normal(0, 1, n_samples)
        factor2 = np.random.normal(0, 1, n_samples)
        factor3 = factor1 * 0.3 + np.random.normal(0, 0.8, n_samples)  # 与factor1相关

        # 创建DataFrame
        test_df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'sig_momentum': factor1,
            'sig_reversal': factor2,
            'sig_trend': factor3
        })

        return test_df

    def render_data_loading(self):
        """渲染数据加载界面"""
        with st.expander("📁 数据加载", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                uploaded_file = st.file_uploader(
                    "上传CSV文件",
                    type=['csv'],
                    help="上传包含因子数据的CSV文件",
                    key="factor_file_upload"
                )

            with col2:
                if st.button("📊 从数据管理加载", key="load_from_data_mgmt"):
                    if 'user_data' in st.session_state and st.session_state.user_data:
                        dataset_names = list(st.session_state.user_data.keys())
                        if dataset_names:
                            selected_dataset = st.selectbox(
                                "选择数据集",
                                dataset_names,
                                key="select_dataset_factor"
                            )
                            if selected_dataset:
                                df = st.session_state.user_data[selected_dataset]
                                self.process_uploaded_data(df, selected_dataset)
                    else:
                        st.warning("数据管理模块中无可用数据")

            with col3:
                if st.button("🧪 使用测试数据", key="use_test_data"):
                    test_df = self.generate_test_data()
                    self.process_uploaded_data(test_df, "测试数据")

            # 处理上传文件
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    self.process_uploaded_data(df, uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ 错误: {e}")
                    logger.error(f"Factor analysis error: {e}\n{traceback.format_exc()}")

    @staticmethod
    def validate_and_prepare_data(df: pd.DataFrame, price_col: str, signal_cols: List[str]) -> pd.DataFrame:
        """验证和准备数据用于SignalPerf初始化"""
        # 创建数据副本
        prepared_df = df.copy()

        # 确保价格列存在且为数值类型
        if price_col not in prepared_df.columns:
            raise ValueError(f"价格列 '{price_col}' 不存在于数据中")

        # 转换价格列为数值类型
        try:
            prepared_df[price_col] = pd.to_numeric(prepared_df[price_col], errors='coerce')
        except Exception as e:
            raise ValueError(f"无法将价格列 '{price_col}' 转换为数值类型: {e}")

        # 验证因子列
        if not signal_cols:
            raise ValueError("因子列列表不能为空")

        for col in signal_cols:
            if col not in prepared_df.columns:
                raise ValueError(f"因子列 '{col}' 不存在于数据中")

            # 尝试转换为数值类型
            try:
                prepared_df[col] = pd.to_numeric(prepared_df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"无法将因子列 '{col}' 转换为数值类型: {e}")

        # 确保有时间戳列
        if 'timestamp' not in prepared_df.columns:
            # 如果没有timestamp列，尝试找到时间相关的列
            time_cols = [col for col in prepared_df.columns if any(keyword in col.lower() for keyword in ['time', 'date'])]
            if time_cols:
                prepared_df['timestamp'] = prepared_df[time_cols[0]]
                logger.info(f"使用 '{time_cols[0]}' 作为时间戳列")
            else:
                # 如果没有时间列，创建一个简单的索引
                prepared_df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(prepared_df), freq='D')
                logger.info("创建了默认的时间戳列")

        # 确保timestamp列是datetime类型
        try:
            prepared_df['timestamp'] = pd.to_datetime(prepared_df['timestamp'])
        except Exception as e:
            logger.warning(f"无法转换时间戳列: {e}")
            prepared_df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(prepared_df), freq='D')

        return prepared_df

    def process_uploaded_data(self, df, filename):
        """处理上传数据"""
        st.success(f"✅ 文件加载成功: {filename}")
        st.info(f"数据形状: {df.shape}")

        # 显示数据预览
        with st.expander("📊 数据预览", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
            st.markdown("**列信息:**")

            # 安全地获取示例值
            example_values = []
            for col in df.columns:
                try:
                    if len(df) > 0:
                        val = df[col].iloc[0]
                        if pd.isna(val):
                            example_values.append('N/A')
                        else:
                            example_values.append(str(val))
                    else:
                        example_values.append('N/A')
                except Exception:
                    example_values.append('N/A')

            col_info = pd.DataFrame({
                '列名': df.columns.tolist(),
                '数据类型': [str(dtype) for dtype in df.dtypes],  # 转换为字符串避免pyarrow错误
                '非空值数量': df.count().tolist(),
                '空值数量': df.isnull().sum().tolist(),
                '示例值': example_values
            })
            st.dataframe(col_info, use_container_width=True)

        # 列配置
        st.subheader("📋 列配置")
        col1, col2, col3 = st.columns(3)

        all_columns = df.columns.tolist()

        with col1:
            default_price_col = 'close' if 'close' in all_columns else None
            if default_price_col is None:
                # 尝试找到其他可能的价格列
                price_candidates = [col for col in all_columns if any(keyword in col.lower() for keyword in ['price', 'close', 'adj', 'value'])]
                default_price_col = price_candidates[0] if price_candidates else all_columns[0]

            price_col = st.selectbox(
                "价格列",
                options=all_columns,
                index=all_columns.index(default_price_col) if default_price_col in all_columns else 0,
                help="选择包含价格数据的列",
                key="factor_price_col"
            )

        with col2:
            signal_options = [
                col for col in all_columns
                if col.lower().startswith('sig') or col.lower().startswith('fea') or col.lower().startswith('factor')
            ]
            # 如果没有找到自动检测的因子列，提供更多选项
            if not signal_options:
                # 排除常见的非因子列
                excluded_keywords = ['timestamp', 'datetime', 'date', 'time', 'close', 'price', 'open', 'high', 'low', 'volume', 'unnamed: 0', 'amount']
                signal_options = [col for col in all_columns if not any(keyword in col.lower() for keyword in excluded_keywords)]

            signal_cols = st.multiselect(
                "因子列",
                options=all_columns,
                default=signal_options if signal_options else [],
                help="选择包含因子数据的列。如果没有自动检测到，请手动选择数值型列作为因子",
                key="factor_signal_cols"
            )

        with col3:
            labeling_method = st.selectbox(
                "标签方法",
                options=['point', 'triple'],
                index=0,
                help="选择因子标签方法：point=点对点收益，triple=三重障碍标签",
                key="factor_labeling_method"
            )

        # 数据验证
        validation_errors = []
        if not price_col:
            validation_errors.append("❌ 请选择价格列")
        if not signal_cols:
            validation_errors.append("❌ 请选择至少一个因子列")
        if price_col in signal_cols:
            validation_errors.append("⚠️ 价格列不应该作为因子列")

        # 检查数据类型
        if price_col and not pd.api.types.is_numeric_dtype(df[price_col]):
            validation_errors.append(f"❌ 价格列 '{price_col}' 不是数值类型")

        for col in signal_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_errors.append(f"⚠️ 因子列 '{col}' 不是数值类型，可能影响分析结果")

        # 显示验证结果
        if validation_errors:
            st.error("**数据验证问题:**")
            for error in validation_errors:
                st.write(error)
        # 显示数据预览
        if len(validation_errors) == 0:
            with st.expander("📊 数据预览", expanded=False):
                st.write("**选中的数据列:**")
                preview_cols = [price_col] + signal_cols
                st.dataframe(df[preview_cols].head(10))

                st.write("**数据统计信息:**")
                st.dataframe(df[preview_cols].describe())

        # 生成分析报告
        if st.button("🚀 生成因子分析报告", type="primary", key="generate_factor_report"):
            if price_col and signal_cols:
                try:
                    with st.spinner("生成因子分析报告..."):
                        # 详细的调试信息
                        st.info("📊 开始处理数据...")

                        # 数据预处理
                        logger.info(f"开始格式化数据，原始数据形状: {df.shape}")

                        # 第一步：验证和准备数据
                        validated_df = self.validate_and_prepare_data(df.copy(), price_col, signal_cols)
                        logger.info(f"数据验证完成，验证后数据形状: {validated_df.shape}")

                        # 第二步：应用format_data格式化
                        formatted_df = format_data(validated_df)
                        logger.info(f"格式化后数据形状: {formatted_df.shape}")

                        # 数据清理
                        initial_len = len(formatted_df)
                        formatted_df = formatted_df.dropna(subset=[price_col] + signal_cols)
                        cleaned_len = len(formatted_df)
                        dropped_rows = initial_len - cleaned_len

                        if dropped_rows > 0:
                            st.warning(f"⚠️ 删除了 {dropped_rows} 行包含空值的数据 ({dropped_rows/initial_len:.1%})")

                        if cleaned_len < 10:
                            st.error("❌ 清理后的数据太少，无法进行有效分析")
                            return

                        st.info("🔧 初始化SignalPerf...")

                        # 创建SignalPerf实例 - 现在数据已经过验证
                        sp = SignalPerf(
                            mode='local',
                            data=formatted_df,
                            price_col=price_col,
                            signal_cols=signal_cols,
                            labeling_method=labeling_method
                        )

                        logger.info("SignalPerf实例创建成功")

                        # 存储到session state
                        st.session_state.factor_signal_perf = sp
                        st.session_state.factor_signal_list = signal_cols
                        st.session_state.factor_default_signal = signal_cols[0]

                        st.success("✅ 因子分析报告生成成功!")
                        st.info(f"📈 已加载 {len(signal_cols)} 个因子: {', '.join(signal_cols)}")

                        # 初始化section states并全部展开
                        if 'factor_section_states' not in st.session_state:
                            st.session_state.factor_section_states = {}

                        # 获取所有section的state_key并展开
                        section_keys = [section.state_key for section in self.sections if hasattr(section, 'state_key')]
                        for key in section_keys:
                            st.session_state.factor_section_states[key] = True

                        # 标记刚生成报告，用于显示提示
                        st.session_state._just_generated_report = True
                        self.run_all_analysis()

                        # 询问用户是否要立即运行所有分析
                        st.markdown("---")
                        col_auto1, col_auto2 = st.columns([1, 1])


                except Exception as e:
                    st.error(f"❌ 生成报告时发生错误: {str(e)}")

                    # 详细的错误信息
                    with st.expander("🔍 详细错误信息", expanded=False):
                        st.code(traceback.format_exc())

                    # 常见问题排查建议
                    st.markdown("**🛠️ 常见问题排查:**")
                    st.markdown("""
                    1. **数据格式问题**: 确保价格列和因子列都是数值类型
                    2. **缺失值问题**: 检查数据中是否有过多的空值
                    3. **列名问题**: 确保选择的列名正确存在于数据中
                    4. **数据量问题**: 确保数据有足够的行数（建议至少100行）
                    5. **时间序列问题**: 如果有时间列，确保格式正确
                    """)

                    logger.error(f"Factor analysis error: {e}\n{traceback.format_exc()}")
            else:
                st.warning("⚠️ 请选择价格列和至少一个因子列")

        # 一键运行所有分析按钮（仅在报告已生成时显示）
        if st.session_state.factor_signal_perf is not None:
            st.markdown("---")
            col_run1, col_run2 = st.columns([1, 1])

            with col_run1:
                if st.button("🚀 一键运行所有分析", type="secondary", key="run_all_analysis"):
                    self.run_all_analysis()

            with col_run2:
                if st.button("🧹 清空所有结果", key="clear_all_results"):
                    if 'factor_analysis_results' in st.session_state:
                        st.session_state.factor_analysis_results.clear()
                    st.success("✅ 已清空所有分析结果")
                    st.rerun()

    def render_analysis_dashboard(self):
        """渲染分析仪表板"""
        sp = st.session_state.factor_signal_perf
        signal_list = st.session_state.factor_signal_list
        default_signal = st.session_state.factor_default_signal

        # 安全检查
        if sp is None:
            st.error("❌ SignalPerf实例未初始化，请重新生成分析报告")
            return

        if not signal_list:
            st.error("❌ 因子列表为空，请重新生成分析报告")
            return

        if not default_signal:
            st.error("❌ 默认因子未设置，请重新生成分析报告")
            return

        # 检查是否需要自动运行分析
        if hasattr(st.session_state, '_auto_run_analysis') and st.session_state._auto_run_analysis:
            st.session_state._auto_run_analysis = False  # 重置标志
            st.info("🚀 正在自动运行所有分析...")
            self.run_all_analysis()
            return  # run_all_analysis会触发st.rerun()，所以这里直接return

        # 显示成功提示（如果刚生成报告）
        if hasattr(st.session_state, '_just_generated_report') and st.session_state._just_generated_report:
            st.success("🎉 因子分析报告已生成！您可以展开下方的分析模块查看结果。")
            st.session_state._just_generated_report = False

        # 全局控制
        st.markdown("### 📊 因子分析仪表板")
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 3])

        with col1:
            st.markdown(f"**当前因子:** `{default_signal}`")
            st.markdown(f"**因子数量:** {len(signal_list)}")

        with col2:
            if st.button("📖 全部展开", key="factor_expand_all"):
                # 确保section_states存在
                if 'factor_section_states' not in st.session_state:
                    st.session_state.factor_section_states = {}
                # 获取所有可用的section keys
                section_keys = [section.state_key for section in self.sections if hasattr(section, 'state_key')]
                for key in section_keys:
                    st.session_state.factor_section_states[key] = True
                st.rerun()

        with col3:
            if st.button("📕 全部收起", key="factor_collapse_all"):
                # 确保section_states存在
                if 'factor_section_states' not in st.session_state:
                    st.session_state.factor_section_states = {}
                # 获取所有可用的section keys
                section_keys = [section.state_key for section in self.sections if hasattr(section, 'state_key')]
                for key in section_keys:
                    st.session_state.factor_section_states[key] = False
                st.rerun()

        with col4:
            if st.button("📄 导出模式" if not st.session_state.factor_export_mode else "🔧 编辑模式",
                        key="toggle_factor_export_mode",
                        help="切换到导出模式生成高密度显示版本"):
                st.session_state.factor_export_mode = not st.session_state.factor_export_mode
                if st.session_state.factor_export_mode:
                    for key in st.session_state.factor_section_states:
                        st.session_state.factor_section_states[key] = True
                st.rerun()

        with col5:
            if st.button("💾 保存分析结果", key="save_factor_analysis"):
                self.save_analysis_results()

        # 导出模式指示器
        if st.session_state.factor_export_mode:
            st.info("🔄 **导出模式已启用** - 显示高密度数据，隐藏所有参数控制", icon="📄")

        st.markdown("---")

        # 渲染所有分析模块
        try:
            for section in self.sections:
                section.render(sp, signal_list, default_signal, st.session_state.factor_export_mode)
        except Exception as e:
            st.error(f"❌ 渲染分析模块时发生错误: {str(e)}")
            logger.error(f"Dashboard render error: {e}\n{traceback.format_exc()}")

            # 提供重置选项
            if st.button("🔄 重置分析状态", key="reset_analysis_state"):
                st.session_state.factor_signal_perf = None
                st.session_state.factor_signal_list = []
                st.session_state.factor_default_signal = None
                st.session_state.factor_analysis_results = {}
                st.rerun()

    def render_welcome_message(self):
        """渲染欢迎信息"""
        st.info("👈 请在上方加载数据以开始因子分析")

        st.subheader("📋 期望的数据格式")
        st.markdown("""
        您的CSV文件应包含:
        - **价格列**: 历史价格数据 (如收盘价、调整收盘价)
        - **因子列**: 一个或多个包含因子值的列 (以'sig'、'fea'或'factor'开头的列会被自动检测)
        - **时间戳列**: (可选) 日期/时间信息

        示例:
        """)

        example_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'close': [100.0, 101.5, 99.8, 102.3, 104.1],
            'sig_momentum': [0.1, -0.2, 0.3, -0.1, 0.4],
            'fea_volume': [0.05, 0.15, -0.1, 0.2, -0.05]
        })

        st.dataframe(example_df, use_container_width=True)

    def save_analysis_results(self):
        """保存分析结果"""
        if st.session_state.factor_signal_perf is not None:
            save_data = {
                'timestamp': pd.Timestamp.now(),
                'module': 'Factor Analysis',
                'signal_list': st.session_state.factor_signal_list,
                'default_signal': st.session_state.factor_default_signal,
                'section_states': st.session_state.factor_section_states.copy(),
                'parameters': {
                    'labeling_method': getattr(st.session_state.factor_signal_perf, 'labeling_method', 'point'),
                    'price_col': getattr(st.session_state.factor_signal_perf, 'price_col', 'close'),
                    'signal_cols': st.session_state.factor_signal_list
                }
            }

            if 'saved_reports' not in st.session_state:
                st.session_state.saved_reports = []

            st.session_state.saved_reports.append(save_data)
            st.success("✅ 分析结果已保存到结果管理模块")
        else:
            st.warning("⚠️ 无分析结果可保存")

    def run_all_analysis(self):
        """一键运行所有分析模块"""
        try:
            st.info("🚀 开始运行所有分析...")

            sp = st.session_state.factor_signal_perf
            signal_list = st.session_state.factor_signal_list
            default_signal = st.session_state.factor_default_signal

            if sp is None or not signal_list or not default_signal:
                st.error("❌ SignalPerf实例或信号列表未初始化")
                return

            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_sections = len(self.sections)
            successful_analyses = 0
            failed_analyses = []

            for i, section in enumerate(self.sections):
                try:
                    status_text.text(f"正在运行: {section.name}")
                    progress_bar.progress((i + 1) / total_sections)

                    # 确保section展开
                    st.session_state.factor_section_states[section.state_key] = True

                    # 运行分析
                    if isinstance(section, AutoAnalysisSection):
                        # 对于自动分析模块，使用预设的默认值
                        self._run_auto_analysis_section(section, sp, signal_list, default_signal)
                        successful_analyses += 1
                    elif isinstance(section, RollingICSection):
                        # 对于特殊的滚动IC分析
                        self._run_rolling_ic_section(section, sp, signal_list, default_signal)
                        successful_analyses += 1
                    else:
                        # 其他类型的分析模块
                        logger.warning(f"未知的分析模块类型: {type(section)}")
                        failed_analyses.append(section.name)

                except Exception as e:
                    logger.error(f"运行 {section.name} 失败: {e}")
                    failed_analyses.append(section.name)
                    continue

            # 清理进度显示
            progress_bar.empty()
            status_text.empty()

            # 显示总结
            if successful_analyses > 0:
                st.success(f"✅ 成功运行 {successful_analyses}/{total_sections} 个分析模块")

                # 自动展开所有模块显示结果
                for section in self.sections:
                    st.session_state.factor_section_states[section.state_key] = True

                st.balloons()  # 添加庆祝动画

            if failed_analyses:
                st.warning(f"⚠️ 以下分析模块运行失败: {', '.join(failed_analyses)}")

            # 显示提示信息
            if successful_analyses > 0:
                st.info("🎉 所有分析已完成！结果已自动显示在下方各模块中。您可以滚动查看所有图表和分析结果。")

            # 刷新页面以显示所有结果
            st.rerun()

        except Exception as e:
            st.error(f"❌ 运行所有分析时发生错误: {str(e)}")
            logger.error(f"Run all analysis error: {e}\n{traceback.format_exc()}")

    def _run_auto_analysis_section(self, section: AutoAnalysisSection, sp: SignalPerf,
                                   signal_list: List[str], default_signal: str):
        """运行自动分析模块"""
        try:
            # 获取函数签名
            func = getattr(sp, section.main_func_name)
            signature = ParameterParser.get_function_signature(func)

            # 准备表单数据 - 使用export_defaults作为基础
            form_data = {}

            # 首先用函数的默认值填充所有参数
            for name, param_info in signature.items():
                form_data[name] = param_info['default']

            # 然后用export_defaults覆盖指定的参数
            if section.export_defaults:
                form_data.update(section.export_defaults)

            # 最后处理特殊参数
            if 'signal_name' in form_data and form_data['signal_name'] is None:
                form_data['signal_name'] = default_signal
            if 'signal_names' in form_data and form_data['signal_names'] is None:
                form_data['signal_names'] = signal_list

            # 执行分析
            result = section.execute_analysis(sp, section.main_func_name, form_data)

            if result is not None:
                # 缓存结果 - 使用与render_content相同的key格式
                cache_key = f"{section.main_func_name}_{default_signal}"
                if 'factor_analysis_results' not in st.session_state:
                    st.session_state.factor_analysis_results = {}
                st.session_state.factor_analysis_results[cache_key] = result

                logger.info(f"成功运行 {section.name} 并缓存结果")
            else:
                logger.warning(f"{section.name} 返回空结果")

        except Exception as e:
            logger.error(f"运行 {section.name} 失败: {e}")
            raise

    def _run_rolling_ic_section(self, section: RollingICSection, sp: SignalPerf,
                                signal_list: List[str], default_signal: str):
        """运行滚动IC分析模块"""
        try:
            # 获取函数签名
            func = getattr(sp, "plot_rolling_ic")
            signature = ParameterParser.get_function_signature(func)

            # 准备基础参数
            base_form_data = {}
            for name, param_info in signature.items():
                base_form_data[name] = param_info['default']

            # 设置默认信号和前瞻天数
            base_form_data['signal_name'] = default_signal
            if 'lookfwd_day' not in base_form_data or base_form_data['lookfwd_day'] is None:
                base_form_data['lookfwd_day'] = 7

            # 准备缓存keys
            cache_key_ts = f"plot_rolling_ic_ts_{default_signal}"
            cache_key_box = f"plot_rolling_ic_box_{default_signal}"

            if 'factor_analysis_results' not in st.session_state:
                st.session_state.factor_analysis_results = {}

            # 运行时间序列图
            ts_data = base_form_data.copy()
            ts_data['plot_type'] = 'ts'
            result_ts = section.execute_analysis(sp, "plot_rolling_ic", ts_data)
            if result_ts is not None:
                st.session_state.factor_analysis_results[cache_key_ts] = result_ts
                logger.info(f"成功运行 {section.name} 时间序列图并缓存结果")

            # 运行箱线图
            box_data = base_form_data.copy()
            box_data['plot_type'] = 'box'
            result_box = section.execute_analysis(sp, "plot_rolling_ic", box_data)
            if result_box is not None:
                st.session_state.factor_analysis_results[cache_key_box] = result_box
                logger.info(f"成功运行 {section.name} 箱线图并缓存结果")

            logger.info(f"成功运行 {section.name}")

        except Exception as e:
            logger.error(f"运行 {section.name} 失败: {e}")
            raise


