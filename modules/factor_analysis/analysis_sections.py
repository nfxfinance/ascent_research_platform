import streamlit as st
import logging
import traceback
from typing import List, Dict, Any, Tuple

from .parameter_parser import ParameterParser
from .chart_renderer import ChartRenderer

logger = logging.getLogger(__name__)


class AnalysisSection:
    """分析模块基类"""

    def __init__(self, name: str, icon: str, description: str, func_names: List[str]):
        self.name = name
        self.icon = icon
        self.description = description
        self.func_names = func_names
        self.state_key = name.lower().replace(' ', '_').replace('&', '&').replace('-', '_')

    def render(self, sp, signal_list: List[str], default_signal: str, export_mode: bool = False):
        """渲染分析模块"""
        # 获取展开状态
        expanded = st.session_state.factor_section_states.get(self.state_key, False)

        with st.expander(f"{self.icon} {self.name}", expanded=expanded):
            # 更新状态
            st.session_state.factor_section_states[self.state_key] = True

            st.markdown(f"**{self.description}**")

            try:
                # 分析逻辑委托给子类实现
                self.render_content(sp, signal_list, default_signal, export_mode)
            except Exception as e:
                st.error(f"❌ Error rendering {self.name}: {e}")
                logger.error(f"Section {self.name} render error: {e}\n{traceback.format_exc()}")

    def render_content(self, sp, signal_list: List[str], default_signal: str, export_mode: bool):
        """子类需要实现的内容渲染方法"""
        pass

    def get_saved_parameters(self) -> Dict[str, Any]:
        """获取保存的参数配置"""
        from modules.factor_export import get_saved_parameters_for_section
        return get_saved_parameters_for_section(self.state_key)

    def _auto_save_to_current_id(self):
        """自动保存分析结果到当前ID"""
        # 检查是否有当前分析ID
        if hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id:
            try:
                from modules.factor_export import update_analysis_results
                update_success = update_analysis_results(
                    st.session_state.current_analysis_id,
                    analysis_results=st.session_state.factor_analysis_results,
                    analysis_params=st.session_state.factor_analysis_params,
                    section_states=st.session_state.factor_section_states
                )
                if update_success:
                    st.success(f"💾 Results automatically saved to ID: `{st.session_state.current_analysis_id}`")
                else:
                    st.warning("⚠️ Failed to auto-save results")
            except ImportError:
                st.warning("Factor export module not available for auto-save")
            except Exception as e:
                st.error(f"❌ Auto-save failed: {e}")
        else:
            st.info("💡 No current analysis ID - results cached locally only")

    def create_parameter_form(self, sp, func_name: str, signal_list: List[str], default_signal: str) -> Tuple[Dict, bool]:
        """创建参数表单"""
        func = getattr(sp, func_name)
        signature = ParameterParser.get_function_signature(func)

        # 获取保存的参数配置
        saved_params = self.get_saved_parameters()

        with st.form(key=f"{self.state_key}_{func_name}_form"):
            # 使用紧凑参数表单，传入保存的参数作为默认值
            form_data = ParameterParser.create_compact_parameter_form(
                signature, signal_list, default_signal, saved_params
            )

            # 显示是否使用了保存的参数
            if saved_params:
                st.info(f"📋 Using saved parameters from configuration")
                with st.expander("🔍 Saved Parameters", expanded=False):
                    st.json(saved_params)

            submit = st.form_submit_button("🚀 Run Analysis")

        return form_data, submit

    def execute_analysis(self, sp, func_name: str, form_data: Dict) -> Any:
        """执行分析"""
        func = getattr(sp, func_name)
        signature = ParameterParser.get_function_signature(func)
        processed_data = ParameterParser.process_form_data(form_data, signature)

        logger.info(f"执行分析函数: {func_name}, 参数: {processed_data}")
        st.session_state.factor_analysis_params[func_name] = processed_data

        try:
            result = func(**processed_data)

            # 缓存结果
            cache_key = f"{func_name}_{hash(str(processed_data))}"
            st.session_state.factor_analysis_results[cache_key] = result

            logger.info(f"分析函数 {func_name} 执行成功，结果类型: {type(result)}")
            if isinstance(result, tuple):
                return result[0]
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

    def create_parameter_form(self, sp, func_name: str, signal_list: List[str], default_signal: str) -> Tuple[Dict, bool]:
        """创建参数表单 - 支持export_defaults和保存的参数作为默认值"""
        func = getattr(sp, func_name)
        signature = ParameterParser.get_function_signature(func)

        # 获取保存的参数配置
        saved_params = self.get_saved_parameters()

        # 合并默认值优先级：保存的参数 > export_defaults > 函数默认值
        merged_defaults = self.export_defaults.copy()
        if saved_params:
            merged_defaults.update(saved_params)

        with st.form(key=f"{self.state_key}_{func_name}_form"):
            # 使用紧凑参数表单，传入合并后的默认值
            form_data = ParameterParser.create_compact_parameter_form(
                signature, signal_list, default_signal, merged_defaults
            )

            # 显示参数来源信息
            if saved_params:
                st.info(f"📋 Using saved parameters from configuration")
                with st.expander("🔍 Saved Parameters", expanded=False):
                    st.json(saved_params)
            elif self.export_defaults:
                st.caption("📄 Using export default parameters")

            submit = st.form_submit_button("🚀 Run Analysis")

        return form_data, submit

    def render_content(self, sp, signal_list: List[str], default_signal: str, export_mode: bool):
        # 生成结果缓存key
        cache_key = f"{self.main_func_name}_{default_signal}"

        # 检查是否有缓存的结果
        has_cached_result = (
            'factor_analysis_results' in st.session_state and
            cache_key in st.session_state.factor_analysis_results
        )

        if not export_mode:
            with st.expander("⚙️ parameter settings", expanded=False):
                form_data, submit = self.create_parameter_form(
                    sp, self.main_func_name, signal_list, default_signal
                )
        else:
            # Export Mode: 只显示缓存结果，不提供运行按钮
            submit = False
            form_data = None

        # 如果有缓存结果，直接显示
        if not submit and has_cached_result:
            try:
                cached_result = st.session_state.factor_analysis_results[cache_key]
                ChartRenderer.render_analysis_output(
                    cached_result,
                    key=f"{self.main_func_name}_{default_signal}_cached"
                )

                # 在非导出模式下显示缓存提示
                if not export_mode:
                    st.caption("💾 Showing cached results - Click 'Run Analysis' to recalculate")

            except Exception as e:
                st.error(f"❌ 显示缓存结果错误: {str(e)}")
                logger.error(f"Cached result display error: {e}")
        elif export_mode:
            # Export Mode 下如果没有缓存结果，显示提示
            st.warning("⚠️ No cached results available. Please run analysis first in Edit Mode.")

        # 只在非导出模式且用户点击了运行按钮时执行分析
        if submit and not export_mode:
            try:
                with st.spinner(f"running {self.name}..."):
                    logger.info(f"Running {self.name} with form_data: {form_data}")
                    result = self.execute_analysis(sp, self.main_func_name, form_data)

                    # 检查结果是否为空
                    if result is None:
                        st.warning(f"⚠️ {self.name} analysis did not return results")
                        return

                    # 缓存结果
                    if 'factor_analysis_results' not in st.session_state:
                        st.session_state.factor_analysis_results = {}
                    st.session_state.factor_analysis_results[cache_key] = result

                    # 创建一个容器来显示结果
                    result_container = st.container()
                    with result_container:
                        ChartRenderer.render_analysis_output(
                            result,
                            key=f"{self.main_func_name}_{form_data.get('signal_name', default_signal)}_{hash(str(form_data))}"
                        )

                    # 自动保存分析结果到当前ID
                    self._auto_save_to_current_id()

            except Exception as e:
                st.error(f"❌ {self.name} analysis error: {str(e)}")
                logger.error(f"{self.main_func_name} error: {e}\n{traceback.format_exc()}")

                # 提供详细的错误信息
                with st.expander("🔍 detailed error information", expanded=False):
                    st.code(traceback.format_exc())
                    st.write("**parameter information:**")
                    if 'form_data' in locals():
                        st.json(form_data)


class RollingICSection(AnalysisSection):
    """滚动IC分析 - 特殊处理显示两个图表"""

    def __init__(self):
        super().__init__(
            "3. Rolling IC Analysis",
            "🔄",
            "Analyze the information coefficient of rolling time windows",
            ["plot_rolling_ic"]
        )

    def render_content(self, sp, signal_list: List[str], default_signal: str, export_mode: bool):
        # 生成结果缓存key
        cache_key = f"rolling_ic_{default_signal}"

        # 检查是否有缓存的结果
        has_cached_result = (
            'factor_analysis_results' in st.session_state and
            cache_key in st.session_state.factor_analysis_results
        )

        if not export_mode:
            with st.expander("⚙️ 参数设置", expanded=False):
                form_data, submit = self.create_parameter_form(
                    sp, "plot_rolling_ic", signal_list, default_signal
                )
        else:
            # Export Mode: 只显示缓存结果，不提供运行按钮
            submit = False
            form_data = None

        # 如果有缓存结果，直接显示
        if not submit and has_cached_result:
            try:
                cached_result = st.session_state.factor_analysis_results[cache_key]

                # 显示两个图表
                if isinstance(cached_result, tuple) and len(cached_result) == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 Rolling IC")
                        ChartRenderer.render_analysis_output(
                            cached_result[0],
                            key=f"rolling_ic_chart_{default_signal}_cached"
                        )
                    with col2:
                        st.subheader("📈 Rolling IC Statistics")
                        ChartRenderer.render_analysis_output(
                            cached_result[1],
                            key=f"rolling_ic_stats_{default_signal}_cached"
                        )
                else:
                    ChartRenderer.render_analysis_output(
                        cached_result,
                        key=f"rolling_ic_{default_signal}_cached"
                    )

                # 在非导出模式下显示缓存提示
                if not export_mode:
                    st.caption("💾 Showing cached results - Click 'Run Analysis' to recalculate")

            except Exception as e:
                st.error(f"❌ 显示缓存结果错误: {str(e)}")
                logger.error(f"Cached result display error: {e}")
        elif export_mode:
            # Export Mode 下如果没有缓存结果，显示提示
            st.warning("⚠️ No cached results available. Please run analysis first in Edit Mode.")

        # 只在非导出模式且用户点击了运行按钮时执行分析
        if submit and not export_mode:
            try:
                with st.spinner("运行滚动IC分析..."):
                    result = self.execute_analysis(sp, "plot_rolling_ic", form_data)

                    # 检查结果是否为空
                    if result is None:
                        st.warning("⚠️ 滚动IC分析未返回结果")
                        return

                    # 缓存结果
                    if 'factor_analysis_results' not in st.session_state:
                        st.session_state.factor_analysis_results = {}
                    st.session_state.factor_analysis_results[cache_key] = result

                    # 显示结果 - 特殊处理滚动IC的两个图表
                    if isinstance(result, tuple) and len(result) == 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📊 Rolling IC")
                            ChartRenderer.render_analysis_output(
                                result[0],
                                key=f"rolling_ic_chart_{form_data.get('signal_name', default_signal)}_{hash(str(form_data))}"
                            )
                        with col2:
                            st.subheader("📈 Rolling IC Statistics")
                            ChartRenderer.render_analysis_output(
                                result[1],
                                key=f"rolling_ic_stats_{form_data.get('signal_name', default_signal)}_{hash(str(form_data))}"
                            )
                    else:
                        ChartRenderer.render_analysis_output(
                            result,
                            key=f"rolling_ic_{form_data.get('signal_name', default_signal)}_{hash(str(form_data))}"
                        )

                    # 自动保存分析结果到当前ID
                    self._auto_save_to_current_id()

            except Exception as e:
                st.error(f"❌ 滚动IC分析错误: {str(e)}")
                logger.error(f"Rolling IC analysis error: {e}\n{traceback.format_exc()}")

                # 提供详细的错误信息
                with st.expander("🔍 详细错误信息", expanded=False):
                    st.code(traceback.format_exc())
                    st.write("**参数信息:**")
                    if 'form_data' in locals():
                        st.json(form_data)


class ROCPRSection(AnalysisSection):
    """ROC和PR曲线组合分析模块"""

    def __init__(self):
        super().__init__(
            "15. ROC & Precision-Recall Curves",
            "📊",
            "ROC Curve and Precision-Recall Curve side by side",
            ["plot_roc_curve", "plot_precision_recall_curve"]
        )
    def render_content(self, sp, signal_list: List[str], default_signal: str, export_mode: bool):
        """渲染ROC和PR曲线并排显示的内容"""
        # 生成结果缓存key
        cache_key = f"roc_pr_{default_signal}"

        # 检查是否有缓存的结果
        has_cached_result = (
            'factor_analysis_results' in st.session_state and
            cache_key in st.session_state.factor_analysis_results
        )

        # 如果有缓存结果，直接显示
        if has_cached_result:
            try:
                charts = st.session_state.factor_analysis_results[cache_key]
                logger.info(f"Using cached results for {self.name}")

                # 渲染并排图表
                from .chart_renderer import ChartRenderer
                ChartRenderer.render_side_by_side_charts(
                    charts['roc'], charts['pr'],
                    "ROC Curve", "Precision-Recall Curve"
                )

                # 在非导出模式下显示缓存提示
                if not export_mode:
                    st.caption("💾 Showing cached results - Generate new charts if needed")

                return
            except Exception as e:
                st.error(f"❌ Cached result display error: {str(e)}")
                logger.error(f"Cached result display error: {e}")

        # Export Mode 下如果没有缓存结果，显示提示
        if export_mode:
            st.warning("⚠️ No cached results available. Please run analysis first in Edit Mode.")
            return

        # 只在非导出模式下创建参数表单
        # 创建参数表单
        with st.form(key=f"form_{self.name.replace(' ', '_').replace('.', '')}"):
            # 基础参数
            col1, col2, col3 = st.columns(3)
            with col1:
                signal_name = st.selectbox(
                    "Signal",
                    options=signal_list,
                    index=signal_list.index(default_signal) if default_signal in signal_list else 0,
                    key=f"signal_{self.name}"
                )
                risk_adj = st.checkbox("Risk Adjusted", value=True, key=f"risk_adj_{self.name}")
            with col2:
                return_window = st.number_input(
                    "Return Window (days)",
                    min_value=0.1,
                    max_value=100.0,
                    value=7.0,
                    step=0.1,
                    key=f"return_window_{self.name}"
                )

            with col3:
                # 图表尺寸
                width = st.number_input("Width", min_value=200, max_value=800, value=300, key=f"width_{self.name}")
                height = st.number_input("Height", min_value=200, max_value=800, value=300, key=f"height_{self.name}")

            submit = st.form_submit_button("🚀 Generate Charts")

            if submit:
                try:
                    with st.spinner("Generating ROC and PR curves..."):
                        # 生成ROC曲线
                        roc_result = sp.plot_roc_curve(
                            signal_name=signal_name,
                            return_window=return_window,
                            risk_adj=risk_adj,
                            width=width/100,  # 转换为英寸
                            height=height/100
                        )

                        # 生成PR曲线
                        pr_result = sp.plot_precision_recall_curve(
                            signal_name=signal_name,
                            return_window=return_window,
                            risk_adj=risk_adj,
                            width=width/100,  # 转换为英寸
                            height=height/100
                        )

                        # 提取图表对象
                        roc_chart = roc_result[0] if isinstance(roc_result, tuple) else roc_result
                        pr_chart = pr_result[0] if isinstance(pr_result, tuple) else pr_result

                        if roc_chart is None or pr_chart is None:
                            st.error("❌ Failed to generate charts")
                            return

                        # 缓存结果
                        charts = {'roc': roc_chart, 'pr': pr_chart}
                        if 'factor_analysis_results' not in st.session_state:
                            st.session_state.factor_analysis_results = {}
                        st.session_state.factor_analysis_results[cache_key] = charts

                        # 渲染并排图表
                        from .chart_renderer import ChartRenderer
                        ChartRenderer.render_side_by_side_charts(
                            roc_chart, pr_chart,
                            "ROC Curve", "Precision-Recall Curve"
                        )

                        logger.info(f"Successfully generated {self.name}")

                        # 自动保存分析结果到当前ID
                        self._auto_save_to_current_id()

                except Exception as e:
                    st.error(f"❌ {self.name} analysis error: {str(e)}")
                    logger.error(f"{self.name} error: {e}\n{traceback.format_exc()}")

                    # 提供详细的错误信息
                    with st.expander("🔍 Detailed error information", expanded=False):
                        st.code(traceback.format_exc())
                        st.write("**Parameter information:**")
                        st.json({
                            'signal_name': signal_name,
                            'return_window': return_window,
                            'risk_adj': risk_adj,
                            'width': width,
                            'height': height
                        })
