#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
import traceback
import streamlit.components.v1 as components
from typing import List, Dict, Any, Optional

# 添加库路径
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
try:
    import warnings
    import numpy as np

    if not hasattr(np, 'string_'):
        np.string_ = np.bytes_

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*numpy.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*numpy.*")
        warnings.filterwarnings("ignore", message=".*np.string_.*")
        from lib.lib_signal_perf import SignalPerf
        # from lib.lib_signal_perf_ic_extensions import add_ic_extensions
        from lib.utils import format_data, setup_logger
except ImportError as e:
    logger.warning(f"导入lib模块时出现问题，可能是numpy版本冲突: {e}")
    SignalPerf = None
    format_data = None
    setup_logger = lambda: None

# 导入重构的模块
from .parameter_parser import ParameterParser
from .chart_renderer import ChartRenderer, create_download_mhtml_button
from .analysis_sections import AnalysisSection, AutoAnalysisSection, RollingICSection, ROCPRSection
from .data_processor import DataProcessor

# 动态加载IC扩展
import importlib
import sys


setup_logger()
logger = logging.getLogger(__name__)
# add_ic_extensions(SignalPerf)

class FactorAnalysisModule:
    """因子分析模块 - 重构优化版本"""

    def __init__(self):
        self.name = "Factor Analysis"
        self.description = "Factor Analysis, including validity verification, IC analysis, and combination analysis"
        self.initialize_state()
        self.setup_sections()
        self.analysis_id = None

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
                '0._price_signal_raw': False,
                '1._signal_diagnostics_analysis': True,
                '2._ic_decay_analysis': True,
                '3._rolling_ic_analysis': True,
                '4._rolling_ic_statistics': True,
                '5._ic_distribution_analysis': True,
                '6._ic_cumulative_analysis': True,
                '7._ic_autocorrelation_analysis': True,
                '8._person_&_spearman_correlation': True,
                '9._ic_surface_robust': True,
                '10._combined_diagnostics': True,
                '11._mean_return_by_quantile': True,
                '12._rolling_fdm': True,
                '13._turnover': True,
                '14._holding_period': True,
                '15._roc_&_precision-recall_curves': True,
                '16._negative_log_loss_curve': True,
            }
        if 'factor_export_mode' not in st.session_state:
            st.session_state.factor_export_mode = False
        if 'factor_analysis_params' not in st.session_state:
            st.session_state.factor_analysis_params = {}
        # 添加分析结果缓存
        if 'factor_analysis_results' not in st.session_state:
            st.session_state.factor_analysis_results = {}
        if 'remote_df' not in st.session_state:
            st.session_state.remote_df = None
        if 'remote_dataset_name' not in st.session_state:
            st.session_state.remote_dataset_name = None


    def setup_sections(self):
        """设置分析模块"""
        self.sections = [

            # 0. price & signal
            AutoAnalysisSection(
                name="0. Price & Signals",
                icon="📈",
                description="plot price and signals",
                func_name="plot_price_signals_raw_plotly",
                export_defaults={'width': 2000, 'height': 1000}
            ),

            # 1. 信号诊断分析
            AutoAnalysisSection(
                name="1. Signal Diagnostics Analysis",
                icon="📈",
                description="Analyze the characteristics and distribution of a single factor",
                func_name="plot_signal_diagnostics_plotly",
                export_defaults={'width': 2000, 'height': 500}
            ),

            # 2. IC衰减分析
            AutoAnalysisSection(
                name="2. IC Decay Analysis",
                icon="📉",
                description="Analyze the decay of information coefficients in different time windows",
                func_name="plot_ic_decay_summary",
                export_defaults={'lookfwd_days': range(0, 31, 5), 'width': 2000, 'height': 600}
            ),

            # 3. 滚动IC分析
            RollingICSection(),

            # 4. 滚动IC统计
            AutoAnalysisSection(
                name="4. Rolling IC Statistics",
                icon="📊",
                description="Analyze the statistics of rolling information coefficients",
                func_name="calc_rolling_ic_stats",
                export_defaults={
                    'lookfwd_day': 0.4,
                    'sample_days': [30, 90, 180, 360, 720],
                    'sample_freq_days': 30,
                    'risk_adj': True
                }
            ),

            # 5. IC分布分析
            AutoAnalysisSection(
                name="5. IC Distribution Analysis",
                icon="📊",
                description="IC Distribution Histogram, Skewness Kurtosis Statistics",
                func_name="plot_ic_distribution",
                export_defaults={
                    'lookfwd_day': 25,
                    'width': 450,
                    'height': 600
                }
            ),

            # 6. IC累积分析
            AutoAnalysisSection(
                name="6. IC Cumulative Analysis",
                icon="📈",
                description="IC Time Series, Cumulative IC",
                func_name="plot_ic_cumulative_ir",
                export_defaults={

                    'lookfwd_day': 25,
                    'sample_day': 30,
                    'width': 1600,
                    'height': 600
                }
            ),

            # 7. IC自相关分析
            AutoAnalysisSection(
                name="7. IC Autocorrelation Analysis",
                icon="🔄",
                description="IC Autocorrelation Coefficient and Confidence Interval",
                func_name="plot_ic_autocorrelation",
                export_defaults={
                    'lookfwd_day': 25,
                    'sample_day': 30,
                    'max_lags': 20,
                    'width': 1600,
                    'height': 600
                }
            ),

            # 8. Person & Spearman correlation
            AutoAnalysisSection(
                name="8. Person & Spearman Correlation",
                icon="📊",
                description="Person & Spearman Correlation",
                func_name="plot_correlation_matrix",
                export_defaults={
                    'lookfwd_days': [25],
                    'begin_time': '2023-03-01'
                }
            ),

            # 9. IC Surface Robust
            AutoAnalysisSection(
                name="9. IC Surface Robust",
                icon="📊",
                description="IC Surface Robust",
                func_name="plot_ic_surface_robust",
                export_defaults={
                    'lookback_days': range(90, 720, 180),
                    'lookfwd_days': range(1, 35, 5),
                    'width': 1400,
                    'height': 600
                }
            ),

            # 10. 综合诊断分析
            AutoAnalysisSection(
                name="10. Combined Diagnostics",
                icon="🔍",
                description="Combined Diagnostics of Factors and Returns",
                func_name="plot_combined_diagnostics",
                export_defaults={
                    'return_window': 25,
                    'width': 1600,
                    'height': 600,
                    'begin_time': None,
                    'end_time': None,
                    'risk_adj': True
                }
            ),

            # 11. Mean Return By Quantile
            AutoAnalysisSection(
                name="11. Mean Return By Quantile",
                icon="📊",
                description="Mean Return By Quantile",
                func_name="plot_mean_return_by_quantile",
                export_defaults={
                    'stat': 'mean',
                    'q': 5,
                    'line_type': 'bar',
                    'by_year': False,
                    'lookfwd_days': [0.4,1,3],
                }
            ),

            # 12. Rolling FDM
            AutoAnalysisSection(
                name="12. Rolling FDM",
                icon="📊",
                description="Rolling FDM",
                func_name="plot_rolling_fdm",
                export_defaults={
                    'width': 1600,
                    'height': 600,
                    'window': 365,
                }
            ),

            # 13. Turnover
            AutoAnalysisSection(
                name="13. Turnover",
                icon="📊",
                description="Turnover",
                func_name="calc_turnover",
                export_defaults={
                    'height': 600,
                    'width': 1600,
                }
            ),

            # 14. Holding Period
            AutoAnalysisSection(
                name="14. Holding Period",
                icon="📊",
                description="Holding Period",
                func_name="calculate_signal_holding_period_by_sign",
                export_defaults={
                    'unit': 'days',
                    'width': 1200,
                    'height': 600
                }
            ),

            # 15. ROC和PR曲线组合
            ROCPRSection(),

            # 16. Negative Log Loss Curve
            AutoAnalysisSection(
                name="16. Negative Log Loss Curve",
                icon="📊",
                description="Negative Log Loss Curve",
                func_name="plot_negative_log_loss_plotly",
                export_defaults={
                    'width': 1200,
                    'height': 600
                }
            ),

        ]

        print(f"st.session_state.factor_signal_list: {st.session_state.factor_signal_list}")


    def render(self):
        """渲染因子分析模块界面"""
        st.markdown("## 🔍 Factor Analysis Module")
        st.markdown("**Deep Factor Analysis, including validity verification, IC analysis, and combination analysis**")

        self.render_data_loading()
        self.render_analysis_dashboard()

        if st.session_state.factor_signal_perf is None:
            self.render_welcome_message()

    def _on_file_upload_change(self):
        # 调用exporter 清理对应id之前的缓存数据
        has_current_id = hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id
        print(f"has_current_id: {has_current_id}")
        if has_current_id:
            try:
                from modules.factor_export import clean_cache_by_id
                clean_cache_by_id(st.session_state.current_analysis_id)
                st.session_state.factor_default_signal = None
                st.session_state.factor_signal_list = None
                st.session_state.factor_analysis_params = {}
            except ImportError:
                st.warning("Factor export module not available")
            except Exception as e:
                st.error(f"❌ Error cleaning cache: {e}")


    def render_data_loading(self):
        """渲染数据加载界面"""
        with st.expander("📁 Data Loading", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload CSV file",
                    type=['csv'],
                    help="Upload a CSV file containing factor data",
                    key="factor_file_upload",
                    on_change=self._on_file_upload_change
                )

            with col2:
                from modules.db_utils import get_all_data_sources
                import paramiko
                import pandas as pd
                import io
                import datetime

                # 1. 选择Remote配置
                remote_configs = [c for c in get_all_data_sources() if c['type'] == 'Remote Server']
                if not remote_configs:
                    st.warning("No remote server config found in Data Management module")
                else:
                    config_names = [f"{c['source_name']}@{c['ip']}:{c['port']}{c['path']}" for c in remote_configs]
                    selected_idx = st.selectbox("Select Remote Config", range(len(config_names)), format_func=lambda i: config_names[i], key="remote_config_select")
                    config = remote_configs[selected_idx]

                    # 2. 连接远程服务器并递归获取csv文件列表
                    all_csv_files = []
                    try:
                        ssh = paramiko.SSHClient()
                        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                        ssh.connect(config['ip'], port=int(config['port']), username=config['username'], password=config['password'])
                        sftp = ssh.open_sftp()
                        def list_csv_files_recursive(remote_path, max_depth=3, current_depth=0):
                            csv_files = []
                            if current_depth >= max_depth:
                                return csv_files
                            try:
                                file_list = sftp.listdir_attr(remote_path)
                                for file_attr in file_list:
                                    file_path = f"{remote_path.rstrip('/')}/{file_attr.filename}"
                                    if file_attr.st_mode & 0o040000:
                                        csv_files.extend(list_csv_files_recursive(file_path, max_depth, current_depth + 1))
                                    elif file_attr.filename.lower().endswith('.csv'):
                                        csv_files.append({
                                            'path': file_path,
                                            'name': file_attr.filename,
                                            'size': file_attr.st_size,
                                            'modified': file_attr.st_mtime
                                        })
                            except Exception as e:
                                st.warning(f"无法访问目录 {remote_path}: {e}")
                            return csv_files
                        all_csv_files = list_csv_files_recursive(config['path'])
                        sftp.close()
                        ssh.close()
                    except Exception as e:
                        st.error(f"连接远程服务器失败: {e}")
                        all_csv_files = []

                    # 3. 搜索输入框、selectbox、加载按钮始终渲染
                    search_text = st.text_input('🔍 search csv files(support fuzzy matching)', '', key='remote_csv_search')
                    filtered_files = [f for f in all_csv_files if search_text.lower() in f['name'].lower()]
                    if filtered_files:
                        file_options = [f"{f['name']} ({f['path']})" for f in filtered_files]
                        # 用 session_state 记住选中项
                        if 'remote_csv_select' not in st.session_state or st.session_state['remote_csv_select'] >= len(file_options):
                            st.session_state['remote_csv_select'] = 0
                        selected_file_idx = st.selectbox(
                            "Select CSV File",
                            range(len(file_options)),
                            format_func=lambda i: file_options[i],
                            key="remote_csv_select"
                        )
                        selected_file = filtered_files[selected_file_idx]
                        if st.button("📥 Load Selected CSV", key="load_selected_remote_csv"):
                            try:
                                ssh = paramiko.SSHClient()
                                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                                ssh.connect(config['ip'], port=int(config['port']), username=config['username'], password=config['password'])
                                sftp = ssh.open_sftp()
                                with sftp.open(selected_file['path'], 'r') as remote_f:
                                    csv_bytes = remote_f.read()
                                    df = pd.read_csv(io.BytesIO(csv_bytes))
                                sftp.close()
                                ssh.close()
                                dataset_name = f"remote_{selected_file['name']}"
                                st.session_state.remote_df = df
                                st.session_state.remote_dataset_name = dataset_name
                                st.success(f"✅ Loaded {selected_file['name']} from remote server.")
                            except Exception as e:
                                st.error(f"❌ 加载远程CSV失败: {e}")
                    else:
                        st.info("No matching CSV files.")

            with col3:
                # download csv by id
                query_params = st.query_params
                if 'id' in query_params:
                    analysis_id = query_params['id']
                    if analysis_id:
                        try:
                            from modules.factor_export import downlaod_csv_by_id
                            csv_file = downlaod_csv_by_id(analysis_id)
                            if csv_file:
                                with open(csv_file, "rb") as f:
                                    st.download_button(
                                        label="Download CSV",
                                        data=f.read(),
                                        file_name=csv_file.name,
                                        mime="text/csv"
                                    )
                            else:
                                st.error("No available data to download")
                        except ImportError:
                            st.warning("Factor export module not available")

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    self.process_uploaded_data(df, uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.error(f"Factor analysis error: {e}\n{traceback.format_exc()}")
            elif st.session_state.remote_df is not None:
                self.process_uploaded_data(st.session_state.remote_df, st.session_state.remote_dataset_name)
            # else:
                # self.process_uploaded_data(None, None)


        # 添加文件列表功能
        with st.expander("📂 Saved Analysis Files", expanded=False):
            try:
                from modules.factor_export import render_analysis_files_list
                render_analysis_files_list()
            except ImportError:
                st.warning("Factor export module not available")
            except Exception as e:
                st.error(f"❌ Error loading analysis files: {e}")

    def process_uploaded_data(self, df, filename):
        """处理上传数据"""
        st.success(f"✅ File loaded successfully: {filename}")
        st.info(f"username: {st.session_state.username}")
        st.info(f"Data shape: {df.shape}")



        # 显示数据预览
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
            st.markdown("**Column Information:**")

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
                'ColumnName': df.columns.tolist(),
                'DataType': [str(dtype) for dtype in df.dtypes],
                'Non-Null Count': df.count().tolist(),
                'Null Count': df.isnull().sum().tolist(),
                'Example Value': example_values
            })
            st.dataframe(col_info, use_container_width=True)

        # 列配置
        st.subheader("📋 Column Configuration")
        col1, col2, col3 = st.columns(3)

        all_columns = df.columns.tolist()

        with col1:
            # 自动检测价格列
            price_col, signal_cols = DataProcessor.detect_columns(df)

            price_col = st.selectbox(
                "Price Column",
                options=all_columns,
                index=all_columns.index(price_col) if price_col in all_columns else 0,
                help="Select the column containing price data",
                key="factor_price_col"
            )

        with col2:
            signal_cols = st.multiselect(
                "Factor Column",
                options=all_columns,
                default=signal_cols if signal_cols else [],
                help="Select the column containing factor data",
                key="factor_signal_cols"
            )

        with col3:
            labeling_method = st.selectbox(
                "Labeling Method",
                options=['point', 'triple'],
                index=0,
                help="Select the factor labeling method",
                key="factor_labeling_method"
            )

        # 保存选项配置
        st.subheader("💾 Save Options")
        col_save1, col_save2 = st.columns(2)

        with col_save1:
            # 检查是否有当前分析ID
            has_current_id = hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id

            if has_current_id:
                overwrite_mode = st.checkbox(
                    f"🔄 Overwrite previous analysis (ID: {st.session_state.current_analysis_id})",
                    value=True,
                    help="If checked, will overwrite the existing analysis. If unchecked, will create a new analysis with new ID.",
                    key="overwrite_analysis"
                )
            else:
                st.info("📝 Will create new analysis ID")
                overwrite_mode = False

            # 添加描述输入框
            current_description = getattr(st.session_state, 'current_analysis_description', '')
            analysis_description = st.text_input(
                "📝 Analysis Description",
                value=current_description,
                placeholder="Enter a description for this analysis...",
                help="Provide a meaningful description for this analysis",
                key="analysis_description_input"
            )

        with col_save2:
            if has_current_id:
                current_url = st.get_option("server.baseUrlPath") or ""
                if hasattr(st, 'get_option') and st.get_option("server.port"):
                    port = st.get_option("server.port")
                    share_url = f"http://h.adpolitan.com:{port}?page=factor&id={st.session_state.current_analysis_id}"
                else:
                    share_url = f"{current_url}?page=factor&id={st.session_state.current_analysis_id}"

                st.markdown("**🔗 Current Share Link:**")
                st.code(share_url, language=None)

        # 数据验证
        validation_errors = []
        if not price_col:
            validation_errors.append("❌ Please select Price Column")
        if not signal_cols:
            validation_errors.append("❌ Please select at least one Factor Column")
        if price_col in signal_cols:
            validation_errors.append("⚠️ Price Column should not be selected as a Factor Column")

        # 检查数据类型
        if price_col and not pd.api.types.is_numeric_dtype(df[price_col]):
            validation_errors.append(f"❌ Price Column '{price_col}' is not a numeric type")

        for col in signal_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_errors.append(f"⚠️ Factor Column '{col}' is not a numeric type")

        # 显示验证结果
        if validation_errors:
            st.error("**Data Validation Issues:**")
            for error in validation_errors:
                st.write(error)

        # 生成分析报告
        if st.button("🚀 Generate Factor Analysis Report", type="primary", key="generate_factor_report"):
            if price_col and signal_cols:
                try:
                    with st.spinner("Generating Factor Analysis Report..."):
                        st.info("📊 Starting data processing...")

                        # 数据预处理
                        logger.info(f"Starting data formatting, original data shape: {df.shape}")

                        # 验证和准备数据
                        validated_df = DataProcessor.validate_and_prepare_data(df.copy(), price_col, signal_cols)
                        logger.info(f"Data validation completed, validated data shape: {validated_df.shape}")

                        # 应用format_data格式化
                        formatted_df = format_data(validated_df)
                        logger.info(f"Data formatting completed, formatted data shape: {formatted_df.shape}")

                        # 数据清理
                        initial_len = len(formatted_df)
                        formatted_df = formatted_df.dropna(subset=[price_col] + signal_cols)
                        cleaned_len = len(formatted_df)
                        dropped_rows = initial_len - cleaned_len

                        if dropped_rows > 0:
                            st.warning(f"⚠️ {dropped_rows} rows containing null values were removed ({dropped_rows/initial_len:.1%})")

                        if cleaned_len < 10:
                            st.error("❌ The cleaned data is too small to perform effective analysis")
                            return

                        st.info("🔧 Initializing SignalPerf...")

                        # 创建SignalPerf实例
                        sp = SignalPerf(
                            mode='local',
                            data=formatted_df,
                            price_col=price_col,
                            signal_cols=signal_cols,
                            labeling_method=labeling_method
                        )

                        logger.info("SignalPerf instance created successfully")

                        # 存储到session state
                        st.session_state.factor_signal_perf = sp
                        st.session_state.factor_signal_list = signal_cols
                        st.session_state.factor_default_signal = signal_cols[0]



                        # 保存数据到文件（自动生成ID或使用现有ID）
                        st.info("💾 Saving analysis data...")
                        try:
                            from modules.factor_export import auto_save_analysis_data
                            analysis_id = auto_save_analysis_data(
                                validated_df,
                                price_col,
                                signal_cols,
                                labeling_method,
                                overwrite_mode=overwrite_mode,
                                description=analysis_description or None
                            )
                            if analysis_id:
                                st.session_state.current_analysis_id = analysis_id
                                st.session_state.current_analysis_description = analysis_description or f"default_{analysis_id}"
                                if overwrite_mode and has_current_id:
                                    st.success(f"✅ Analysis data updated for ID: `{analysis_id}`")
                                else:
                                    st.success(f"✅ Analysis data saved with new ID: `{analysis_id}`")
                        except ImportError:
                            st.warning("Factor export module not available")
                        except Exception as e:
                            st.error(f"❌ Failed to save analysis data: {e}")

                        st.success("✅ Factor Analysis Report Generated Successfully!")
                        st.info(f"📈 {len(signal_cols)} factors loaded: {', '.join(signal_cols)}")

                        # 初始化section states并全部展开
                        if 'factor_section_states' not in st.session_state:
                            st.session_state.factor_section_states = {}

                        # 获取所有section的state_key并展开
                        section_keys = [section.state_key for section in self.sections if hasattr(section, 'state_key')]
                        for key in section_keys:
                            st.session_state.factor_section_states[key] = True

                        # 标记刚生成报告
                        st.session_state._just_generated_report = True
                        self.run_all_analysis()

                except Exception as e:
                    st.error(f"❌ Error occurred while generating report: {str(e)}")

                    # 详细的错误信息
                    with st.expander("🔍 Detailed Error Information", expanded=False):
                        st.code(traceback.format_exc())

                    logger.error(f"Factor analysis error: {e}\n{traceback.format_exc()}")
            else:
                st.warning("⚠️ Please select Price Column and at least one Factor Column")

    def render_analysis_dashboard(self):
        """渲染分析仪表板"""
        # 检查URL参数
        query_params = st.query_params
        if 'id' in query_params:
            analysis_id = query_params['id']
            st.session_state.current_analysis_id = analysis_id
            if hasattr(st.session_state, '_url_loaded_id'):
                if st.session_state._url_loaded_id != analysis_id:
                    st.session_state._url_loaded_id = analysis_id
                    self.load_analysis_from_url(analysis_id)
            else:
                st.session_state._url_loaded_id = analysis_id
                self.load_analysis_from_url(analysis_id)

        sp = st.session_state.factor_signal_perf
        signal_list = st.session_state.factor_signal_list
        default_signal = st.session_state.factor_default_signal

        # 安全检查
        if sp is None:
            return

        if not signal_list:
            st.error("❌ Factor Column list is empty, please regenerate the analysis report")
            return

        # 显示成功提示
        if hasattr(st.session_state, '_just_generated_report') and st.session_state._just_generated_report:
            st.success("🎉 Factor analysis report generated! You can expand the analysis modules below to view the results.")
            st.session_state._just_generated_report = False

        # 全局控制
        st.markdown("### 📊 Factor Analysis Dashboard")
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

        with col1:
            # Create a dropdown to select the current factor
            selected_signal = st.selectbox(
                "**Current Factor(default factor):**",
                options=signal_list,
                index=signal_list.index(default_signal) if default_signal in signal_list else 0,
                key="factor_selector"
            )

            # Update the default signal if changed
            if selected_signal != default_signal:
                st.session_state.factor_default_signal = selected_signal
                st.rerun()
            st.markdown(f"**Number of Factors:** {len(signal_list)}")

        with col2:
            if st.button("📖 Expand All", key="factor_expand_all"):
                if 'factor_section_states' not in st.session_state:
                    st.session_state.factor_section_states = {}
                section_keys = [section.state_key for section in self.sections if hasattr(section, 'state_key')]
                for key in section_keys:
                    st.session_state.factor_section_states[key] = True
                st.rerun()

        with col3:
            if st.button("📕 Collapse All", key="factor_collapse_all"):
                if 'factor_section_states' not in st.session_state:
                    st.session_state.factor_section_states = {}
                section_keys = [section.state_key for section in self.sections if hasattr(section, 'state_key')]
                for key in section_keys:
                    st.session_state.factor_section_states[key] = False

                print("st.session_state.factor_section_states:", st.session_state.factor_section_states.keys())

                st.rerun()

        with col4:
            if st.button("📄 Export Mode" if not st.session_state.factor_export_mode else "🔧 Edit Mode",
                        key="toggle_factor_export_mode"):
                st.session_state.factor_export_mode = not st.session_state.factor_export_mode
                st.rerun()

        with col5:
            # 根据是否有现有ID显示不同的按钮文本
            has_current_id = hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id
            if has_current_id:
                save_button_text = f"💾 Save/Share Analysis ({st.session_state.current_analysis_id})"
            else:
                save_button_text = "💾 Save/Share Analysis Results"

            if st.button(save_button_text, key="save_factor_analysis"):
                self.save_analysis_results()

        with col6:
            if st.button("🖨️ Print/Export", help="Use browser print function to export", key="print_page_btn"):
                buttons_js = """
                <script>
                function printPage() {
                    try {
                        window.parent.print();
                    } catch (e) {
                        alert('Print function is not available');
                    }
                }
                </script>
                <div style="margin: 10px 0;">
                    <button onclick="printPage()" style="
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                        cursor: pointer;
                    ">Print PDF</button>
                </div>
                """
                components.html(buttons_js, height=60)

        # Export mode indicator
        if st.session_state.factor_export_mode:
            st.info("📄 **Export Mode Enabled** - Displaying cached results only, parameter controls hidden", icon="📄")


        # Analysis Controls
        st.markdown("### 🚀 Analysis Controls")
        col_run1, col_run2 = st.columns([1, 1])

        with col_run1:
            if st.button("🚀 Run All Analysis", type="primary", key="run_all_analysis_dashboard"):
                self.run_all_analysis()

        with col_run2:
            if st.button("🧹 Clear All Results", key="clear_all_results_dashboard"):
                if 'factor_analysis_results' in st.session_state:
                    st.session_state.factor_analysis_results.clear()
                st.success("✅ All analysis results cleared")
                st.rerun()

        st.markdown("---")

        # 渲染所有分析模块
        try:
            for section in self.sections:
                section.render(sp, signal_list, default_signal, st.session_state.factor_export_mode)
        except Exception as e:
            st.error(f"❌ Error occurred while rendering analysis modules: {str(e)}")
            logger.error(f"Dashboard render error: {e}\n{traceback.format_exc()}")

            # 提供重置选项
            if st.button("🔄 Reset Analysis State", key="reset_analysis_state"):
                st.session_state.factor_signal_perf = None
                st.session_state.factor_signal_list = []
                st.session_state.factor_default_signal = None
                st.session_state.factor_analysis_results = {}
                st.rerun()
    def load_analysis_from_url(self, analysis_id: str):
        """从URL参数加载分析"""
        try:
            from modules.factor_export import load_analysis_by_id
            if load_analysis_by_id(analysis_id):
                st.success(f"✅ Analysis loaded from URL: `{analysis_id}`")

                # 使用load的参数覆盖所有函数的form_data，然后执行分析
                # st.info("🚀 Running analysis with loaded parameters...")
                self.run_all_analysis()
            else:
                st.error(f"❌ Failed to load analysis from URL: `{analysis_id}`")
        except ImportError:
            st.warning("Factor export module not available")
        except Exception as e:
            st.error(f"❌ Failed to load analysis from URL: {e}")

    def render_welcome_message(self):
        """渲染欢迎消息"""
        st.info("👈 Please load data above to start factor analysis")

    def save_analysis_results(self):
        """保存分析结果"""
        try:
            from modules.factor_export import save_analysis_with_id
            analysis_id = save_analysis_with_id()
            if analysis_id:
                st.success(f"✅ Analysis saved with ID: `{analysis_id}`")
                st.session_state.last_analysis_id = analysis_id
        except ImportError:
            st.warning("Factor export module not available")
        except Exception as e:
            st.error(f"❌ Failed to save analysis: {e}")

    def run_all_analysis(self):
        """运行所有分析"""
        if st.session_state.factor_signal_perf is None:
            st.error("❌ No data loaded")
            return

        sp = st.session_state.factor_signal_perf
        signal_list = st.session_state.factor_signal_list
        default_signal = st.session_state.factor_default_signal

        # 检查是否有保存的参数
        has_saved_params = 'factor_analysis_params' in st.session_state and st.session_state.factor_analysis_params
        if has_saved_params:
            st.info(f"🔧 Using saved parameters for analysis functions")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            total_sections = len(self.sections)

            for i, section in enumerate(self.sections):
                progress = (i + 1) / total_sections
                status_text.text(f"Running {section.name}... ({i+1}/{total_sections})")
                progress_bar.progress(progress)

                print(f"Running {section.name}... ({i+1}/{total_sections})")

                try:
                    if isinstance(section, AutoAnalysisSection):
                        self._run_auto_analysis_section(section, sp, signal_list, default_signal)
                    elif isinstance(section, RollingICSection):
                        self._run_rolling_ic_section(section, sp, signal_list, default_signal)
                    elif isinstance(section, ROCPRSection):
                        self._run_roc_pr_section(section, sp, signal_list, default_signal)

                except Exception as e:
                    logger.error(f"Error in section {section.name}: {e}")
                    continue

            status_text.text("✅ All analyses completed!")
            progress_bar.progress(1.0)

            # 自动保存分析结果到现有ID
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
                        st.success(f"🎉 All analyses completed and results saved to ID: `{st.session_state.current_analysis_id}`")
                except ImportError:
                    st.warning("Factor export module not available for auto-save")
                except Exception as e:
                    st.error(f"❌ Failed to auto-save results: {e}")
            else:
                st.success("🎉 All analyses completed successfully!")

        except Exception as e:
            st.error(f"❌ Error during batch analysis: {e}")
            logger.error(f"Batch analysis error: {e}\n{traceback.format_exc()}")

        finally:
            # 清理进度条
            progress_bar.empty()
            status_text.empty()

    def _run_auto_analysis_section(self, section: AutoAnalysisSection, sp, signal_list: List[str], default_signal: str):
        """运行自动分析模块"""
        try:
            # 准备表单数据
            func = getattr(sp, section.main_func_name)
            signature = ParameterParser.get_function_signature(func)

            # 获取保存的参数配置
            saved_params = section.get_saved_parameters()

            # 参数优先级：保存的参数 > export_defaults > 函数默认值
            form_data = {}

            # 首先用函数的默认值填充所有参数
            for name, param_info in signature.items():
                form_data[name] = param_info['default']

            # 然后用export_defaults覆盖指定的参数
            if section.export_defaults:
                form_data.update(section.export_defaults)

            # 最后用保存的参数覆盖（最高优先级）
            if saved_params:
                form_data.update(saved_params)
                logger.info(f"Using saved parameters for {section.main_func_name}: {saved_params}")

            # 处理特殊参数
            if 'signal_name' in form_data and form_data['signal_name'] is None:
                form_data['signal_name'] = default_signal
            if 'signal_names' in form_data and form_data['signal_names'] is None:
                form_data['signal_names'] = signal_list

            # 执行分析
            result = section.execute_analysis(sp, section.main_func_name, form_data)

            # 缓存结果
            cache_key = f"{section.main_func_name}_{default_signal}"
            if 'factor_analysis_results' not in st.session_state:
                st.session_state.factor_analysis_results = {}
            st.session_state.factor_analysis_results[cache_key] = result

        except Exception as e:
            logger.error(f"Error in auto analysis section {section.name}: {e}")
            raise

    def _run_rolling_ic_section(self, section: RollingICSection, sp, signal_list: List[str], default_signal: str):
        """运行滚动IC分析模块"""
        try:
            # 获取保存的参数配置
            saved_params = section.get_saved_parameters()

            # 导出模式下的默认参数
            form_data = {
                'signal_name': default_signal,
                'lookfwd_day': 25,
            }

            # 如果有保存的参数，使用保存的参数
            if saved_params:
                form_data.update(saved_params)
                logger.info(f"Using saved parameters for plot_rolling_ic: {saved_params}")
                # 确保signal_name正确
                if 'signal_name' not in form_data or form_data['signal_name'] is None:
                    form_data['signal_name'] = default_signal

            # 执行分析
            result = section.execute_analysis(sp, "plot_rolling_ic", form_data)

            # 缓存结果
            cache_key = f"rolling_ic_{default_signal}"
            if 'factor_analysis_results' not in st.session_state:
                st.session_state.factor_analysis_results = {}
            st.session_state.factor_analysis_results[cache_key] = result

        except Exception as e:
            logger.error(f"Error in rolling IC section: {e}")
            raise
    def _run_roc_pr_section(self, section: ROCPRSection, sp, signal_list: List[str], default_signal: str):
        """运行ROC和PR曲线分析模块"""
        try:
            # 获取保存的参数配置
            saved_params = section.get_saved_parameters()

            # 导出模式下的默认参数
            form_data = {
                'signal_name': default_signal,
                'return_window': 7.0,
                'risk_adj': True,
                'width': 300,
                'height': 300
            }

            # 如果有保存的参数，使用保存的参数
            if saved_params:
                form_data.update(saved_params)
                logger.info(f"Using saved parameters for ROC/PR curves: {saved_params}")
                # 确保signal_name正确
                if 'signal_name' not in form_data or form_data['signal_name'] is None:
                    form_data['signal_name'] = default_signal

            # 生成ROC曲线
            roc_result = sp.plot_roc_curve(
                signal_name=form_data['signal_name'],
                return_window=form_data['return_window'],
                risk_adj=form_data['risk_adj'],
                width=form_data['width']/100,  # 转换为英寸
                height=form_data['height']/100
            )

            # 生成PR曲线
            pr_result = sp.plot_precision_recall_curve(
                signal_name=form_data['signal_name'],
                return_window=form_data['return_window'],
                risk_adj=form_data['risk_adj'],
                width=form_data['width']/100,  # 转换为英寸
                height=form_data['height']/100
            )

            # 提取图表对象
            roc_chart = roc_result[0] if isinstance(roc_result, tuple) else roc_result
            pr_chart = pr_result[0] if isinstance(pr_result, tuple) else pr_result

            if roc_chart is None or pr_chart is None:
                logger.error("Failed to generate ROC or PR charts")
                return

            # 缓存结果
            charts = {'roc': roc_chart, 'pr': pr_chart}
            cache_key = f"roc_pr_{default_signal}"
            if 'factor_analysis_results' not in st.session_state:
                st.session_state.factor_analysis_results = {}
            st.session_state.factor_analysis_results[cache_key] = charts

        except Exception as e:
            logger.error(f"Error in ROC and PR section: {e}")
            raise
