#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np

if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*numpy.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*numpy.*")
        warnings.filterwarnings("ignore", message=".*np.string_.*")
        import importlib
        import lib.lib_signal_perf
        importlib.reload(lib.lib_signal_perf)
        from lib.lib_signal_perf import SignalPerf

        # from lib.lib_signal_perf_ic_extensions import add_ic_extensions
        from lib.utils import format_data, setup_logger
except ImportError as e:
    SignalPerf = None
    format_data = None
    setup_logger = lambda: None

from .parameter_parser import ParameterParser
from .analysis_sections import AutoAnalysisSection
from .data_processor import DataProcessor

setup_logger()
logger = logging.getLogger(__name__)

class FactorAnalysisModule:
    """因子分析模块 - 重构优化版本"""

    def __init__(self):
        self.name = "Factor Analysis"
        self.description = "Factor Analysis, including validity verification, IC analysis, and combination analysis"
        self.analysis_id = None
        self.initialize_state()
        self.setup_sections()

    # ==================== 初始化相关方法 ====================
    def initialize_state(self):
        """初始化模块状态"""
        # Core state variables
        state_defaults = {
            'factor_signal_perf': None,
            'factor_signal_list': [],
            'factor_default_signal': None,
            'factor_export_mode': False,
            'factor_analysis_params': {},
            'factor_analysis_results': {},
            'remote_df': None,
            'remote_dataset_name': None,
            'dataframe': pd.DataFrame(),
            'start_date': None,
            'end_date': None,
            'formatted_df': pd.DataFrame(),
            'price_col': None,
            'signal_cols': None,
            'labeling_method': None
        }

        for key, default_value in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

        # Section states with expanded defaults
        if 'factor_section_states' not in st.session_state:
            st.session_state.factor_section_states = {
                '_price_signal_raw': False,
                '_signal_diagnostics_analysis': True,
                '_ic_decay_analysis': True,
                '_rolling_ic_analysis': True,
                '_rolling_ic_all_types_one_sample': True,
                '_rolling_ic_statistics': True,
                '_ic_distribution_analysis': True,
                '_ic_cumulative_analysis': True,
                '_ic_autocorrelation_analysis': True,
                '_person_&_spearman_correlation': True,
                '_ic_surface_robust': True,
                '_combined_diagnostics': True,
                '_mean_return_by_quantile': True,
                '_rolling_fdm': True,
                '_turnover': True,
                '_holding_period': True,
                '_roc_&_precision-recall_curves': True,
                '_negative_log_loss_curve': True,
                '_simple_backtest_account_curve': True,
                '_backtest_account_curve_&_rolling_ic': True,
            }

    def setup_sections(self):
        """设置分析模块"""
        self.sections = [
            # 0. Price & Signals
            AutoAnalysisSection(
                name="Price & Signals", icon="📈", description="plot price and signals",
                func_name="plot_price_signals_raw", index=0,
                export_defaults={'width': 1200, 'height': 600}
            ),
            # 1. Signal Diagnostics
            AutoAnalysisSection(
                name="Signal Diagnostics Analysis", icon="📈",
                description="Analyze the characteristics and distribution of a single factor",
                func_name="plot_signal_diagnostics_plotly", index=1,
                export_defaults={'width': 1200, 'height': 400}
            ),
            # 2. IC Decay Analysis
            AutoAnalysisSection(
                name="IC Decay Analysis", icon="📉",
                description="Analyze the decay of information coefficients in different time windows",
                func_name="plot_ic_decay_summary", index=2,
                export_defaults={'lookfwd_days': range(0, 35, 5), 'width': 1200, 'height': 500}
            ),
            # 3. Backtest & Rolling IC
            AutoAnalysisSection(
                name="Backtest Account Curve & Rolling IC", icon="📊",
                description="Backtest Account Curve & Rolling IC",
                func_name="plot_bt_account_curve_and_rolling_ic", index=3,
                export_defaults={'width': 1600, 'height': 600, 'begin_time': None, 'end_time': None, 'fdm': -1, 'max_long': 2, "max_short": -2}
            ),
            # 4. Rolling IC All Types
            AutoAnalysisSection(
                name="Rolling IC All Types One Sample", icon="📊",
                description="Rolling IC All Types One Sample",
                func_name="plot_rolling_ic_all_types_one_sample", index=4,
                export_defaults={'width': 1600, 'height': 300}
            ),
            # 5. Combined Diagnostics
            AutoAnalysisSection(
                name="Combined Diagnostics", icon="🔍",
                description="Combined Diagnostics of Factors and Returns",
                func_name="plot_combined_diagnostics", index=5,
                export_defaults={'return_window': 25, 'width': 1000, 'height': 400, 'begin_time': None, 'end_time': None, 'risk_adj': False}
            ),
            # 6. Rolling IC Statistics
            AutoAnalysisSection(
                name="Rolling IC Statistics", icon="📊",
                description="Analyze the statistics of rolling information coefficients",
                func_name="calc_rolling_ic_stats", index=6,
                export_defaults={'lookfwd_day': 0.4, 'sample_days': [30, 90, 180, 360, 720], 'sample_freq_days': 30, 'risk_adj': False}
            ),
            # 7. IC Distribution
            AutoAnalysisSection(
                name="IC Distribution Analysis", icon="📊",
                description="IC Distribution Histogram, Skewness Kurtosis Statistics",
                func_name="plot_ic_distribution", index=7,
                export_defaults={'lookfwd_day': 25, 'width': 600, 'height': 400, 'nbins': 20}
            ),
            # 8. IC Autocorrelation
            AutoAnalysisSection(
                name="IC Autocorrelation Analysis", icon="🔄",
                description="IC Partial Autocorrelation and Autocorrelation",
                func_name="plot_ic_pacf_acf", index=8,
                export_defaults={'lookfwd_day': 25, 'sample_day': 30, 'max_lags': 20, 'width': 1200, 'height': 300}
            ),
            # 9. Correlation Matrix
            AutoAnalysisSection(
                name="Person & Spearman Correlation", icon="📊",
                description="Person & Spearman Correlation",
                func_name="plot_correlation_matrix", index=9,
                export_defaults={'lookfwd_days': [25], 'width': 8, 'height': 3}
            ),
            # 10. IC Surface Robust
            AutoAnalysisSection(
                name="IC Surface Robust", icon="📊", description="IC Surface Robust",
                func_name="plot_ic_surface_robust", index=10,
                export_defaults={'lookback_days': [30, 60, 90, 180, 360, 720], 'lookfwd_days': range(0, 35, 5), 'width': 1200, 'height': 400}
            ),
            # 11. Mean Return By Quantile
            AutoAnalysisSection(
                name="Mean Return By Quantile", icon="📊", description="Mean Return By Quantile",
                func_name="plot_mean_return_by_quantile", index=11,
                export_defaults={'stat': 'mean', 'q': 5, 'line_type': 'bar', 'by_year': False, 'lookfwd_days': [0.4,1,3], 'risk_adj': False, 'height': 300, 'width': 600}
            ),
            # 12. Rolling FDM
            AutoAnalysisSection(
                name="Rolling FDM", icon="📊", description="Rolling FDM",
                func_name="plot_rolling_fdm", index=12,
                export_defaults={'width': 1200, 'height': 300, 'window': 365}
            ),
            # 13. Negative Log Loss
            AutoAnalysisSection(
                name="Negative Log Loss Curve", icon="📊", description="Negative Log Loss Curve",
                func_name="plot_negative_log_loss_boxplot", index=13,
                export_defaults={'width': 800, 'height': 400}
            ),
            # 14. ROC and PR Curves
            AutoAnalysisSection(
                name="ROC and PR Curves", icon="📊", description="ROC and PR Curves",
                func_name="plot_roc_pr_curve", index=14,
                export_defaults={'width': 700, 'height': 350, 'return_window': 7, 'risk_adj': False}
            ),
            # 15. Turnover
            AutoAnalysisSection(
                name="Turnover", icon="📊", description="Turnover",
                func_name="calc_turnover", index=15,
                export_defaults={'width': 1200, 'height': 300}
            ),
            # 16. Holding Period
            AutoAnalysisSection(
                name="Holding Period", icon="📊", description="Holding Period",
                func_name="calculate_signal_holding_period_by_sign", index=16,
                export_defaults={'unit': 'days', 'width': 1200, 'height': 300}
            ),
        ]

    # ==================== 主渲染方法 ====================
    def render(self):
        st.markdown("""
        <style>
        # [data-testid="stImageContainer"] {
        #     width: 60% !important;
        #     # min-width: 1000px !important;
        #     max-width: none !important;
        # }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("## 🔍 Factor Analysis Module")
        st.markdown("**Deep Factor Analysis, including validity verification, IC analysis, and combination analysis**")

        self.render_data_loading()
        self.render_analysis_dashboard()

        if st.session_state.factor_signal_perf is None:
            self.render_welcome_message()

    # ==================== 数据加载相关方法 ====================
    def render_data_loading(self):
        """渲染数据加载界面"""
        # 检查URL参数
        self._handle_url_parameters()

        with st.expander("📁 Data Loading", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                uploaded_file = st.file_uploader(
                    "Upload CSV file", type=['csv'],
                    help="Upload a CSV file containing factor data",
                    key="factor_file_upload", on_change=self._on_file_upload_change
                )

            with col2:
                self._render_remote_data_loader()

            # 处理上传的文件或远程数据
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.dataframe = df
                    st.session_state.start_date = df['timestamp'].min()
                    st.session_state.end_date = df['timestamp'].max()
                    self.process_uploaded_data(df, uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.error(f"Factor analysis error: {e}\n{traceback.format_exc()}")
            elif st.session_state.remote_df is not None:
                st.session_state.dataframe = st.session_state.remote_df
                st.session_state.start_date = st.session_state.remote_df['timestamp'].min()
                st.session_state.end_date = st.session_state.remote_df['timestamp'].max()
                self.process_uploaded_data(st.session_state.remote_df, st.session_state.remote_dataset_name)

        # 显示分析文件列表和数据预览
        self._render_analysis_files_and_preview()

    def _handle_url_parameters(self):
        """处理URL参数"""
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

    def _on_file_upload_change(self):
        """文件上传变更处理"""
        has_current_id = hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id
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

    def _render_remote_data_loader(self):
        """渲染远程数据加载器"""
        from modules.db_utils import get_all_data_sources
        import paramiko
        import pandas as pd
        import io

        # 获取远程配置
        remote_configs = [c for c in get_all_data_sources() if c['type'] == 'Remote Server']
        if not remote_configs:
            st.warning("No remote server config found in Data Management module")
            return

        config_names = [f"{c['source_name']}@{c['ip']}:{c['port']}{c['path']}" for c in remote_configs]
        selected_idx = st.selectbox("Select Remote Config", range(len(config_names)),
                                  format_func=lambda i: config_names[i], key="remote_config_select")
        config = remote_configs[selected_idx]

        # 获取CSV文件列表
        all_csv_files = self._get_remote_csv_files(config)

        # 搜索和选择文件
        search_text = st.text_input('🔍 search csv files(support fuzzy matching)', '', key='remote_csv_search')
        filtered_files = [f for f in all_csv_files if search_text.lower() in f['name'].lower()]

        if filtered_files:
            file_options = [f"{f['name']} ({f['path']})" for f in filtered_files]
            if 'remote_csv_select' not in st.session_state or st.session_state['remote_csv_select'] >= len(file_options):
                st.session_state['remote_csv_select'] = 0

            selected_file_idx = st.selectbox("Select CSV File", range(len(file_options)),
                                          format_func=lambda i: file_options[i], key="remote_csv_select")
            selected_file = filtered_files[selected_file_idx]

            if st.button("📥 Load Selected CSV", key="load_selected_remote_csv"):
                self._load_remote_csv_file(config, selected_file)
        else:
            st.info("No matching CSV files.")

    def _get_remote_csv_files(self, config):
        """获取远程CSV文件列表"""
        import paramiko

        all_csv_files = []
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(config['ip'], port=int(config['port']),
                       username=config['username'], password=config['password'])
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
                                'path': file_path, 'name': file_attr.filename,
                                'size': file_attr.st_size, 'modified': file_attr.st_mtime
                            })
                except Exception as e:
                    st.warning(f"无法访问目录 {remote_path}: {e}")
                return csv_files

            all_csv_files = list_csv_files_recursive(config['path'])
            sftp.close()
            ssh.close()
        except Exception as e:
            st.error(f"连接远程服务器失败: {e}")

        return all_csv_files

    def _load_remote_csv_file(self, config, selected_file):
        """加载远程CSV文件"""
        import paramiko
        import pandas as pd
        import io

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(config['ip'], port=int(config['port']),
                       username=config['username'], password=config['password'])
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

    def _render_analysis_files_and_preview(self):
        """渲染分析文件列表和数据预览"""
        # 分析文件列表
        with st.expander("📂 Saved Analysis Files", expanded=False):
            try:
                from modules.factor_export import render_analysis_files_list
                render_analysis_files_list()
            except ImportError:
                st.warning("Factor export module not available")
            except Exception as e:
                st.error(f"❌ Error loading analysis files: {e}")

        # 数据预览
        with st.expander("📊 Data Preview", expanded=False):
            self._render_data_preview()

    def _render_data_preview(self):
        """渲染数据预览"""
        st.markdown("**Data Preview(head 10 rows):**")
        st.dataframe(st.session_state.dataframe.head(10), use_container_width=True)
        st.markdown("**Data Preview(tail 10 rows):**")
        st.dataframe(st.session_state.dataframe.tail(10), use_container_width=True)
        st.markdown("**Column Information:**")

        # 安全地获取示例值
        example_values = []
        for col in st.session_state.dataframe.columns:
            try:
                if len(st.session_state.dataframe) > 0:
                    val = st.session_state.dataframe[col].iloc[0]
                    example_values.append('N/A' if pd.isna(val) else str(val))
                else:
                    example_values.append('N/A')
            except Exception:
                example_values.append('N/A')

        col_info = pd.DataFrame({
            'ColumnName': st.session_state.dataframe.columns.tolist(),
            'DataType': [str(dtype) for dtype in st.session_state.dataframe.dtypes],
            'Non-Null Count': st.session_state.dataframe.count().tolist(),
            'Null Count': st.session_state.dataframe.isnull().sum().tolist(),
            'Example Value': example_values
        })
        st.dataframe(col_info, use_container_width=True)

    # ==================== 数据处理相关方法 ====================
    def process_uploaded_data(self, df, filename):
        """处理上传数据"""
        st.success(f"✅ File loaded successfully: {filename}")
        st.info(f"username: {st.session_state.username}")
        st.info(f"Data shape: {df.shape}")

        # 列配置
        self._render_column_configuration(df)

        # 保存选项配置
        self._render_save_options()

    def _render_column_configuration(self, df):
        """渲染列配置"""
        st.subheader("📋 Column Configuration")
        col1, col2, col3 = st.columns(3)

        all_columns = df.columns.tolist()
        price_col, signal_cols = DataProcessor.detect_columns(df)

        with col1:
            price_col = st.selectbox("Price Column", options=all_columns,
                                   index=all_columns.index(price_col) if price_col in all_columns else 0,
                                   help="Select the column containing price data", key="factor_price_col")

        with col2:
            signal_cols = st.multiselect("Factor Column", options=all_columns,
                                       default=signal_cols if signal_cols else [],
                                       help="Select the column containing factor data", key="factor_signal_cols")

        with col3:
            labeling_method = st.selectbox("Labeling Method", options=['point', 'triple'], index=0,
                                         help="Select the factor labeling method", key="factor_labeling_method")

        # 生成分析报告
        self._render_analysis_generation(df, price_col, signal_cols, labeling_method)

    def _render_save_options(self):
        """渲染保存选项"""
        st.subheader("💾 Save Options")
        col_save1, col_save2 = st.columns(2)

        with col_save1:
            has_current_id = hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id

            if has_current_id:
                overwrite_mode = st.checkbox(f"🔄 Overwrite previous analysis (ID: {st.session_state.current_analysis_id})",
                                           value=True, help="If checked, will overwrite the existing analysis. If unchecked, will create a new analysis with new ID.",
                                           key="overwrite_analysis")
            else:
                st.info("📝 Will create new analysis ID")
                overwrite_mode = False

            current_description = getattr(st.session_state, 'current_analysis_description', '')
            analysis_description = st.text_input("📝 Analysis Description", value=current_description,
                                                placeholder="Enter a description for this analysis...",
                                                help="Provide a meaningful description for this analysis",
                                                key="analysis_description_input")

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

    def _render_analysis_generation(self, df, price_col, signal_cols, labeling_method):
        """渲染分析生成部分"""
        # 数据验证
        validation_errors = self._validate_data_configuration(df, price_col, signal_cols)

        if validation_errors:
            st.error("**Data Validation Issues:**")
            for error in validation_errors:
                st.write(error)

        # 生成分析报告按钮
        if st.button("🚀 Generate Factor Analysis Report", type="primary", key="generate_factor_report"):
            if price_col and signal_cols:
                self._generate_factor_analysis_report(df, price_col, signal_cols, labeling_method)
            else:
                st.warning("⚠️ Please select Price Column and at least one Factor Column")

    def _validate_data_configuration(self, df, price_col, signal_cols):
        """验证数据配置"""
        validation_errors = []

        if not price_col:
            validation_errors.append("❌ Please select Price Column")
        if not signal_cols:
            validation_errors.append("❌ Please select at least one Factor Column")
        if price_col in signal_cols:
            validation_errors.append("⚠️ Price Column should not be selected as a Factor Column")
        if price_col and not pd.api.types.is_numeric_dtype(df[price_col]):
            validation_errors.append(f"❌ Price Column '{price_col}' is not a numeric type")

        for col in signal_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_errors.append(f"⚠️ Factor Column '{col}' is not a numeric type")

        return validation_errors

    def _generate_factor_analysis_report(self, df, price_col, signal_cols, labeling_method):
        """生成因子分析报告"""
        try:
            with st.spinner("Generating Factor Analysis Report..."):
                st.info("📊 Starting data processing...")

                # 数据预处理
                logger.info(f"Starting data formatting, original data shape: {df.shape}")
                validated_df = DataProcessor.validate_and_prepare_data(df.copy(), price_col, signal_cols)
                logger.info(f"Data validation completed, validated data shape: {validated_df.shape}")

                formatted_df = format_data(validated_df)
                logger.info(f"Data formatting completed, formatted data shape: {formatted_df.shape}")

                # 数据清理
                initial_len = len(formatted_df)
                st.session_state.formatted_df = formatted_df.dropna(subset=[price_col] + signal_cols)
                cleaned_len = len(st.session_state.formatted_df)
                dropped_rows = initial_len - cleaned_len

                if dropped_rows > 0:
                    st.warning(f"⚠️ {dropped_rows} rows containing null values were removed ({dropped_rows/initial_len:.1%})")

                if cleaned_len < 10:
                    st.error("❌ The cleaned data is too small to perform effective analysis")
                    return

                # 初始化SignalPerf
                st.info("🔧 Initializing SignalPerf...")
                self._initialize_signal_perf(price_col, signal_cols, labeling_method)

                # 保存数据
                st.info("💾 Saving analysis data...")
                self._save_analysis_data(validated_df, price_col, signal_cols, labeling_method)

                st.success("✅ Factor Analysis Report Generated Successfully!")
                st.info(f"📈 {len(signal_cols)} factors loaded: {', '.join(signal_cols)}")

                # 初始化并运行分析
                self._initialize_section_states()
                st.session_state._just_generated_report = True
                self.run_all_analysis()

        except Exception as e:
            st.error(f"❌ Error occurred while generating report: {str(e)}")
            with st.expander("🔍 Detailed Error Information", expanded=False):
                st.code(traceback.format_exc())
            logger.error(f"Factor analysis error: {e}\n{traceback.format_exc()}")

    def _initialize_signal_perf(self, price_col, signal_cols, labeling_method):
        """初始化SignalPerf"""
        st.session_state.factor_signal_list = signal_cols
        st.session_state.signal_cols = signal_cols
        st.session_state.factor_default_signal = signal_cols[0]
        st.session_state.price_col = price_col
        st.session_state.labeling_method = labeling_method

        st.session_state.factor_signal_perf = SignalPerf(
            mode='local',
            data=st.session_state.formatted_df,
            price_col=st.session_state.price_col,
            signal_cols=st.session_state.signal_cols,
            labeling_method=st.session_state.labeling_method
        )
        logger.info("SignalPerf instance created successfully")

    def _save_analysis_data(self, validated_df, price_col, signal_cols, labeling_method):
        """保存分析数据"""
        try:
            from modules.factor_export import auto_save_analysis_data
            overwrite_mode = st.session_state.get("overwrite_analysis", False)
            analysis_description = st.session_state.get("analysis_description_input", "")

            analysis_id = auto_save_analysis_data(
                validated_df, price_col, signal_cols, labeling_method,
                overwrite_mode=overwrite_mode, description=analysis_description or None
            )

            if analysis_id:
                st.session_state.current_analysis_id = analysis_id
                st.session_state.current_analysis_description = analysis_description or f"default_{analysis_id}"

                has_current_id = hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id
                if overwrite_mode and has_current_id:
                    st.success(f"✅ Analysis data updated for ID: `{analysis_id}`")
                else:
                    st.success(f"✅ Analysis data saved with new ID: `{analysis_id}`")
        except ImportError:
            st.warning("Factor export module not available")
        except Exception as e:
            st.error(f"❌ Failed to save analysis data: {e}")

    def _initialize_section_states(self):
        """初始化section states"""
        if 'factor_section_states' not in st.session_state:
            st.session_state.factor_section_states = {}

        section_keys = [section.state_key for section in self.sections if hasattr(section, 'state_key')]
        for key in section_keys:
            st.session_state.factor_section_states[key] = True

    # ==================== 分析仪表板相关方法 ====================
    def render_analysis_dashboard(self):
        """渲染分析仪表板"""
        sp = st.session_state.factor_signal_perf
        signal_list = st.session_state.factor_signal_list
        default_signal = st.session_state.factor_default_signal

        if sp is None:
            return

        if not signal_list:
            st.error("❌ Factor Column list is empty, please regenerate the analysis report")
            return

        # 显示成功提示
        if hasattr(st.session_state, '_just_generated_report') and st.session_state._just_generated_report:
            st.success("🎉 Factor analysis report generated! You can expand the analysis modules below to view the results.")
            st.session_state._just_generated_report = False

        # 渲染控制面板
        self._render_dashboard_controls(signal_list, default_signal)

        # 渲染分析控制
        self._render_analysis_controls()

        # 渲染所有分析模块
        st.markdown("---")
        try:
            for section in self.sections:
                section.render(sp, signal_list, default_signal, st.session_state.factor_export_mode)
        except Exception as e:
            st.error(f"❌ Error occurred while rendering analysis modules: {str(e)}")
            logger.error(f"Dashboard render error: {e}\n{traceback.format_exc()}")
            self._render_reset_option()

    def _render_dashboard_controls(self, signal_list, default_signal):
        """渲染仪表板控制"""
        st.markdown("### 📊 Factor Analysis Dashboard")
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

        with col1:
            selected_signal = st.selectbox("**Current Factor(default factor):**", options=signal_list,
                                         index=signal_list.index(default_signal) if default_signal in signal_list else 0,
                                         key="factor_selector")

            if selected_signal != default_signal:
                st.session_state.factor_default_signal = selected_signal
                st.rerun()
            st.markdown(f"**Number of Factors:** {len(signal_list)}")

        with col2:
            if st.button("📖 Expand All", key="factor_expand_all"):
                self._toggle_all_sections(True)

        with col3:
            if st.button("📕 Collapse All", key="factor_collapse_all"):
                self._toggle_all_sections(False)

        with col4:
            if st.button("📄 Export Mode" if not st.session_state.factor_export_mode else "🔧 Edit Mode",
                        key="toggle_factor_export_mode"):
                st.session_state.factor_export_mode = not st.session_state.factor_export_mode
                st.rerun()

        with col5:
            self._render_save_button()

        with col6:
            self._render_print_button()

        # Export mode indicator
        if st.session_state.factor_export_mode:
            st.info("📄 **Export Mode Enabled** - Displaying cached results only, parameter controls hidden", icon="📄")

    def _toggle_all_sections(self, expand: bool):
        """切换所有section的展开状态"""
        if 'factor_section_states' not in st.session_state:
            st.session_state.factor_section_states = {}

        section_keys = [section.state_key for section in self.sections if hasattr(section, 'state_key')]
        for key in section_keys:
            st.session_state.factor_section_states[key] = expand

        st.rerun()

    def _render_save_button(self):
        """渲染保存按钮"""
        has_current_id = hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id
        save_button_text = f"💾 Save/Share Analysis ({st.session_state.current_analysis_id})" if has_current_id else "💾 Save/Share Analysis Results"

        if st.button(save_button_text, key="save_factor_analysis"):
            self.save_analysis_results()

    def _render_print_button(self):
        """渲染打印按钮"""
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

    def _render_analysis_controls(self):
        """渲染分析控制"""
        st.markdown("### 🚀 Analysis Controls")
        col_run1, col_run2, col_run3, col_run4 = st.columns([1, 1, 1, 1])

        with col_run1:
            self._render_date_range_selector()

        with col_run2:
            if st.button("🚀 Run All Analysis", type="primary", key="run_all_analysis_dashboard"):
                self.run_all_analysis()

        with col_run3:
            if st.button("🧹 Clear All Results", key="clear_all_results_dashboard"):
                if 'factor_analysis_results' in st.session_state:
                    st.session_state.factor_analysis_results.clear()
                st.success("✅ All analysis results cleared")
                st.rerun()

        with col_run4:
            self._render_download_button()

    def _render_date_range_selector(self):
        """渲染日期范围选择器"""
        if st.session_state.start_date and st.session_state.end_date:
            def on_date_range_change():
                date_range = st.session_state["factor_date_range"]
                if len(date_range) == 2 and date_range[0] and date_range[1]:
                    st.session_state.start_date = date_range[0]
                    st.session_state.end_date = date_range[1]
                    start = pd.to_datetime(date_range[0])
                    end = pd.to_datetime(date_range[1])
                    temp_df = st.session_state.formatted_df[
                        (st.session_state.formatted_df['timestamp'] >= start) &
                        (st.session_state.formatted_df['timestamp'] <= end)
                    ]
                    st.session_state.factor_signal_perf = SignalPerf(
                        mode='local', data=temp_df,
                        price_col=st.session_state.price_col,
                        signal_cols=st.session_state.signal_cols,
                        labeling_method=st.session_state.labeling_method
                    )

            st.date_input("choose date range",
                         value=(st.session_state.start_date, st.session_state.end_date),
                         label_visibility="collapsed", key="factor_date_range",
                         format="YYYY-MM-DD", help="choose date range",
                         on_change=on_date_range_change)

    def _render_download_button(self):
        """渲染下载按钮"""
        query_params = st.query_params
        if 'id' in query_params:
            analysis_id = query_params['id']
            if analysis_id:
                try:
                    from modules.factor_export import downlaod_csv_by_id
                    csv_file = downlaod_csv_by_id(analysis_id)
                    if csv_file:
                        with open(csv_file, "rb") as f:
                            st.download_button("Download CSV", data=f.read(),
                                             file_name=csv_file.name, mime="text/csv")
                    else:
                        st.error("No available data to download")
                except ImportError:
                    st.warning("Factor export module not available")

    def _render_reset_option(self):
        """渲染重置选项"""
        if st.button("🔄 Reset Analysis State", key="reset_analysis_state"):
            st.session_state.factor_signal_perf = None
            st.session_state.factor_signal_list = []
            st.session_state.factor_default_signal = None
            st.session_state.factor_analysis_results = {}
            st.rerun()

    # ==================== 分析执行相关方法 ====================
    def run_all_analysis(self):
        """运行所有分析"""
        if st.session_state.factor_signal_perf is None:
            st.error("❌ No data loaded")
            return

        sp = st.session_state.factor_signal_perf
        signal_list = st.session_state.factor_signal_list
        default_signal = st.session_state.factor_default_signal

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

                try:
                    if isinstance(section, AutoAnalysisSection):
                        self._run_auto_analysis_section(section, sp, signal_list, default_signal)
                except Exception as e:
                    logger.error(f"Error in section {section.name}: {e}")
                    continue

            status_text.text("✅ All analyses completed!")
            progress_bar.progress(1.0)

            # 自动保存分析结果
            self._auto_save_results()

        except Exception as e:
            st.error(f"❌ Error during batch analysis: {e}")
            logger.error(f"Batch analysis error: {e}\n{traceback.format_exc()}")
        finally:
            progress_bar.empty()
            status_text.empty()

    def _run_auto_analysis_section(self, section: AutoAnalysisSection, sp, signal_list: List[str], default_signal: str):
        """运行自动分析模块"""
        try:
            func = getattr(sp, section.main_func_name)
            signature = ParameterParser.get_function_signature(func)
            saved_params = section.get_saved_parameters()

            # 构建参数：函数默认值 -> export_defaults -> 保存的参数
            form_data = {name: param_info['default'] for name, param_info in signature.items()}

            if section.export_defaults:
                form_data.update(section.export_defaults)
            if saved_params:
                form_data.update(saved_params)
                logger.info(f"Using saved parameters for {section.main_func_name}: {saved_params}")

            # 处理特殊参数
            if 'signal_name' in form_data and form_data['signal_name'] is None:
                form_data['signal_name'] = default_signal
            if 'signal_names' in form_data and form_data['signal_names'] is None:
                form_data['signal_names'] = signal_list

            # 执行分析并缓存结果
            result = section.execute_analysis(sp, section.main_func_name, form_data)
            cache_key = f"{section.main_func_name}_{default_signal}"

            if 'factor_analysis_results' not in st.session_state:
                st.session_state.factor_analysis_results = {}
            st.session_state.factor_analysis_results[cache_key] = result

        except Exception as e:
            logger.error(f"Error in auto analysis section {section.name}: {e}")
            raise

    def _auto_save_results(self):
        """自动保存分析结果"""
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

    # ==================== 辅助方法 ====================
    def load_analysis_from_url(self, analysis_id: str):
        """从URL参数加载分析"""
        try:
            from modules.factor_export import load_analysis_by_id
            if load_analysis_by_id(analysis_id):
                st.success(f"✅ Analysis loaded from URL: `{analysis_id}`")
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
