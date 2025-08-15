#!/usr/bin/env python3

"""
EDA Analysis Module - Exploratory Data Analysis using ydata-profiling
EDA 分析模块 - 使用 ydata-profiling 进行探索性数据分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import warnings
import re
from bs4 import BeautifulSoup

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from ydata_profiling import ProfileReport
    from ydata_profiling.config import Config
except ImportError:
    st.error("❌ ydata-profiling is not installed. Please install it using: pip install ydata-profiling")
    st.stop()

logger = logging.getLogger(__name__)

class EDAAnalysisModule:
    """EDA Analysis Module using ydata-profiling"""

    def __init__(self):
        """Initialize EDA module"""
        self.module_name = "EDA Analysis"
        self.version = "1.0.0"

        # Initialize session state for EDA
        if 'eda_data' not in st.session_state:
            st.session_state.eda_data = {}
        if 'eda_reports' not in st.session_state:
            st.session_state.eda_reports = {}
        if 'eda_config' not in st.session_state:
            st.session_state.eda_config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default ydata-profiling configuration"""
        return {
            "title": "Data Profiling Report",
            "minimal": False,
            "explorative": True,
            "sensitive": True,
            "correlations": {
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": False},
                "phi_k": {"calculate": True},
                "cramers": {"calculate": True}
            },
            "missing_diagrams": {
                "matrix": True,
                "bar": True,
                "heatmap": True,
                "dendrogram": True
            },
            "interactions": {
                "continuous": True,
                "targets": []
            },
            "samples": {
                "head": 5,
                "tail": 5
            }
        }

    def _extract_report_content(self, html_content: str) -> str:
        """Extract core content from ydata-profiling HTML report, removing outer framework"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # 查找主要内容区域
            main_content = None

            # 尝试不同的选择器来找到核心内容
            selectors = [
                'div.container-fluid',  # ydata-profiling常用的容器
                'div.container',
                'main',
                'div[role="main"]',
                'body > div',
                'div.tab-content',
                'div.profile-report'
            ]

            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    main_content = content
                    break

            if not main_content:
                # 如果找不到特定容器，尝试获取body内容
                main_content = soup.find('body')
                if main_content:
                    # 移除script和style标签
                    for tag in main_content(['script', 'style']):
                        tag.decompose()

            if main_content:
                # 提取所有CSS样式
                styles = []
                for style_tag in soup.find_all('style'):
                    styles.append(style_tag.get_text())

                # 获取外部CSS链接
                css_links = []
                for link in soup.find_all('link', rel='stylesheet'):
                    if link.get('href'):
                        css_links.append(f'<link rel="stylesheet" href="{link.get("href")}">')

                # 构建清洁的HTML
                clean_html = f"""
                <style>
                /* 基础样式重置和优化 */
                .streamlit-container {{
                    max-width: 100% !important;
                    padding: 0 !important;
                }}

                /* ydata-profiling 样式优化 */
                .container-fluid, .container {{
                    max-width: 100% !important;
                    padding: 10px !important;
                    margin: 0 !important;
                }}

                /* 卡片样式优化 */
                .card {{
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}

                .card-header {{
                    background-color: #f8f9fa;
                    border-bottom: 1px solid #dee2e6;
                    padding: 10px 15px;
                    font-weight: 600;
                }}

                .card-body {{
                    padding: 15px;
                }}

                /* 表格样式优化 */
                table {{
                    width: 100% !important;
                    margin-bottom: 1rem;
                    background-color: transparent;
                    font-size: 0.9rem;
                }}

                .table {{
                    border-collapse: collapse;
                }}

                .table td, .table th {{
                    padding: 8px;
                    vertical-align: top;
                    border-top: 1px solid #dee2e6;
                }}

                /* 导航标签优化 */
                .nav-tabs {{
                    border-bottom: 2px solid #dee2e6;
                    margin-bottom: 20px;
                }}

                .nav-tabs .nav-link {{
                    border: none;
                    border-bottom: 2px solid transparent;
                    color: #495057;
                    padding: 10px 15px;
                }}

                .nav-tabs .nav-link.active {{
                    color: #007bff;
                    border-bottom-color: #007bff;
                    background: none;
                }}

                /* 统计指标样式 */
                .stats {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 20px;
                }}

                .stat-item {{
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 6px;
                    padding: 10px;
                    min-width: 120px;
                    text-align: center;
                }}

                /* 图表容器优化 */
                .plot-container {{
                    margin: 15px 0;
                    text-align: center;
                }}

                /* 隐藏可能的外层导航 */
                .navbar, .header, .top-nav {{
                    display: none !important;
                }}

                /* 响应式设计 */
                @media (max-width: 768px) {{
                    .container-fluid, .container {{
                        padding: 5px !important;
                    }}

                    .card-body {{
                        padding: 10px;
                    }}

                    table {{
                        font-size: 0.8rem;
                    }}
                }}

                {chr(10).join(styles)}
                </style>

                <div class="ydata-profiling-content">
                    {main_content}
                </div>
                """

                return clean_html
            else:
                # 如果无法提取内容，返回优化的原始HTML
                return self._optimize_original_html(html_content)

        except Exception as e:
            logger.warning(f"Error extracting report content: {e}")
            return self._optimize_original_html(html_content)

    def _optimize_original_html(self, html_content: str) -> str:
        """优化原始HTML，移除不必要的元素"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # 移除可能的导航栏、头部等
            unwanted_selectors = [
                'nav.navbar',
                '.header',
                '.top-navigation',
                '.sidebar',
                'header',
                '.nav-header'
            ]

            for selector in unwanted_selectors:
                for element in soup.select(selector):
                    element.decompose()

            # 添加优化样式
            style_tag = soup.new_tag('style')
            style_tag.string = """
            body {
                margin: 0;
                padding: 10px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }
            .container-fluid, .container {
                max-width: 100% !important;
                padding: 0 !important;
            }
            .navbar, .header, .top-nav {
                display: none !important;
            }
            """

            if soup.head:
                soup.head.append(style_tag)

            return str(soup)

        except Exception as e:
            logger.warning(f"Error optimizing HTML: {e}")
            return html_content

    def render(self):
        """Main render method for EDA module"""
        st.markdown("## 📊 Exploratory Data Analysis (EDA)")
        st.markdown("**Powered by ydata-profiling** - Automated data profiling and analysis")

        # Create tabs for different EDA functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📂 Data Upload",
            "⚙️ Configuration",
            "🔍 Generate Report",
            "📊 View Reports",
            "💾 Export Results"
        ])

        with tab1:
            self._render_data_upload()

        with tab2:
            self._render_configuration()

        with tab3:
            self._render_generate_report()

        with tab4:
            self._render_view_reports()

        with tab5:
            self._render_export_results()

    def _render_data_upload(self):
        """Render data upload interface"""
        st.markdown("### 📂 Data Upload and Selection")

        # Multiple data source options
        data_source = st.radio(
            "Select Data Source:",
            ["Upload File", "Use Existing Data", "Sample Datasets"],
            horizontal=True
        )

        if data_source == "Upload File":
            self._render_file_upload()
        elif data_source == "Use Existing Data":
            self._render_existing_data()
        else:
            self._render_sample_datasets()

    def _render_file_upload(self):
        """Render file upload interface"""
        st.markdown("#### 📁 Upload Data File")

        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv', 'xlsx', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )

        if uploaded_file is not None:
            try:
                # Determine file type and load data
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    # CSV upload options
                    col1, col2 = st.columns(2)
                    with col1:
                        separator = st.selectbox("Separator", [',', ';', '\t', '|'], index=0)
                        encoding = st.selectbox("Encoding", ['utf-8', 'gbk', 'gb2312', 'latin-1'], index=0)
                    with col2:
                        header_row = st.number_input("Header Row", min_value=0, value=0)
                        index_col = st.selectbox("Index Column", [None, 0, 1, 2, 3], index=0)

                    df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding, header=header_row, index_col=index_col)

                elif file_extension in ['xlsx', 'xls']:
                    # Excel upload options
                    col1, col2 = st.columns(2)
                    with col1:
                        sheet_name = st.text_input("Sheet Name", value=0, help="Sheet name or index (0 for first sheet)")
                        try:
                            sheet_name = int(sheet_name)
                        except:
                            pass
                    with col2:
                        header_row = st.number_input("Header Row", min_value=0, value=0)

                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)

                elif file_extension == 'json':
                    df = pd.read_json(uploaded_file)

                elif file_extension == 'parquet':
                    df = pd.read_parquet(uploaded_file)

                else:
                    st.error("Unsupported file format")
                    return

                # Store data
                dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])

                if st.button("💾 Load Dataset"):
                    st.session_state.eda_data[dataset_name] = df
                    st.success(f"✅ Dataset '{dataset_name}' loaded successfully!")
                    st.info(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

                # Preview data
                if not df.empty:
                    st.markdown("#### 👁️ Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)

                    # Basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

    def _render_existing_data(self):
        """Render interface to use existing data from session state"""
        st.markdown("#### 🗃️ Use Existing Data")

        # Check for existing data in session state
        available_data = {}

        # Check user_data from other modules
        if 'user_data' in st.session_state and st.session_state.user_data:
            available_data.update(st.session_state.user_data)

        # Check eda_data
        if st.session_state.eda_data:
            available_data.update(st.session_state.eda_data)

        if available_data:
            selected_dataset = st.selectbox(
                "Select Dataset:",
                list(available_data.keys())
            )

            if selected_dataset:
                df = available_data[selected_dataset]
                st.success(f"✅ Selected dataset: {selected_dataset}")
                st.info(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

                # Preview
                st.markdown("#### 👁️ Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
        else:
            st.warning("⚠️ No existing data found. Please upload data first or use other modules to load data.")

    def _render_sample_datasets(self):
        """Render sample datasets for testing"""
        st.markdown("#### 🎯 Sample Datasets")

        sample_options = {
            "Iris Dataset": "Classic iris flower dataset",
            "Stock Data": "Sample stock price data",
            "Sales Data": "Sample sales transaction data",
            "Customer Data": "Sample customer demographics data"
        }

        selected_sample = st.selectbox(
            "Choose Sample Dataset:",
            list(sample_options.keys())
        )

        st.info(f"📝 {sample_options[selected_sample]}")

        if st.button("📥 Load Sample Dataset"):
            df = self._generate_sample_data(selected_sample)
            st.session_state.eda_data[selected_sample] = df
            st.success(f"✅ Sample dataset '{selected_sample}' loaded!")
            st.dataframe(df.head(), use_container_width=True)

    def _generate_sample_data(self, sample_type: str) -> pd.DataFrame:
        """Generate sample data for testing"""
        np.random.seed(42)

        if sample_type == "Iris Dataset":
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['species'] = iris.target_names[iris.target]
            return df

        elif sample_type == "Stock Data":
            dates = pd.date_range('2023-01-01', periods=252, freq='D')
            price = 100
            prices = []
            volumes = []

            for _ in range(252):
                change = np.random.normal(0, 0.02)
                price *= (1 + change)
                prices.append(price)
                volumes.append(np.random.randint(1000, 10000))

            return pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': [p + np.random.normal(0, 1) for p in prices],
                'volume': volumes
            })

        elif sample_type == "Sales Data":
            return pd.DataFrame({
                'product_id': np.random.randint(1, 101, 1000),
                'customer_id': np.random.randint(1001, 2001, 1000),
                'sales_amount': np.random.exponential(100, 1000),
                'quantity': np.random.randint(1, 10, 1000),
                'discount': np.random.uniform(0, 0.3, 1000),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
                'date': pd.date_range('2023-01-01', periods=1000, freq='H')
            })

        else:  # Customer Data
            return pd.DataFrame({
                'customer_id': range(1, 501),
                'age': np.random.randint(18, 80, 500),
                'income': np.random.normal(50000, 15000, 500),
                'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500),
                'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 500),
                'num_children': np.random.poisson(1.5, 500),
                'city_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], 500),
                'spending_score': np.random.randint(1, 101, 500)
            })

    def _render_configuration(self):
        """Render profiling configuration interface"""
        st.markdown("### ⚙️ Profiling Configuration")
        st.markdown("Customize the ydata-profiling report settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📋 Basic Settings")

            config = st.session_state.eda_config.copy()

            config["title"] = st.text_input(
                "Report Title",
                value=config["title"]
            )

            config["minimal"] = st.checkbox(
                "Minimal Report",
                value=config["minimal"],
                help="Generate a minimal report with essential statistics only"
            )

            config["explorative"] = st.checkbox(
                "Explorative Mode",
                value=config["explorative"],
                help="Include advanced statistical analysis"
            )

            config["sensitive"] = st.checkbox(
                "Sensitive Data Mode",
                value=config["sensitive"],
                help="Enable sensitive data detection and handling"
            )

        with col2:
            st.markdown("#### 📊 Analysis Options")

            # Correlation settings
            st.markdown("**Correlation Analysis:**")
            config["correlations"]["pearson"]["calculate"] = st.checkbox(
                "Pearson Correlation",
                value=config["correlations"]["pearson"]["calculate"]
            )

            config["correlations"]["spearman"]["calculate"] = st.checkbox(
                "Spearman Correlation",
                value=config["correlations"]["spearman"]["calculate"]
            )

            config["correlations"]["kendall"]["calculate"] = st.checkbox(
                "Kendall Correlation",
                value=config["correlations"]["kendall"]["calculate"]
            )

            config["correlations"]["phi_k"]["calculate"] = st.checkbox(
                "Phi-K Correlation",
                value=config["correlations"]["phi_k"]["calculate"]
            )

            # Missing data diagrams
            st.markdown("**Missing Data Diagrams:**")
            config["missing_diagrams"]["matrix"] = st.checkbox(
                "Missing Data Matrix",
                value=config["missing_diagrams"]["matrix"]
            )

            config["missing_diagrams"]["bar"] = st.checkbox(
                "Missing Data Bar Chart",
                value=config["missing_diagrams"]["bar"]
            )

            config["missing_diagrams"]["heatmap"] = st.checkbox(
                "Missing Data Heatmap",
                value=config["missing_diagrams"]["heatmap"]
            )

        # Sample settings
        st.markdown("#### 🔍 Sample Display Settings")
        col1, col2 = st.columns(2)
        with col1:
            config["samples"]["head"] = st.number_input(
                "Number of head samples",
                min_value=0, max_value=20,
                value=config["samples"]["head"]
            )
        with col2:
            config["samples"]["tail"] = st.number_input(
                "Number of tail samples",
                min_value=0, max_value=20,
                value=config["samples"]["tail"]
            )

        # Save configuration
        if st.button("💾 Save Configuration"):
            st.session_state.eda_config = config
            st.success("✅ Configuration saved successfully!")

        # Reset to defaults
        if st.button("🔄 Reset to Defaults"):
            st.session_state.eda_config = self._get_default_config()
            st.success("✅ Configuration reset to defaults!")
            st.rerun()

    def _render_generate_report(self):
        """Render report generation interface"""
        st.markdown("### 🔍 Generate Profiling Report")

        # Select dataset
        if not st.session_state.eda_data:
            st.warning("⚠️ No datasets available. Please upload data first.")
            return

        selected_dataset = st.selectbox(
            "Select Dataset for Analysis:",
            list(st.session_state.eda_data.keys())
        )

        if selected_dataset:
            df = st.session_state.eda_data[selected_dataset]

            # Display dataset info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # Advanced options
            with st.expander("🔧 Advanced Options"):
                # Column selection
                st.markdown("**Column Selection:**")
                all_columns = df.columns.tolist()

                use_all_columns = st.checkbox("Use All Columns", value=True)

                if not use_all_columns:
                    selected_columns = st.multiselect(
                        "Select Columns to Include:",
                        all_columns,
                        default=all_columns
                    )
                else:
                    selected_columns = all_columns

                # Data sampling for large datasets
                if df.shape[0] > 10000:
                    st.markdown("**Data Sampling (for large datasets):**")
                    use_sampling = st.checkbox("Enable Data Sampling", value=True)

                    if use_sampling:
                        sample_size = st.slider(
                            "Sample Size",
                            min_value=1000,
                            max_value=min(50000, df.shape[0]),
                            value=min(10000, df.shape[0])
                        )
                        sample_method = st.selectbox(
                            "Sampling Method",
                            ["Random", "First N rows", "Last N rows"]
                        )
                    else:
                        sample_size = df.shape[0]
                        sample_method = "No sampling"
                else:
                    use_sampling = False
                    sample_size = df.shape[0]
                    sample_method = "No sampling"

            # Generate report button
            if st.button("🚀 Generate Profiling Report", type="primary"):
                # Prepare data
                analysis_df = df[selected_columns].copy()

                # Apply sampling if needed
                if use_sampling and sample_size < df.shape[0]:
                    if sample_method == "Random":
                        analysis_df = analysis_df.sample(n=sample_size, random_state=42)
                    elif sample_method == "First N rows":
                        analysis_df = analysis_df.head(sample_size)
                    else:  # Last N rows
                        analysis_df = analysis_df.tail(sample_size)

                # Generate report
                with st.spinner("🔄 Generating profiling report... This may take a few minutes for large datasets."):
                    try:
                        # Create ProfileReport with configuration
                        config = st.session_state.eda_config

                        # Convert config to ydata-profiling format
                        profile_config = {
                            "title": config["title"],
                            "minimal": config["minimal"],
                            "explorative": config["explorative"],
                            "sensitive": config["sensitive"],
                            "correlations": config["correlations"],
                            "missing_diagrams": config["missing_diagrams"],
                            "samples": config["samples"]
                        }

                        profile = ProfileReport(
                            analysis_df,
                            **profile_config
                        )

                        # Generate report
                        report_html = profile.to_html()

                        # Extract and clean the report content
                        clean_html = self._extract_report_content(report_html)

                        # Store report
                        report_key = f"{selected_dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.eda_reports[report_key] = {
                            'dataset': selected_dataset,
                            'html': report_html,  # 保留原始HTML
                            'clean_html': clean_html,  # 保存清洁版本
                            'config': config,
                            'timestamp': datetime.now(),
                            'shape': analysis_df.shape,
                            'columns': selected_columns,
                            'sampling_info': {
                                'used': use_sampling,
                                'method': sample_method,
                                'size': sample_size
                            }
                        }

                        st.success(f"✅ Profiling report generated successfully! Report ID: {report_key}")

                        # Show basic statistics
                        st.markdown("#### 📊 Quick Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Analyzed Rows", analysis_df.shape[0])
                        with col2:
                            st.metric("Analyzed Columns", analysis_df.shape[1])
                        with col3:
                            st.metric("Numeric Columns", analysis_df.select_dtypes(include=[np.number]).shape[1])
                        with col4:
                            st.metric("Categorical Columns", analysis_df.select_dtypes(include=['object', 'category']).shape[1])

                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        logger.error(f"Error in EDA report generation: {e}")

    def _render_view_reports(self):
        """Render interface to view generated reports"""
        st.markdown("### 📊 View Generated Reports")

        if not st.session_state.eda_reports:
            st.info("📝 No reports generated yet. Go to 'Generate Report' tab to create your first report.")
            return

        # Report selection
        report_keys = list(st.session_state.eda_reports.keys())

        # Create a more user-friendly display
        report_options = {}
        for key in report_keys:
            report = st.session_state.eda_reports[key]
            display_name = f"{report['dataset']} - {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            report_options[display_name] = key

        selected_display = st.selectbox(
            "Select Report to View:",
            list(report_options.keys())
        )

        if selected_display:
            selected_key = report_options[selected_display]
            report = st.session_state.eda_reports[selected_key]

            # Display report metadata
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📋 Report Information")
                st.markdown(f"**Dataset:** {report['dataset']}")
                st.markdown(f"**Generated:** {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Data Shape:** {report['shape'][0]} rows × {report['shape'][1]} columns")

            with col2:
                st.markdown("#### ⚙️ Configuration Used")
                st.markdown(f"**Title:** {report['config']['title']}")
                st.markdown(f"**Minimal:** {report['config']['minimal']}")
                st.markdown(f"**Explorative:** {report['config']['explorative']}")
                st.markdown(f"**Sensitive:** {report['config']['sensitive']}")

            # Sampling information
            if report['sampling_info']['used']:
                st.markdown("#### 🎯 Sampling Information")
                st.info(f"📊 Sampling used: {report['sampling_info']['method']} - {report['sampling_info']['size']} rows")

            # Display report
            st.markdown("#### 📊 Profiling Report")
            st.markdown("---")

            # 显示选项
            display_mode = st.radio(
                "Report Display Mode:",
                ["Streamlit Optimized (Recommended)", "Original Full Report"],
                index=1,
                help="Choose how to display the report"
            )

            if display_mode == "Streamlit Optimized (Recommended)":
                # 使用清洁版本的HTML
                clean_html = report.get('clean_html', report['html'])
                st.components.v1.html(clean_html, height=1000, scrolling=True)
            else:
                # 使用原始HTML
                st.components.v1.html(report['html'], height=800, scrolling=True)

    def _render_export_results(self):
        """Render export interface"""
        st.markdown("### 💾 Export Results")

        if not st.session_state.eda_reports:
            st.info("📝 No reports available for export. Generate reports first.")
            return

        # Report selection for export
        report_keys = list(st.session_state.eda_reports.keys())
        report_options = {}
        for key in report_keys:
            report = st.session_state.eda_reports[key]
            display_name = f"{report['dataset']} - {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            report_options[display_name] = key

        selected_display = st.selectbox(
            "Select Report to Export:",
            list(report_options.keys())
        )

        if selected_display:
            selected_key = report_options[selected_display]
            report = st.session_state.eda_reports[selected_key]

            # Export options
            st.markdown("#### 📤 Export Options")

            col1, col2 = st.columns(2)

            with col1:
                # Export as HTML
                if st.button("📄 Download HTML Report (Original)"):
                    html_content = report['html']

                    # Create download link
                    b64 = base64.b64encode(html_content.encode()).decode()
                    filename = f"eda_report_{report['dataset']}_{report['timestamp'].strftime('%Y%m%d_%H%M%S')}.html"
                    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Click to download original HTML report</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Export clean HTML
                if st.button("📄 Download Optimized HTML Report"):
                    clean_html = report.get('clean_html', report['html'])

                    b64 = base64.b64encode(clean_html.encode()).decode()
                    filename = f"eda_report_optimized_{report['dataset']}_{report['timestamp'].strftime('%Y%m%d_%H%M%S')}.html"
                    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Click to download optimized HTML report</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Export configuration as JSON
                if st.button("⚙️ Download Configuration"):
                    config_json = json.dumps(report['config'], indent=2)
                    b64 = base64.b64encode(config_json.encode()).decode()
                    filename = f"eda_config_{report['dataset']}_{report['timestamp'].strftime('%Y%m%d_%H%M%S')}.json"
                    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Click to download configuration</a>'
                    st.markdown(href, unsafe_allow_html=True)

            with col2:
                # Export dataset summary
                if st.button("📊 Download Dataset Summary"):
                    # Create summary
                    summary = {
                        "dataset_name": report['dataset'],
                        "analysis_timestamp": report['timestamp'].isoformat(),
                        "data_shape": report['shape'],
                        "columns_analyzed": report['columns'],
                        "sampling_info": report['sampling_info'],
                        "configuration": report['config']
                    }

                    summary_json = json.dumps(summary, indent=2, default=str)
                    b64 = base64.b64encode(summary_json.encode()).decode()
                    filename = f"eda_summary_{report['dataset']}_{report['timestamp'].strftime('%Y%m%d_%H%M%S')}.json"
                    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Click to download summary</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Clear reports
                if st.button("🗑️ Clear All Reports", type="secondary"):
                    if st.button("⚠️ Confirm Clear All", type="secondary"):
                        st.session_state.eda_reports = {}
                        st.success("✅ All reports cleared!")
                        st.rerun()

            # Display report metadata
            st.markdown("#### 📋 Report Metadata")
            metadata_df = pd.DataFrame([{
                "Property": "Dataset Name",
                "Value": report['dataset']
            }, {
                "Property": "Generation Time",
                "Value": report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            }, {
                "Property": "Data Shape",
                "Value": f"{report['shape'][0]} rows × {report['shape'][1]} columns"
            }, {
                "Property": "Columns Analyzed",
                "Value": len(report['columns'])
            }, {
                "Property": "Sampling Used",
                "Value": "Yes" if report['sampling_info']['used'] else "No"
            }])

            st.dataframe(metadata_df, use_container_width=True)


# Create global instance
def create_eda_module():
    """Factory function to create EDA module instance"""
    return EDAAnalysisModule()

if __name__ == "__main__":
    # Test the module
    module = EDAAnalysisModule()
    module.render()
