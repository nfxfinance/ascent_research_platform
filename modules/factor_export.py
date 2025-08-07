#!/usr/bin/env python3

import streamlit as st
import json
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
import html

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

def generate_analysis_id() -> str:
    """Generate unique ID for analysis session"""
    return str(uuid.uuid4())[:8]

def auto_save_analysis_data(data_df, price_col: str, signal_cols: list, labeling_method: str, overwrite_mode: bool = False, description: str = None):
    """
    Auto save analysis data when generating report

    Args:
        data_df: Original DataFrame
        price_col: Price column name
        signal_cols: List of signal column names
        labeling_method: Labeling method
        overwrite_mode: Whether to overwrite existing analysis
        description: Analysis description

    Returns:
        str: Analysis ID
    """
    # 决定使用的ID
    if overwrite_mode and hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id:
        analysis_id = st.session_state.current_analysis_id
        st.info(f"🔄 Overwriting existing analysis: {analysis_id}")
    else:
        analysis_id = generate_analysis_id()
        st.info(f"📝 Creating new analysis: {analysis_id}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create save directory
    save_dir = Path("saved_analyses")
    save_dir.mkdir(exist_ok=True)

    # 如果没有提供描述，生成默认描述
    if not description:
        # 计算现有文件数量以生成默认描述
        existing_files = list(save_dir.glob("analysis_*.json"))
        default_num = len(existing_files) + 1
        description = f"default_{default_num}"

    # Prepare basic data for saving
    save_data = {
        'analysis_id': analysis_id,
        'timestamp': timestamp,
        'module': 'Factor Analysis',
        'description': description,
        'signal_list': signal_cols,
        'default_signal': signal_cols[0] if signal_cols else None,
        'parameters': {
            'labeling_method': labeling_method,
            'price_col': price_col,
            'signal_cols': signal_cols,
        },
        'status': 'data_saved',  # 标记只保存了数据，还没有分析结果
        'username': st.session_state.get('username', ''),  # 新增用户名字段
    }
    if st.session_state.get('username', '') == '':
        raise Exception("Username is not set")

    # Save files
    save_file = save_dir / f"analysis_{analysis_id}.json"
    csv_file = save_dir / f"data_{analysis_id}.csv"

    try:
        # Save JSON with metadata
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)

        # Save CSV data
        data_df.to_csv(csv_file, index=False)

        return analysis_id

    except Exception as e:
        st.error(f"❌ Failed to save analysis data: {e}")
        return None

def update_analysis_results(analysis_id: str, analysis_results: dict = None, analysis_params: dict = None, section_states: dict = None):
    """
    Update analysis results for existing analysis ID

    Args:
        analysis_id: Analysis ID to update
        analysis_results: Analysis results to save
        analysis_params: Analysis parameters to save
        section_states: Section states to save
    """
    save_dir = Path("saved_analyses")
    save_file = save_dir / f"analysis_{analysis_id}.json"

    if not save_file.exists():
        st.error(f"❌ Analysis {analysis_id} not found")
        return False

    try:
        # Load existing data
        with open(save_file, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        # Update with new results
        save_data['last_updated'] = datetime.now().strftime("%Y%m%d_%H%M%S")

        if analysis_results is not None:
            save_data['analysis_results'] = analysis_results

        if analysis_params is not None:
            save_data['analysis_params'] = analysis_params

        if section_states is not None:
            save_data['section_states'] = section_states

        save_data['status'] = 'complete'  # 标记分析完成

        # Save updated data
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)

        return True

    except Exception as e:
        st.error(f"❌ Failed to update analysis results: {e}")
        return False

def get_saved_parameters_for_section(section_key: str) -> dict:
    """Get saved parameters for a specific section"""
    if 'factor_analysis_params' not in st.session_state:
        return {}

    # Convert section key to function name
    # For example: "_signal_diagnostics_analysis" -> "plot_signal_diagnostics_plotly"
    section_to_func_map = {
        '_signal_diagnostics_analysis': 'plot_signal_diagnostics_plotly',
        '_ic_decay_analysis': 'plot_ic_decay_summary',
        '_rolling_ic_analysis': 'plot_rolling_ic',
        '_rolling_ic_statistics': 'calc_rolling_ic_stats',
        '_ic_distribution_analysis': 'plot_ic_distribution',
        '_ic_cumulative_analysis': 'plot_ic_cumulative_ir',
        '_ic_autocorrelation_analysis': 'plot_ic_pacf',
        '_person_&_spearman_correlation': 'plot_correlation_matrix',
        '_ic_surface_robust': 'plot_ic_surface_robust',
        '_combined_diagnostics': 'plot_combined_diagnostics',
        '_mean_return_by_quantile': 'plot_mean_return_by_quantile',
        '_rolling_fdm': 'plot_rolling_fdm',
        '_turnover': 'calc_turnover',
        '_holding_period': 'calculate_signal_holding_period_by_sign',
        '_roc_&_precision-recall_curves': 'plot_roc_curve',  # Special case, handled separately
        '_negative_log_loss_curve': 'plot_negative_log_loss_plotly',
    }

    func_name = section_to_func_map.get(section_key)
    if func_name and func_name in st.session_state.factor_analysis_params:
        return st.session_state.factor_analysis_params[func_name]

    return {}

def save_analysis_with_id():
    """Save analysis results with unique ID and create shareable link"""
    if st.session_state.factor_signal_perf is None:
        st.warning("⚠️ No analysis results to save")
        return None

    # 使用现有的ID或生成新ID
    if hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id:
        analysis_id = st.session_state.current_analysis_id
        is_update = True
        st.info(f"🔄 Updating existing analysis: {analysis_id}")
    else:
        analysis_id = generate_analysis_id()
        is_update = False
        st.info(f"📝 Creating new analysis: {analysis_id}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create save directory
    save_dir = Path("saved_analyses")
    save_dir.mkdir(exist_ok=True)

    # Get the original DataFrame from SignalPerf instance
    sp = st.session_state.factor_signal_perf
    original_data = None

    import pandas as pd
    original_data = pd.merge(sp.prices, sp.signals, on='timestamp')

    if original_data is None:
        st.error("❌ Cannot access original data from SignalPerf instance")
        return None

    # 获取描述（从session state或使用默认值）
    description = getattr(st.session_state, 'current_analysis_description', None)
    if not description:
        # 计算现有文件数量以生成默认描述
        existing_files = list(save_dir.glob("analysis_*.json"))
        default_num = len(existing_files) + 1
        description = f"default_{default_num}"

    # Prepare data for saving
    save_data = {
        'analysis_id': analysis_id,
        'timestamp': timestamp,
        'last_updated': timestamp,
        'module': 'Factor Analysis',
        'description': description,
        'signal_list': st.session_state.factor_signal_list,
        'default_signal': st.session_state.factor_default_signal,
        'section_states': st.session_state.factor_section_states.copy(),
        'analysis_params': st.session_state.factor_analysis_params.copy(),
        'parameters': {
            'labeling_method': getattr(sp, 'labeling_method', 'point'),
            'price_col': getattr(sp, 'price_col', 'close'),
            'signal_cols': st.session_state.factor_signal_list,
        },
        'status': 'complete',
        'username': st.session_state.username
    }

    # Save to JSON file
    save_file = save_dir / f"analysis_{analysis_id}.json"

    # Also save CSV file separately for easier access
    csv_file = save_dir / f"data_{analysis_id}.csv"

    try:
        # Save JSON with metadata and results
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)

        # Save CSV data separately
        original_data.to_csv(csv_file, index=False)

        # Update session state
        st.session_state.current_analysis_id = analysis_id
        st.session_state.current_analysis_description = description

        # Add to saved reports
        if 'saved_reports' not in st.session_state:
            st.session_state.saved_reports = []

        # Update existing report or add new one
        existing_report = None
        for i, report in enumerate(st.session_state.saved_reports):
            if report.get('analysis_id') == analysis_id:
                existing_report = i
                break

        if existing_report is not None:
            st.session_state.saved_reports[existing_report] = save_data
        else:
            st.session_state.saved_reports.append(save_data)

        # Generate shareable link
        current_url = st.get_option("server.baseUrlPath") or ""
        # Use the current protocol and host
        if hasattr(st, 'get_option') and st.get_option("server.port"):
            port = st.get_option("server.port")
            share_url = f"http://h.adpolitan.com:{port}?page=factor&id={analysis_id}"
        else:
            share_url = f"{current_url}?page=factor&id={analysis_id}"

        if is_update:
            st.success(f"✅ Analysis updated for ID: `{analysis_id}`")
        else:
            st.success(f"✅ Analysis saved with ID: `{analysis_id}`")

        st.info(f"📁 Files: analysis_{analysis_id}.json, data_{analysis_id}.csv")
        st.markdown(f"🔗 **Share Link:** [{share_url}]({share_url})")

        # 使用原生Streamlit组件展示可选择的链接
        st.markdown("**📋 Copy Link Below:**")
        st.code(share_url, language=None)
        return analysis_id

    except Exception as e:
        st.error(f"❌ Failed to save analysis: {e}")
        return None

def downlaod_csv_by_id(analysis_id: str):
    """Download CSV file by ID"""
    print(f"Downloading CSV file by ID: {analysis_id}")
    save_dir = Path("saved_analyses")
    csv_file = save_dir / f"data_{analysis_id}.csv"
    if csv_file.exists():
        return csv_file
    return None

def load_analysis_by_id(analysis_id: str):
    """Load analysis results by ID and recreate SignalPerf instance"""
    save_dir = Path("saved_analyses")
    analysis_file = save_dir / f"analysis_{analysis_id}.json"
    csv_file = save_dir / f"data_{analysis_id}.csv"

    if not analysis_file.exists():
        st.error(f"❌ Analysis with ID '{analysis_id}' not found")
        return False

    try:
        # Load metadata and results
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load CSV data
        if csv_file.exists():
            # Load from separate CSV file (preferred)
            import pandas as pd
            original_df = pd.read_csv(csv_file)
            st.session_state.dataframe = original_df
            st.session_state.start_date = original_df['timestamp'].min()
            st.session_state.end_date = original_df['timestamp'].max()
        else:
            st.error("❌ Original data not found in saved analysis")
            return False

        # Recreate SignalPerf instance
        try:
            # Import required modules
            import sys
            import os

            # Add AscentQuantMaster to path if needed
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ascentquant_python_path = os.path.join(project_root, 'AscentQuantMaster', 'python')
            if os.path.exists(ascentquant_python_path) and ascentquant_python_path not in sys.path:
                sys.path.insert(0, ascentquant_python_path)

            from lib.lib_signal_perf import SignalPerf
            from lib.utils import format_data

            # # Add IC extensions to SignalPerf (needed for analysis methods)
            # try:
            #     from lib.lib_signal_perf_ic_extensions import add_ic_extensions
            #     add_ic_extensions(SignalPerf)
            # except ImportError:
            #     st.warning("⚠️ IC extensions not available - some analysis methods may not work")

            st.session_state.formatted_df = format_data(original_df)

            # Extract parameters
            params = data['parameters']
            st.session_state.price_col = params['price_col']
            st.session_state.signal_cols = params['signal_cols']
            st.session_state.labeling_method = params.get('labeling_method', 'point')

            # Create new SignalPerf instance
            sp = SignalPerf(
                mode='local',
                data=st.session_state.formatted_df,
                price_col=st.session_state.price_col,
                signal_cols=st.session_state.signal_cols,
                labeling_method=st.session_state.labeling_method
            )

            # Restore session state
            st.session_state.factor_signal_perf = sp
            st.session_state.factor_signal_list = data['signal_list']
            st.session_state.factor_default_signal = data['default_signal']
            st.session_state.factor_section_states = data.get('section_states', {})
            # st.session_state.factor_analysis_results = data['analysis_results']
            st.session_state.factor_analysis_params = data['analysis_params']

            # CRITICAL: Ensure export mode is disabled after loading
            # This allows the "Run Analysis" buttons to show up
            st.session_state.factor_export_mode = False

            # Clear any auto-run flags to prevent conflicts
            if hasattr(st.session_state, '_auto_run_analysis'):
                st.session_state._auto_run_analysis = False
            if hasattr(st.session_state, '_just_generated_report'):
                st.session_state._just_generated_report = False

            # st.success(f"✅ Analysis '{analysis_id}' loaded successfully with original data")
            # st.info(f"📊 Recreated SignalPerf with {len(signal_cols)} factors: {', '.join(signal_cols)}")
            # st.info("🔧 **Edit mode enabled** - You can now run new analyses or modify parameters")
            return True

        except Exception as e:
            st.error(f"❌ Failed to recreate SignalPerf instance: {e}")
            # Still load what we can (results only)
            st.session_state.factor_signal_list = data['signal_list']
            st.session_state.factor_default_signal = data['default_signal']
            st.session_state.factor_section_states = data.get('section_states', {})
            # st.session_state.factor_analysis_results = data['analysis_results']
            st.session_state.factor_analysis_params = data['analysis_params']
            st.session_state.factor_signal_perf = None

            # Ensure export mode is disabled even in fallback case
            st.session_state.factor_export_mode = False

            st.warning("⚠️ Loaded analysis results only. You'll need to reload the original data to generate new charts.")

            return True

    except Exception as e:
        st.error(f"❌ Failed to load analysis: {e}")
        return False

def export_to_html() -> str:
    """Export analysis results to HTML"""
    if st.session_state.factor_signal_perf is None:
        return None

    # Prepare HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Factor Analysis Report</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Factor Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Default Signal: {st.session_state.factor_default_signal}</p>
            <p>Total Factors: {len(st.session_state.factor_signal_list)}</p>
        </div>
    """

    # Add analysis results
    for result_key, result_data in st.session_state.factor_analysis_results.items():
        html_content += f"""
        <div class="section">
            <h2>{result_key}</h2>
        """

        if isinstance(result_data, dict):
            for key, value in result_data.items():
                if isinstance(value, (int, float)):
                    html_content += f'<div class="metric"><strong>{key}:</strong> {value:.4f}</div>'
                elif isinstance(value, str):
                    html_content += f'<div class="metric"><strong>{key}:</strong> {html.escape(value)}</div>'

        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    return html_content

def render_export_section():
    """Render export functionality section"""
    st.markdown("### 📄 Export & Share")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💾 Save with ID", help="Save analysis with unique ID for sharing"):
            analysis_id = save_analysis_with_id()
            if analysis_id:
                st.session_state.last_analysis_id = analysis_id

    with col2:
        if st.button("📄 Export HTML", help="Export analysis as HTML document"):
            with st.spinner("Generating export..."):
                export_data = export_to_html()
                if export_data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"factor_analysis_{timestamp}.html"

                    st.download_button(
                        label="📥 Download Report",
                        data=export_data.encode('utf-8'),
                        file_name=filename,
                        mime="text/html",
                        key="download_factor_report"
                    )

    with col3:
        if st.button("📊 Export Data", help="Export raw analysis data as JSON"):
            if st.session_state.factor_analysis_results:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'signal_list': st.session_state.factor_signal_list,
                    'default_signal': st.session_state.factor_default_signal,
                    'results': st.session_state.factor_analysis_results
                }

                json_data = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                st.download_button(
                    label="📥 Download JSON",
                    data=json_data.encode('utf-8'),
                    file_name=f"factor_data_{timestamp}.json",
                    mime="application/json",
                    key="download_factor_data"
                )
            else:
                st.warning("⚠️ No analysis data to export")

    with col4:
        # Load analysis by ID
        load_id = st.text_input("🔗 Load by ID", placeholder="Enter analysis ID", key="load_analysis_id")
        if st.button("🔄 Load", help="Load saved analysis by ID"):
            if load_id:
                if load_analysis_by_id(load_id):
                    st.rerun()
            else:
                st.warning("⚠️ Please enter an analysis ID")

    # Show last saved ID if available
    if hasattr(st.session_state, 'last_analysis_id'):
        st.info(f"💡 Last saved ID: `{st.session_state.last_analysis_id}`")

def list_all_analysis_files():
    """List all analysis files with their descriptions and metadata"""
    save_dir = Path("saved_analyses")
    if not save_dir.exists():
        return []

    analysis_files = []

    for json_file in save_dir.glob("analysis_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查是否有描述字段，如果没有则添加默认描述
            if 'description' not in data:
                # 从文件名提取ID并生成默认描述
                analysis_id = data.get('analysis_id', json_file.stem.replace('analysis_', ''))
                data['description'] = f"default_{analysis_id}"

                # 更新文件以包含描述
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            analysis_files.append({
                'analysis_id': data.get('analysis_id', ''),
                'description': data.get('description', 'No description'),
                'timestamp': data.get('timestamp', ''),
                'last_updated': data.get('last_updated', data.get('timestamp', '')),
                'signal_list': data.get('signal_list', []),
                'status': data.get('status', 'unknown'),
                'file_path': str(json_file),
                'username': data.get('username', 'unkown')
            })

        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

    # 按时间戳排序（最新的在前）
    analysis_files.sort(key=lambda x: x.get('last_updated', x.get('timestamp', '')), reverse=True)

    return analysis_files

def update_analysis_description(analysis_id: str, new_description: str):
    """Update the description of an analysis file"""
    save_dir = Path("saved_analyses")
    save_file = save_dir / f"analysis_{analysis_id}.json"

    if not save_file.exists():
        return False

    try:
        # Load existing data
        with open(save_file, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        # Update description
        save_data['description'] = new_description
        save_data['last_updated'] = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save updated data
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)

        # Update session state if this is the current analysis
        if hasattr(st.session_state, 'current_analysis_id') and st.session_state.current_analysis_id == analysis_id:
            st.session_state.current_analysis_description = new_description

        return True

    except Exception as e:
        print(f"Error updating description for {analysis_id}: {e}")
        return False

def clean_cache_by_id(analysis_id: str):
    """Clean cache by ID"""
    save_dir = Path("saved_analyses")
    save_file = save_dir / f"analysis_{analysis_id}.json"
    csv_file = save_dir / f"data_{analysis_id}.csv"
    if save_file.exists():
        save_file.unlink()
    if csv_file.exists():
        csv_file.unlink()

def render_analysis_files_list():
    """Render a list of all analysis files with descriptions"""
    st.markdown("### 📂 Saved Analysis Files")

    table_data = []
    analysis_files = list_all_analysis_files()

    for file_info in analysis_files:
        # 格式化时间戳
        timestamp = file_info['timestamp']
        if len(timestamp) == 15:  # YYYYMMDD_HHMMSS
            formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
        else:
            formatted_time = timestamp

        # 格式化信号列表
        signals = ', '.join(file_info['signal_list'][:3])  # 显示前3个信号
        if len(file_info['signal_list']) > 3:
            signals += f" (+{len(file_info['signal_list'])-3} more)"

        table_data.append({
            'ID': file_info['analysis_id'],
            'Description': file_info['description'],
            'Signals': signals,
            'Created': formatted_time,
            'Username': file_info['username'],
        })

    # 显示表格前添加搜索框
    search_text = st.text_input('🔍 search analysis files (support ID, description, signals, username)', '',
    key='analysis_file_search')

    df = pd.DataFrame(table_data)
    if search_text:
        mask = df.apply(lambda row: search_text.lower() in str(row['ID']).lower()
                                   or search_text.lower() in str(row['Description']).lower()
                                   or search_text.lower() in str(row['Signals']).lower()
                                   or search_text.lower() in str(row['Username']).lower(), axis=1)
        df = df[mask]

    if df.empty:
        st.info("📄 No saved analysis files found")
        return

    # 构建 AgGrid 配置
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_default_column(editable=False, groupable=True)
    gb.configure_selection('multiple', use_checkbox=True)

    grid_response = AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        theme='streamlit',
        height=400
    )

    selected = grid_response['selected_rows']
    st.info(selected)
    if selected is not None and len(selected) > 0:
        selected_ids = selected['ID']
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Load", key="aggrid_load_btn"):
                if len(selected_ids) == 1:
                    if load_analysis_by_id(selected_ids[0]):
                        st.session_state.current_analysis_id = selected_ids[0]
                        # 更新页面url path
                        # st.update_page_config(url=f"/?page=factor&id={selected_ids[0]}")
                        # Streamlit 没有 update_page_config 方法，这里可以考虑用 st.experimental_set_query_params 代替
                        st.query_params["page"] = "factor"
                        st.query_params["id"] = selected_ids[0]
                        st.success(f"✅ Analysis {selected_ids[0]} loaded successfully")
                        st.rerun()
                else:
                    st.error("Please select only one analysis to load")
        with col2:
            if st.button("🔄 Modify Description", key="aggrid_modify_description_btn"):
                if len(selected_ids) == 1:
                    current_description = df.loc[df['ID'] == selected_ids[0], 'Description'].values[0]
                    # 使用 session_state 来管理修改描述的状态
                    if f"modify_desc_mode_{selected_ids[0]}" not in st.session_state:
                        st.session_state[f"modify_desc_mode_{selected_ids[0]}"] = False
                    st.session_state[f"modify_desc_mode_{selected_ids[0]}"] = True
                else:
                    st.error("Please select only one analysis to modify description")
            if len(selected_ids) == 1 and st.session_state.get(f"modify_desc_mode_{selected_ids[0]}", False):
                current_description = df.loc[df['ID'] == selected_ids[0], 'Description'].values[0]
                if f"new_desc_value_{selected_ids[0]}" not in st.session_state:
                    st.session_state[f"new_desc_value_{selected_ids[0]}"] = current_description
                new_description = st.text_input("Enter new description",
                                               value=st.session_state[f"new_desc_value_{selected_ids[0]}"],
                                               key=f"modify_desc_{selected_ids[0]}",
                                               on_change=lambda: st.session_state.update({f"new_desc_value_{selected_ids[0]}": st.session_state[f"modify_desc_{selected_ids[0]}"]}))

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("💾 Save Update", key="aggrid_modify_description_save_btn"):
                        st.info(f"Updating description for {selected_ids[0]} to {new_description}")
                        update_analysis_description(selected_ids[0], new_description)
                        st.success(f"✅ Description updated successfully")
                        st.session_state[f"modify_desc_mode_{selected_ids[0]}"] = False
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel", key="aggrid_modify_description_cancel_btn"):
                        st.session_state[f"modify_desc_mode_{selected_ids[0]}"] = False
                        st.rerun()
        with col3:
            if st.button("🗑️ Delete", key="aggrid_delete_btn"):
                for selected_id in selected_ids:
                    clean_cache_by_id(selected_id)
                    st.success(f"✅ Analysis {selected_id} deleted successfully")
                st.rerun()
    else:
        st.info("Please select a record in the table")
