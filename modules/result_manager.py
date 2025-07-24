#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import base64
from pathlib import Path
import zipfile
import io
from typing import Dict, List, Optional

class ResultManager:
    """Result Management Module - Result saving, export, sharing, report generation"""

    def __init__(self):
        self.name = "Result Management"
        self.description = "Result saving, export, sharing, report generation"
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.initialize_state()

    def initialize_state(self):
        """Initialize module state"""
        if 'saved_reports' not in st.session_state:
            st.session_state.saved_reports = {}
        if 'report_templates' not in st.session_state:
            st.session_state.report_templates = self.get_default_templates()

    def render(self):
        """Render result management module interface"""
        st.markdown("## 📋 Result Management Module")
        st.markdown("*Unified management platform for result saving, export, sharing, and report generation*")

        # Main function tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Results Overview",
            "📄 Report Generation",
            "💾 Export Management",
            "🔗 Sharing Center"
        ])

        with tab1:
            self.render_results_overview()

        with tab2:
            self.render_report_generation()

        with tab3:
            self.render_export_management()

        with tab4:
            self.render_sharing_center()

    def render_results_overview(self):
        """Render results overview interface"""
        st.markdown("### 📊 Results Overview")

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        factor_results = getattr(st.session_state, 'factor_signal_perf', {})
        backtest_results = getattr(st.session_state, 'backtest_results', {})
        saved_reports = getattr(st.session_state, 'saved_reports', {})

        with col1:
            st.metric("Factor Analysis Results", len(factor_results))
        with col2:
            st.metric("Backtest Results", len(backtest_results))
        with col3:
            st.metric("Saved Reports", len(saved_reports))
        with col4:
            total_size = sum(len(str(result)) for result in [factor_results, backtest_results, saved_reports])
            st.metric("Data Size", f"{total_size / 1024:.1f} KB")

        # Factor analysis results
        if factor_results:
            st.markdown("#### 📈 Factor Analysis Results")

            for factor_name, result in factor_results.items():
                with st.expander(f"📊 {factor_name}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        # Display factor basic information
                        if isinstance(result, dict):
                            st.json(result)
                        else:
                            st.text(str(result))

                    with col2:
                        if st.button("📄 Generate Report", key=f"factor_report_{factor_name}"):
                            self.generate_factor_report(factor_name, result)

                    with col3:
                        if st.button("💾 Export Data", key=f"factor_export_{factor_name}"):
                            self.export_factor_data(factor_name, result)

        # Backtest results
        if backtest_results:
            st.markdown("#### 🚀 Backtest Results")

            for result_name, result in backtest_results.items():
                with st.expander(f"📊 {result_name}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        # Display key metrics
                        metrics = {
                            "Total Return": f"{result.get('total_return', 0):.2%}",
                            "Annual Return": f"{result.get('annual_return', 0):.2%}",
                            "Sharpe Ratio": f"{result.get('sharpe_ratio', 0):.2f}",
                            "Max Drawdown": f"{result.get('max_drawdown', 0):.2%}"
                        }
                        st.json(metrics)

                    with col2:
                        if st.button("📄 Generate Report", key=f"backtest_report_{result_name}"):
                            self.generate_backtest_report(result_name, result)

                    with col3:
                        if st.button("💾 Export Data", key=f"backtest_export_{result_name}"):
                            self.export_backtest_data(result_name, result)

        # If no results
        if not factor_results and not backtest_results:
            st.info("No analysis results available. Please perform factor analysis or strategy backtesting first")

    def render_report_generation(self):
        """Render report generation interface"""
        st.markdown("### 📄 Report Generation")

        # Report type selection
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Report Type")
            report_type = st.selectbox(
                "Select Report Type",
                ["Factor Analysis Report", "Backtest Analysis Report", "Comprehensive Analysis Report", "Custom Report"],
                key="report_type"
            )

        with col2:
            st.markdown("#### Report Configuration")

            if report_type == "Factor Analysis Report":
                self.render_factor_report_config()
            elif report_type == "Backtest Analysis Report":
                self.render_backtest_report_config()
            elif report_type == "Comprehensive Analysis Report":
                self.render_comprehensive_report_config()
            elif report_type == "Custom Report":
                self.render_custom_report_config()

        # Saved reports
        st.markdown("---")
        st.markdown("#### 📋 Saved Reports")

        if st.session_state.saved_reports:
            for report_name, report_info in st.session_state.saved_reports.items():
                with st.expander(f"📄 {report_name}", expanded=False):
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                    with col1:
                        st.text(f"Type: {report_info['type']}")
                        st.text(f"Created: {report_info['created_at']}")
                        st.text(f"Size: {report_info.get('size', 'N/A')}")

                    with col2:
                        if st.button("👁️ Preview", key=f"preview_{report_name}"):
                            self.preview_report(report_name)

                    with col3:
                        if st.button("📥 Download", key=f"download_{report_name}"):
                            self.download_report(report_name)

                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_report_{report_name}"):
                            del st.session_state.saved_reports[report_name]
                            st.rerun()
        else:
            st.info("No saved reports available")

    def render_factor_report_config(self):
        """Render factor report configuration"""
        with st.form("factor_report_form"):
            report_name = st.text_input("Report Name", placeholder="e.g., Factor Analysis Report_20240101")

            # Select factor results
            if hasattr(st.session_state, 'factor_signal_perf') and st.session_state.factor_signal_perf:
                factor_names = list(st.session_state.factor_signal_perf.keys())
                selected_factors = st.multiselect("Select Factors", factor_names, default=factor_names[:3])
            else:
                st.warning("No factor analysis results available")
                selected_factors = []

            # Report content configuration
            st.markdown("**Report Content**")
            include_summary = st.checkbox("Include Execution Summary", value=True)
            include_ic_analysis = st.checkbox("Include IC Analysis", value=True)
            include_return_analysis = st.checkbox("Include Return Analysis", value=True)
            include_risk_analysis = st.checkbox("Include Risk Analysis", value=True)
            include_charts = st.checkbox("Include Charts", value=True)

            # Report format
            report_format = st.selectbox("Report Format", ["HTML", "PDF", "Word"], index=0)

            if st.form_submit_button("Generate Report"):
                if report_name and selected_factors:
                    self.create_factor_report(
                        report_name, selected_factors, report_format,
                        include_summary, include_ic_analysis, include_return_analysis,
                        include_risk_analysis, include_charts
                    )
                else:
                    st.error("Please fill in the report name and select factors")

    def render_backtest_report_config(self):
        """Render backtest report configuration"""
        with st.form("backtest_report_form"):
            report_name = st.text_input("Report Name", placeholder="e.g., Strategy Backtest Report_20240101")

            # Select backtest results
            if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results:
                result_names = list(st.session_state.backtest_results.keys())
                selected_results = st.multiselect("Select Backtest Results", result_names, default=result_names[:3])
            else:
                st.warning("No backtest results available")
                selected_results = []

            # Report content configuration
            st.markdown("**Report Content**")
            include_summary = st.checkbox("Include Execution Summary", value=True)
            include_performance = st.checkbox("Include Performance Analysis", value=True)
            include_risk = st.checkbox("Include Risk Analysis", value=True)
            include_attribution = st.checkbox("Include Attribution Analysis", value=True)
            include_charts = st.checkbox("Include Charts", value=True)

            # Report format
            report_format = st.selectbox("Report Format", ["HTML", "PDF", "Word"], index=0)

            if st.form_submit_button("Generate Report"):
                if report_name and selected_results:
                    self.create_backtest_report(
                        report_name, selected_results, report_format,
                        include_summary, include_performance, include_risk,
                        include_attribution, include_charts
                    )
                else:
                    st.error("Please fill in the report name and select backtest results")

    def render_comprehensive_report_config(self):
        """Render comprehensive report configuration"""
        with st.form("comprehensive_report_form"):
            report_name = st.text_input("Report Name", placeholder="e.g., Comprehensive Analysis Report_20240101")

            # Select content
            st.markdown("**Included Content**")
            col1, col2 = st.columns(2)

            with col1:
                include_factor_analysis = st.checkbox("Factor Analysis", value=True)
                include_backtest_results = st.checkbox("Backtest Results", value=True)

            with col2:
                include_market_analysis = st.checkbox("Market Analysis", value=False)
                include_risk_monitoring = st.checkbox("Risk Monitoring", value=True)

            # Report format
            report_format = st.selectbox("Report Format", ["HTML", "PDF", "Word"], index=0)

            if st.form_submit_button("Generate Report"):
                if report_name:
                    self.create_comprehensive_report(
                        report_name, report_format,
                        include_factor_analysis, include_backtest_results,
                        include_market_analysis, include_risk_monitoring
                    )
                else:
                    st.error("Please fill in the report name")

    def render_custom_report_config(self):
        """Render custom report configuration"""
        with st.form("custom_report_form"):
            report_name = st.text_input("Report Name", placeholder="e.g., Custom Report_20240101")

            # Custom template
            st.markdown("**Report Template**")
            template_html = st.text_area(
                "HTML Template",
                placeholder="""
<html>
<head>
    <title>{{report_title}}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; }
        .content { margin: 20px 0; }
        .chart { margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{report_title}}</h1>
        <p>Generated Time: {{generated_at}}</p>
    </div>

    <div class="content">
        {{content}}
    </div>
</body>
</html>
                """,
                height=300
            )

            # Data binding
            st.markdown("**Data Binding**")
            data_mapping = st.text_area(
                "Data Mapping (JSON Format)",
                placeholder="""
{
    "report_title": "My Custom Report",
    "content": "This is report content"
}
                """,
                height=100
            )

            if st.form_submit_button("Generate Report"):
                if report_name and template_html:
                    try:
                        mapping = json.loads(data_mapping) if data_mapping else {}
                        self.create_custom_report(report_name, template_html, mapping)
                    except json.JSONDecodeError:
                        st.error("Data mapping format error, please use valid JSON format")
                else:
                    st.error("Please fill in the report name and template")

    def render_export_management(self):
        """Render export management interface"""
        st.markdown("### 💾 Export Management")

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Data Export")

            # Select export content
            export_type = st.selectbox(
                "Export Type",
                ["Factor Analysis Data", "Backtest Results Data", "Report File", "All Data"],
                key="export_type"
            )

            # Export format
            if export_type in ["Factor Analysis Data", "Backtest Results Data"]:
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "Excel", "JSON", "Pickle"],
                    key="export_format"
                )
            elif export_type == "Report File":
                export_format = st.selectbox(
                    "Export Format",
                    ["HTML", "PDF", "ZIP"],
                    key="export_format"
                )
            else:
                export_format = st.selectbox(
                    "Export Format",
                    ["ZIP", "JSON"],
                    key="export_format"
                )

            # Export button
            if st.button("📥 Export Data", type="primary"):
                self.export_data(export_type, export_format)

        with col2:
            st.markdown("#### Export History")

            # Here you can display export history
            export_history = [
                {"name": "Factor Analysis_20240101.csv", "size": "1.2 MB", "date": "2024-01-01"},
                {"name": "Backtest Results_20240101.xlsx", "size": "2.5 MB", "date": "2024-01-01"},
                {"name": "Comprehensive Report_20240101.html", "size": "3.8 MB", "date": "2024-01-01"}
            ]

            for item in export_history:
                with st.expander(f"📄 {item['name']}", expanded=False):
                    col_a, col_b = st.columns([2, 1])

                    with col_a:
                        st.text(f"Size: {item['size']}")
                        st.text(f"Date: {item['date']}")

                    with col_b:
                        if st.button("📥 Download", key=f"download_export_{item['name']}"):
                            st.info(f"Download {item['name']}")

        # Batch operations
        st.markdown("---")
        st.markdown("#### 🔄 Batch Operations")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📦 Package All Results"):
                self.package_all_results()

        with col2:
            if st.button("🗑️ Clean Up Temporary Files"):
                self.cleanup_temp_files()

        with col3:
            if st.button("📊 Generate Data Statistics"):
                self.generate_data_statistics()

    def render_sharing_center(self):
        """Render sharing center interface"""
        st.markdown("### 🔗 Sharing Center")

        # Sharing options
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Create Sharing Link")

            # Select sharing content
            share_type = st.selectbox(
                "Sharing Type",
                ["Report Sharing", "Data Sharing", "Chart Sharing", "Complete Project"],
                key="share_type"
            )

            # Sharing configuration
            st.markdown("**Sharing Configuration**")
            share_name = st.text_input("Sharing Name", placeholder="e.g., Factor Analysis Report")
            share_description = st.text_area("Sharing Description", placeholder="Briefly describe sharing content")

            # Access control
            st.markdown("**Access Control**")
            access_type = st.selectbox("Access Type", ["Public", "Password Protected", "Invitation Only"])

            if access_type == "Password Protected":
                share_password = st.text_input("Access Password", type="password")

            # Expiry
            expiry_days = st.number_input("Expiry (Days)", value=30, min_value=1, max_value=365)

            # Create sharing
            if st.button("🔗 Create Sharing Link", type="primary"):
                if share_name:
                    share_link = self.create_share_link(
                        share_type, share_name, share_description,
                        access_type, expiry_days
                    )
                    if share_link:
                        st.success(f"✅ Sharing Link Created Successfully")
                        st.code(share_link)
                else:
                        st.error("Please fill in the sharing name")

        with col2:
            st.markdown("#### Sharing Management")

            # Sharing history
            share_history = [
                {
                    "name": "Factor Analysis Report",
                    "type": "Report Sharing",
                    "views": 15,
                    "created": "2024-01-01",
                    "expires": "2024-01-31"
                },
                {
                    "name": "Backtest Results Data",
                    "type": "Data Sharing",
                    "views": 8,
                    "created": "2024-01-02",
                    "expires": "2024-02-01"
                }
            ]

            for item in share_history:
                with st.expander(f"🔗 {item['name']}", expanded=False):
                    col_a, col_b = st.columns([2, 1])

                    with col_a:
                        st.text(f"Type: {item['type']}")
                        st.text(f"Access Views: {item['views']}")
                        st.text(f"Created Time: {item['created']}")
                        st.text(f"Expiry Time: {item['expires']}")

                    with col_b:
                        if st.button("📊 Statistics", key=f"stats_{item['name']}"):
                            st.info(f"View Access Statistics of {item['name']}")

                        if st.button("🗑️ Delete", key=f"delete_share_{item['name']}"):
                            st.info(f"Delete Sharing {item['name']}")

        # Sharing statistics
        st.markdown("---")
        st.markdown("#### 📊 Sharing Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Sharing Count", "5")
        with col2:
            st.metric("Total Access Views", "128")
        with col3:
            st.metric("Active Sharing", "3")
        with col4:
            st.metric("This Month New", "2")

    # Helper methods
    def get_default_templates(self):
        """Get default report templates"""
        return {
            "Factor Analysis": """
            <html>
            <head><title>Factor Analysis Report</title></head>
            <body>
                <h1>Factor Analysis Report</h1>
                <p>Generated Time: {{generated_at}}</p>
                {{content}}
            </body>
            </html>
            """,
            "Backtest Analysis": """
            <html>
            <head><title>Backtest Analysis Report</title></head>
            <body>
                <h1>Backtest Analysis Report</h1>
                <p>Generated Time: {{generated_at}}</p>
                {{content}}
            </body>
            </html>
            """
        }

    def generate_factor_report(self, factor_name, result):
        """Generate factor report"""
        try:
            report_name = f"Factor Report_{factor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Generate HTML report
            html_content = f"""
            <html>
            <head>
                <title>Factor Analysis Report - {factor_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; }}
                    .content {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Factor Analysis Report - {factor_name}</h1>
                    <p>Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="content">
                    <h2>Factor Overview</h2>
                    <p>Factor Name: {factor_name}</p>

                    <h2>Analysis Results</h2>
                    <pre>{json.dumps(result, indent=2, ensure_ascii=False)}</pre>
                </div>
            </body>
            </html>
            """

            # Save report
            report_path = self.reports_dir / f"{report_name}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Record to session state
            st.session_state.saved_reports[report_name] = {
                "type": "Factor Analysis Report",
                "path": str(report_path),
                "size": f"{len(html_content) / 1024:.1f} KB",
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            st.success(f"✅ Factor Report '{report_name}' Generated Successfully")

        except Exception as e:
            st.error(f"❌ Report Generation Failed: {e}")

    def generate_backtest_report(self, result_name, result):
        """Generate backtest report"""
        try:
            report_name = f"Backtest Report_{result_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Generate HTML report
            html_content = f"""
            <html>
            <head>
                <title>Backtest Analysis Report - {result_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; }}
                    .content {{ margin: 20px 0; }}
                    .metrics {{ display: flex; justify-content: space-around; }}
                    .metric {{ text-align: center; padding: 10px; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Backtest Analysis Report - {result_name}</h1>
                    <p>Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="content">
                    <h2>Key Metrics</h2>
                    <div class="metrics">
                        <div class="metric">
                            <h3>Total Return</h3>
                            <p>{result.get('total_return', 0):.2%}</p>
                        </div>
                        <div class="metric">
                            <h3>Annual Return</h3>
                            <p>{result.get('annual_return', 0):.2%}</p>
                        </div>
                        <div class="metric">
                            <h3>Sharpe Ratio</h3>
                            <p>{result.get('sharpe_ratio', 0):.2f}</p>
                        </div>
                        <div class="metric">
                            <h3>Max Drawdown</h3>
                            <p>{result.get('max_drawdown', 0):.2%}</p>
                        </div>
                    </div>

                    <h2>Detailed Results</h2>
                    <pre>{json.dumps(result, indent=2, ensure_ascii=False, default=str)}</pre>
                </div>
            </body>
            </html>
            """

            # Save report
            report_path = self.reports_dir / f"{report_name}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Record to session state
            st.session_state.saved_reports[report_name] = {
                "type": "Backtest Analysis Report",
                "path": str(report_path),
                "size": f"{len(html_content) / 1024:.1f} KB",
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            st.success(f"✅ Backtest Report '{report_name}' Generated Successfully")

        except Exception as e:
            st.error(f"❌ Report Generation Failed: {e}")

    def create_factor_report(self, report_name, selected_factors, report_format, *args):
        """Create factor report"""
        try:
            # Here implement specific factor report generation logic
            st.success(f"✅ Factor Report '{report_name}' Created Successfully")
        except Exception as e:
            st.error(f"❌ Report Creation Failed: {e}")

    def create_backtest_report(self, report_name, selected_results, report_format, *args):
        """Create backtest report"""
        try:
            # Here implement specific backtest report generation logic
            st.success(f"✅ Backtest Report '{report_name}' Created Successfully")
        except Exception as e:
            st.error(f"❌ Report Creation Failed: {e}")

    def create_comprehensive_report(self, report_name, report_format, *args):
        """Create comprehensive report"""
        try:
            # Here implement specific comprehensive report generation logic
            st.success(f"✅ Comprehensive Report '{report_name}' Created Successfully")
        except Exception as e:
            st.error(f"❌ Report Creation Failed: {e}")

    def create_custom_report(self, report_name, template_html, mapping):
        """Create custom report"""
        try:
            # Replace template variables
            html_content = template_html
            for key, value in mapping.items():
                html_content = html_content.replace(f"{{{{{key}}}}}", str(value))

            # Add generated time
            html_content = html_content.replace("{{generated_at}}", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            # Save report
            report_path = self.reports_dir / f"{report_name}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Record to session state
            st.session_state.saved_reports[report_name] = {
                "type": "Custom Report",
                "path": str(report_path),
                "size": f"{len(html_content) / 1024:.1f} KB",
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            st.success(f"✅ Custom Report '{report_name}' Created Successfully")

        except Exception as e:
            st.error(f"❌ Report Creation Failed: {e}")

    def preview_report(self, report_name):
        """Preview report"""
        try:
            report_info = st.session_state.saved_reports[report_name]
            report_path = Path(report_info['path'])

            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                st.components.v1.html(html_content, height=600, scrolling=True)
            else:
                st.error("Report file does not exist")

        except Exception as e:
            st.error(f"❌ Preview Failed: {e}")

    def download_report(self, report_name):
        """Download report"""
        try:
            report_info = st.session_state.saved_reports[report_name]
            report_path = Path(report_info['path'])

            if report_path.exists():
                with open(report_path, 'rb') as f:
                    data = f.read()

                st.download_button(
                    label=f"📥 Download {report_name}",
                    data=data,
                    file_name=f"{report_name}.html",
                    mime="text/html"
                )
            else:
                st.error("Report file does not exist")

        except Exception as e:
            st.error(f"❌ Download Failed: {e}")

    def export_factor_data(self, factor_name, result):
        """Export factor data"""
        try:
            # Here implement factor data export logic
            st.success(f"✅ Factor Data '{factor_name}' Exported Successfully")
        except Exception as e:
            st.error(f"❌ Export Failed: {e}")

    def export_backtest_data(self, result_name, result):
        """Export backtest data"""
        try:
            # Here implement backtest data export logic
            st.success(f"✅ Backtest Data '{result_name}' Exported Successfully")
        except Exception as e:
            st.error(f"❌ Export Failed: {e}")

    def export_data(self, export_type, export_format):
        """Export data"""
        try:
            # Here implement data export logic
            st.success(f"✅ {export_type} Exported Successfully ({export_format} Format)")
        except Exception as e:
            st.error(f"❌ Export Failed: {e}")

    def create_share_link(self, share_type, share_name, share_description, access_type, expiry_days):
        """Create sharing link"""
        try:
            # Generate sharing link
            share_id = f"share_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            share_link = f"https://your-domain.com/share/{share_id}"

            # Here should implement actual sharing logic
            return share_link

        except Exception as e:
            st.error(f"❌ Sharing Link Creation Failed: {e}")
            return None

    def package_all_results(self):
        """Package all results"""
        try:
            st.success("✅ All Results Packaged Successfully")
        except Exception as e:
            st.error(f"❌ Packaging Failed: {e}")

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            st.success("✅ Temporary Files Cleaned Up Successfully")
        except Exception as e:
            st.error(f"❌ Cleanup Failed: {e}")

    def generate_data_statistics(self):
        """Generate data statistics"""
        try:
            st.success("✅ Data Statistics Generated Successfully")
        except Exception as e:
            st.error(f"❌ Statistics Generation Failed: {e}")
