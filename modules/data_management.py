#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import sqlite3
import pickle
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px

# 在文件顶部导入db_utils
from modules.db_utils import (
    init_data_source_table,
    insert_or_update_data_source,
    get_all_data_sources,
    delete_data_source
)

logger = logging.getLogger(__name__)

class DataManagementModule:
    """Data Management Module - Data source integration, cleaning, storage, updates"""

    def __init__(self):
        self.name = "Data Management"
        self.description = "Data source integration, cleaning, storage, updates"
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.initialize_state()

    def initialize_state(self):
        """Initialize module state"""
        if 'data_sources' not in st.session_state:
            st.session_state.data_sources = {}
        if 'data_cleaning_rules' not in st.session_state:
            st.session_state.data_cleaning_rules = {}
        if 'data_update_schedule' not in st.session_state:
            st.session_state.data_update_schedule = {}

    def render(self):
        """Render data management module interface"""
        st.markdown("## 📊 Data Management Module")
        st.markdown("*Unified management platform for data source integration, cleaning, storage, and updates*")

        # Main function tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Data Sources",
            "🧹 Data Cleaning",
            "💾 Data Storage",
            "🔄 Data Updates",
            "📊 Data Overview"
        ])

        with tab1:
            self.render_data_sources()

        with tab2:
            self.render_data_cleaning()

        with tab3:
            self.render_data_storage()

        with tab4:
            self.render_data_updates()

        with tab5:
            self.render_data_overview()

    def render_data_sources(self):
        """Render data source management interface"""
        st.markdown("### 📁 Data Source Management")

        # Data source type selection
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Data Source Type")
            data_source_type = st.selectbox(
                "Select Data Source Type",
                ["Local Files", "Database", "API Interface", "Web Scraper", "Third-party Services", "Remote Server"],
                key="data_source_type"
            )

        with col2:
            st.markdown("#### Data Source Configuration")

            if data_source_type == "Local Files":
                self.render_local_file_source()
            elif data_source_type == "Database":
                self.render_database_source()
            elif data_source_type == "API Interface":
                self.render_api_source()
            elif data_source_type == "Web Scraper":
                self.render_web_scraper_source()
            elif data_source_type == "Third-party Services":
                self.render_third_party_source()
            elif data_source_type == "Remote Server":
                self.render_remote_server_source()

        # Configured data sources
        st.markdown("---")
        st.markdown("#### Configured Data Sources")

        # 用数据库查询所有数据源
        all_db_sources = get_all_data_sources()
        # 合并 session_state 和 db 的数据源（以 session_state 为主，防止未刷新时丢失）
        sources = {**{d["source_name"]: d for d in all_db_sources}, **st.session_state.data_sources}

        if sources:
            for source_name, source_config in sources.items():
                with st.expander(f"📊 {source_name} ({source_config['type']})", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.json(source_config)

                    with col2:
                        if st.button("🔄 Test Connection", key=f"test_{source_name}"):
                            self.test_data_source(source_name)

                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{source_name}"):
                            delete_data_source(source_name)
                            if source_name in st.session_state.data_sources:
                                del st.session_state.data_sources[source_name]
                            st.rerun()
        else:
            st.info("No configured data sources")

    def render_local_file_source(self):
        """Render local file data source configuration"""
        with st.form("local_file_form"):
            source_name = st.text_input("Data Source Name", placeholder="e.g.: Stock Historical Data")
            file_path = st.text_input("File Path", placeholder="e.g.: /path/to/data.csv")
            file_type = st.selectbox("File Type", ["CSV", "Excel", "JSON", "Parquet"])

            # File upload
            uploaded_file = st.file_uploader(
                "Or Upload File",
                type=['csv', 'xlsx', 'json', 'parquet'],
                key="local_file_upload"
            )

            encoding = st.selectbox("Encoding", ["utf-8", "gbk", "gb2312"])

            if st.form_submit_button("Add Data Source"):
                if source_name:
                    config = {
                        "type": "Local Files",
                        "file_path": file_path,
                        "file_type": file_type,
                        "encoding": encoding,
                        "uploaded_file": uploaded_file.name if uploaded_file else None,
                        "created_at": datetime.now().isoformat()
                    }

                    # If there's an uploaded file, save to local
                    if uploaded_file:
                        file_save_path = self.data_dir / uploaded_file.name
                        with open(file_save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        config["file_path"] = str(file_save_path)

                    st.session_state.data_sources[source_name] = config
                    st.success(f"✅ Data source '{source_name}' added successfully")
                    st.rerun()

    def render_database_source(self):
        """Render database data source configuration"""
        with st.form("database_form"):
            source_name = st.text_input("Data Source Name", placeholder="e.g.: Production Database")
            db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL", "SQLite", "Oracle", "SQL Server"])

            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host Address", placeholder="localhost")
                port = st.number_input("Port", value=3306 if db_type == "MySQL" else 5432)
                database = st.text_input("Database Name", placeholder="database_name")

            with col2:
                username = st.text_input("Username", placeholder="username")
                password = st.text_input("Password", type="password")
                table_name = st.text_input("Table Name", placeholder="table_name")

            if st.form_submit_button("Add Data Source"):
                if source_name and host and database:
                    config = {
                        "type": "Database",
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "database": database,
                        "username": username,
                        "password": "***",  # Don't save plain text password
                        "table_name": table_name,
                        "created_at": datetime.now().isoformat()
                    }

                    st.session_state.data_sources[source_name] = config
                    st.success(f"✅ Data source '{source_name}' added successfully")
                    st.rerun()

    def render_api_source(self):
        """Render API interface data source configuration"""
        with st.form("api_form"):
            source_name = st.text_input("Data Source Name", placeholder="e.g.: Financial API")
            api_url = st.text_input("API Address", placeholder="https://api.example.com/data")
            api_method = st.selectbox("Request Method", ["GET", "POST", "PUT", "DELETE"])

            # API parameters
            st.markdown("**API Parameters**")
            params = st.text_area("Parameters (JSON format)", placeholder='{"symbol": "AAPL", "period": "1d"}')

            # Authentication information
            st.markdown("**Authentication Information**")
            auth_type = st.selectbox("Authentication Type", ["None", "API Key", "Bearer Token", "Basic Auth"])

            if auth_type == "API Key":
                api_key = st.text_input("API Key", type="password")
            elif auth_type == "Bearer Token":
                token = st.text_input("Token", type="password")
            elif auth_type == "Basic Auth":
                col1, col2 = st.columns(2)
                with col1:
                    auth_username = st.text_input("Username")
                with col2:
                    auth_password = st.text_input("Password", type="password")

            update_frequency = st.selectbox("Update Frequency", ["Manual", "Hourly", "Daily", "Weekly", "Monthly"])

            if st.form_submit_button("Add Data Source"):
                if source_name and api_url:
                    config = {
                        "type": "API Interface",
                        "api_url": api_url,
                        "method": api_method,
                        "params": params,
                        "auth_type": auth_type,
                        "update_frequency": update_frequency,
                        "created_at": datetime.now().isoformat()
                    }

                    st.session_state.data_sources[source_name] = config
                    st.success(f"✅ Data source '{source_name}' added successfully")
                    st.rerun()

    def render_web_scraper_source(self):
        """Render web scraper data source configuration"""
        with st.form("scraper_form"):
            source_name = st.text_input("Data Source Name", placeholder="e.g.: Financial Website Data")
            target_url = st.text_input("Target URL", placeholder="https://example.com/data")

            # Scraper configuration
            st.markdown("**Scraper Configuration**")
            col1, col2 = st.columns(2)
            with col1:
                selector_type = st.selectbox("Selector Type", ["CSS Selector", "XPath", "Regular Expression"])
                selector = st.text_input("Selector", placeholder="e.g.: .data-table")

            with col2:
                delay = st.number_input("Request Interval (seconds)", value=1.0, min_value=0.1)
                max_pages = st.number_input("Maximum Pages", value=10, min_value=1)

            # Request header settings
            user_agent = st.text_input(
                "User-Agent",
                value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )

            update_frequency = st.selectbox("Update Frequency", ["Manual", "Hourly", "Daily", "Weekly", "Monthly"])

            if st.form_submit_button("Add Data Source"):
                if source_name and target_url:
                    config = {
                        "type": "Web Scraper",
                        "target_url": target_url,
                        "selector_type": selector_type,
                        "selector": selector,
                        "delay": delay,
                        "max_pages": max_pages,
                        "user_agent": user_agent,
                        "update_frequency": update_frequency,
                        "created_at": datetime.now().isoformat()
                    }

                    st.session_state.data_sources[source_name] = config
                    st.success(f"✅ Data source '{source_name}' added successfully")
                    st.rerun()

    def render_third_party_source(self):
        """Render third-party service data source configuration"""
        with st.form("third_party_form"):
            source_name = st.text_input("Data Source Name", placeholder="e.g.: Wind Data")
            service_type = st.selectbox(
                "Service Type",
                ["Wind", "Bloomberg", "Tushare", "AKShare", "Yahoo Finance", "Alpha Vantage", "Other"]
            )

            # Service configuration
            st.markdown("**Service Configuration**")
            if service_type in ["Wind", "Bloomberg"]:
                st.info("Need to install corresponding client software and API permissions")
                api_key = st.text_input("API Key", type="password")
            elif service_type == "Tushare":
                token = st.text_input("Tushare Token", type="password")
            elif service_type == "Alpha Vantage":
                api_key = st.text_input("API Key", type="password")
            else:
                custom_config = st.text_area("Custom Configuration (JSON format)", placeholder='{"key": "value"}')

            data_types = st.multiselect(
                "Data Types",
                ["Stock Quotes", "Futures Quotes", "Fund Data", "Macro Data", "Financial Data", "News Data", "Other"]
            )

            update_frequency = st.selectbox("Update Frequency", ["Manual", "Hourly", "Daily", "Weekly", "Monthly"])

            if st.form_submit_button("Add Data Source"):
                if source_name and service_type:
                    config = {
                        "type": "Third-party Service",
                        "service_type": service_type,
                        "data_types": data_types,
                        "update_frequency": update_frequency,
                        "created_at": datetime.now().isoformat()
                    }

                    st.session_state.data_sources[source_name] = config
                    st.success(f"✅ Data source '{source_name}' added successfully")
                    st.rerun()
    def render_remote_server_source(self):
        """远程服务器数据源配置与CSV文件浏览"""
        st.markdown("##### Remote Server Configuration")
        source_name = st.text_input("Data Source Name", placeholder="e.g.: Wind Data")
        ip = st.text_input("Server IP", key="remote_ip")
        port = st.number_input("Port", value=22, min_value=1, max_value=65535, key="remote_port")
        path = st.text_input("Directory Path", value="/", key="remote_path")
        username = st.text_input("Username", key="remote_user")
        password = st.text_input("Password", type="password", key="remote_pwd")

        # 初始化表结构
        init_data_source_table()

        if st.button("🔍 List CSV Files", key="list_remote_csv"):
            if not (ip and path and username and password and port):
                st.error("请填写所有字段。")
            else:
                import paramiko
                try:
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(ip, port=int(port), username=username, password=password)
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
                    csv_files = list_csv_files_recursive(path)
                    if csv_files:
                        st.success(f"找到 {len(csv_files)} 个CSV文件:")
                    else:
                        st.info("在目录中未找到CSV文件。")
                    sftp.close()
                    ssh.close()
                except Exception as e:
                    st.error(f"连接失败或列出文件失败: {e}")

        if st.button("Add Data Source"):
            if not (ip and path and username and password and port):
                st.error("请填写所有字段。")
            else:
                config = {
                    "source_name": source_name,
                    "type": "Remote Server",
                    "ip": ip,
                    "port": port,
                    "path": path,
                    "username": username,
                    "password": password,
                    "created_at": datetime.now().isoformat()
                }
                try:
                    insert_or_update_data_source(config)
                    st.session_state.data_sources[source_name] = config
                    st.success(f"✅ 数据源 '{source_name}' 已成功添加并保存到数据库")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 保存到数据库失败: {e}")

    def render_data_cleaning(self):
        """Render data cleaning interface"""
        st.markdown("### 🧹 Data Cleaning")

        # Select dataset
        if 'user_data' in st.session_state and st.session_state.user_data:
            dataset_names = list(st.session_state.user_data.keys())
            selected_dataset = st.selectbox("Select Dataset", dataset_names, key="cleaning_dataset")

            if selected_dataset:
                df = st.session_state.user_data[selected_dataset]

                # Data quality check
                st.markdown("#### 📊 Data Quality Check")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    missing_count = df.isnull().sum().sum()
                    st.metric("Missing Values", missing_count)
                with col4:
                    duplicate_count = df.duplicated().sum()
                    st.metric("Duplicate Rows", duplicate_count)

                # Data overview
                st.markdown("#### 🔍 Data Overview")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Data Types**")
                    dtype_df = pd.DataFrame({
                        'Column Name': df.columns,
                        'Data Type': df.dtypes.astype(str),
                        'Missing Values': df.isnull().sum().values,
                        'Missing Rate': (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%'
                    })
                    st.dataframe(dtype_df, use_container_width=True)

                with col2:
                    st.markdown("**Data Distribution**")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        selected_col = st.selectbox("Select Numeric Column", numeric_cols, key="dist_col")
                        if selected_col:
                            fig = px.histogram(df, x=selected_col, title=f"{selected_col} Distribution")
                            st.plotly_chart(fig, use_container_width=True)

                # Data cleaning rules
                st.markdown("#### ⚙️ Data Cleaning Rules")

                cleaning_options = st.multiselect(
                    "Select Cleaning Operations",
                    [
                        "Delete Duplicate Rows",
                        "Handle Missing Values",
                        "Outlier Detection",
                        "Data Type Conversion",
                        "Column Name Standardization",
                        "Data Standardization/Normalization"
                    ],
                    key="cleaning_options"
                )

                # Display specific configurations based on selected cleaning operations
                if "Handle Missing Values" in cleaning_options:
                    st.markdown("**Missing Values Handling**")
                    missing_strategy = st.selectbox(
                        "Handling Strategy",
                        ["Delete Rows with Missing Values", "Delete Columns with Missing Values", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill"],
                        key="missing_strategy"
                    )

                if "Outlier Detection" in cleaning_options:
                    st.markdown("**Outlier Detection**")
                    outlier_method = st.selectbox(
                        "Detection Method",
                        ["IQR Method", "Z-Score Method", "Isolation Forest", "LOF Method"],
                        key="outlier_method"
                    )

                if "Data Standardization/Normalization" in cleaning_options:
                    st.markdown("**Data Standardization**")
                    scaling_method = st.selectbox(
                        "Scaling Method",
                        ["Z-Score Standardization", "Min-Max Normalization", "Robust Standardization"],
                        key="scaling_method"
                    )

                # Execute cleaning
                if st.button("🧹 Execute Data Cleaning", type="primary"):
                    cleaned_df = self.apply_cleaning_rules(df, cleaning_options)

                    # Save cleaned data
                    cleaned_name = f"{selected_dataset}_cleaned"
                    st.session_state.user_data[cleaned_name] = cleaned_df

                    st.success(f"✅ Data cleaning completed! Cleaned data saved as: {cleaned_name}")

                    # Display before and after comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Before**")
                        st.info(f"Shape: {df.shape}")
                        st.info(f"Missing Values: {df.isnull().sum().sum()}")

                    with col2:
                        st.markdown("**After**")
                        st.info(f"Shape: {cleaned_df.shape}")
                        st.info(f"Missing Values: {cleaned_df.isnull().sum().sum()}")

        else:
            st.info("Please load dataset first")

    def render_data_storage(self):
        """Render data storage interface"""
        st.markdown("### 💾 Data Storage")

        # Storage configuration
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Local Storage")

            # Display current dataset
            if 'user_data' in st.session_state and st.session_state.user_data:
                st.markdown("**Current Dataset**")
                for name, df in st.session_state.user_data.items():
                    with st.expander(f"📊 {name} ({df.shape[0]} rows x {df.shape[1]} columns)"):
                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            if st.button("💾 Save as CSV", key=f"save_csv_{name}"):
                                self.save_to_csv(df, name)

                        with col_b:
                            if st.button("💾 Save as Excel", key=f"save_excel_{name}"):
                                self.save_to_excel(df, name)

                        with col_c:
                            if st.button("💾 Save as Parquet", key=f"save_parquet_{name}"):
                                self.save_to_parquet(df, name)
            else:
                st.info("No dataset")

        with col2:
            st.markdown("#### Database Storage")

            # Database configuration
            with st.form("db_storage_form"):
                db_name = st.text_input("Database Name", value="quantitative_data.db")
                table_name = st.text_input("Table Name", placeholder="table_name")

                if_exists = st.selectbox(
                    "If Table Exists",
                    ["Replace", "Append", "Fail"],
                    key="if_exists"
                )

                if st.form_submit_button("Save to Database"):
                    if 'user_data' in st.session_state and st.session_state.user_data:
                        dataset_name = st.selectbox(
                            "Select Dataset",
                            list(st.session_state.user_data.keys()),
                            key="db_dataset"
                        )

                        if dataset_name and table_name:
                            self.save_to_database(
                                st.session_state.user_data[dataset_name],
                                db_name,
                                table_name,
                                if_exists
                            )

        # Storage history
        st.markdown("---")
        st.markdown("#### 📁 Storage History")

        storage_files = list(self.data_dir.glob("*"))
        if storage_files:
            for file_path in storage_files:
                with st.expander(f"📄 {file_path.name}"):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.text(f"Path: {file_path}")
                        st.text(f"Size: {file_path.stat().st_size / 1024:.1f} KB")
                        st.text(f"Modified Time: {datetime.fromtimestamp(file_path.stat().st_mtime)}")

                    with col2:
                        if st.button("📥 Load", key=f"load_{file_path.name}"):
                            self.load_from_file(file_path)

                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_file_{file_path.name}"):
                            file_path.unlink()
                            st.rerun()
        else:
            st.info("No storage files")

    def render_data_updates(self):
        """Render data updates interface"""
        st.markdown("### 🔄 Data Updates")

        # Update task management
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📅 Update Schedule")

            # Create update task
            with st.form("update_task_form"):
                task_name = st.text_input("Task Name", placeholder="e.g.: Daily Stock Price Update")

                # Select data source
                if st.session_state.data_sources:
                    data_source = st.selectbox(
                        "Data Source",
                        list(st.session_state.data_sources.keys()),
                        key="update_data_source"
                    )
                else:
                    st.warning("Please configure data source first")
                    data_source = None

                # Update frequency
                frequency = st.selectbox(
                    "Update Frequency",
                    ["Manual", "Hourly", "Daily", "Weekly", "Monthly"],
                    key="update_frequency"
                )

                # Update time
                if frequency != "Manual":
                    update_time = st.time_input("Update Time", value=datetime.now().time())

                # Data cleaning rules
                apply_cleaning = st.checkbox("Apply Data Cleaning Rules")

                if st.form_submit_button("Create Update Task"):
                    if task_name and data_source:
                        task_config = {
                            "name": task_name,
                            "data_source": data_source,
                            "frequency": frequency,
                            "update_time": update_time.isoformat() if frequency != "Manual" else None,
                            "apply_cleaning": apply_cleaning,
                            "created_at": datetime.now().isoformat(),
                            "last_run": None,
                            "status": "Pending Execution"
                        }

                        st.session_state.data_update_schedule[task_name] = task_config
                        st.success(f"✅ Update task '{task_name}' created successfully")
                        st.rerun()

        with col2:
            st.markdown("#### 📊 Update Status")

            if st.session_state.data_update_schedule:
                for task_name, task_config in st.session_state.data_update_schedule.items():
                    with st.expander(f"⏰ {task_name}", expanded=False):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.text(f"Data Source: {task_config['data_source']}")
                            st.text(f"Frequency: {task_config['frequency']}")
                            st.text(f"Status: {task_config['status']}")
                            st.text(f"Last Run: {task_config['last_run'] or 'Never Run'}")

                        with col_b:
                            if st.button("▶️ Execute Immediately", key=f"run_{task_name}"):
                                self.run_update_task(task_name)

                            if st.button("🗑️ Delete Task", key=f"delete_task_{task_name}"):
                                del st.session_state.data_update_schedule[task_name]
                                st.rerun()
            else:
                st.info("No update tasks")

        # Update log
        st.markdown("---")
        st.markdown("#### 📝 Update Log")

        # Here you can display update history records
        st.info("Update log functionality to be implemented")

    def render_data_overview(self):
        """Render data overview interface"""
        st.markdown("### 📊 Data Overview")

        if 'user_data' in st.session_state and st.session_state.user_data:
            # Dataset statistics
            col1, col2, col3, col4 = st.columns(4)

            total_datasets = len(st.session_state.user_data)
            total_rows = sum(len(df) for df in st.session_state.user_data.values())
            total_cols = sum(len(df.columns) for df in st.session_state.user_data.values())
            total_size = sum(df.memory_usage(deep=True).sum() for df in st.session_state.user_data.values())

            with col1:
                st.metric("Dataset Count", total_datasets)
            with col2:
                st.metric("Total Rows", f"{total_rows:,}")
            with col3:
                st.metric("Total Columns", total_cols)
            with col4:
                st.metric("Memory Usage", f"{total_size / 1024 / 1024:.1f} MB")

            # Dataset details
            st.markdown("---")
            st.markdown("#### 📋 Dataset Details")

            for name, df in st.session_state.user_data.items():
                with st.expander(f"📊 {name}", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Basic Information**")
                        st.text(f"Shape: {df.shape}")
                        st.text(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                        st.text(f"Missing Values: {df.isnull().sum().sum()}")
                        st.text(f"Duplicate Rows: {df.duplicated().sum()}")

                        # Data type distribution
                        st.markdown("**Data Types**")
                        dtype_counts = df.dtypes.value_counts()
                        for dtype, count in dtype_counts.items():
                            st.text(f"{dtype}: {count}")

                    with col2:
                        st.markdown("**Data Preview**")
                        st.dataframe(df.head(), use_container_width=True)

                        # Numeric column statistics
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            st.markdown("**Numeric Column Statistics**")
                            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No dataset")

            # Data import hint
            st.markdown("#### 💡 Data Import Hint")
            st.markdown("""
            You can import data through the following ways:
            1. **Data Source Management** - Configure various data sources
            2. **Local File Upload** - Upload CSV, Excel, etc. directly
            3. **API Interface** - Connect to third-party data services
            4. **Database Connection** - Import data from database
            """)

    # Helper methods
    def test_data_source(self, source_name):
        """Test data source connection"""
        config = st.session_state.data_sources[source_name]

        try:
            if config['type'] == "Local Files":
                # Test if file exists and is readable
                file_path = config['file_path']
                if os.path.exists(file_path):
                    # Try reading file header
                    if config['file_type'] == "CSV":
                        pd.read_csv(file_path, nrows=1)
                    elif config['file_type'] == "Excel":
                        pd.read_excel(file_path, nrows=1)
                    st.success("✅ File connection test successful")
                else:
                    st.error("❌ File does not exist")

            elif config['type'] == "API Interface":
                # Test API connection
                st.info("🔄 API connection test functionality to be implemented")

            elif config['type'] == "Database":
                # Test database connection
                st.info("🔄 Database connection test functionality to be implemented")

            else:
                st.info("🔄 Connection test functionality for this data source type to be implemented")

        except Exception as e:
            st.error(f"❌ Connection test failed: {e}")

    def apply_cleaning_rules(self, df, cleaning_options):
        """Apply data cleaning rules"""
        cleaned_df = df.copy()

        try:
            if "Delete Duplicate Rows" in cleaning_options:
                cleaned_df = cleaned_df.drop_duplicates()

            if "Handle Missing Values" in cleaning_options:
                # Simple missing values handling example
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())

            if "Column Name Standardization" in cleaning_options:
                # Standardize column names
                cleaned_df.columns = [col.lower().replace(' ', '_') for col in cleaned_df.columns]

            return cleaned_df

        except Exception as e:
            st.error(f"❌ Data cleaning failed: {e}")
            return df

    def save_to_csv(self, df, name):
        """Save as CSV file"""
        try:
            file_path = self.data_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            st.success(f"✅ Data saved as: {file_path}")
        except Exception as e:
            st.error(f"❌ Save failed: {e}")

    def save_to_excel(self, df, name):
        """Save as Excel file"""
        try:
            file_path = self.data_dir / f"{name}.xlsx"
            df.to_excel(file_path, index=False)
            st.success(f"✅ Data saved as: {file_path}")
        except Exception as e:
            st.error(f"❌ Save failed: {e}")

    def save_to_parquet(self, df, name):
        """Save as Parquet file"""
        try:
            file_path = self.data_dir / f"{name}.parquet"
            df.to_parquet(file_path, index=False)
            st.success(f"✅ Data saved as: {file_path}")
        except Exception as e:
            st.error(f"❌ Save failed: {e}")

    def save_to_database(self, df, db_name, table_name, if_exists):
        """Save to database"""
        try:
            db_path = self.data_dir / db_name
            conn = sqlite3.connect(db_path)

            if_exists_map = {"Replace": "replace", "Append": "append", "Fail": "fail"}
            df.to_sql(table_name, conn, if_exists=if_exists_map[if_exists], index=False)

            conn.close()
            st.success(f"✅ Data saved to database: {db_path}, Table: {table_name}")
        except Exception as e:
            st.error(f"❌ Save failed: {e}")

    def load_from_file(self, file_path):
        """Load data from file"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                st.error(f"Unsupported file format: {file_path.suffix}")
                return

            # Add to user data
            dataset_name = file_path.stem
            st.session_state.user_data[dataset_name] = df
            st.success(f"✅ Data loaded: {dataset_name}")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Load failed: {e}")

    def run_update_task(self, task_name):
        """Execute update task"""
        try:
            task_config = st.session_state.data_update_schedule[task_name]

            # Update task status
            task_config['status'] = 'Executing'
            task_config['last_run'] = datetime.now().isoformat()

            # Here should execute actual data update based on data source type
            # Currently just simulating
            st.info(f"🔄 Executing update task: {task_name}")

            # Simulated execution successful
            task_config['status'] = 'Execution Successful'
            st.success(f"✅ Update task '{task_name}' executed successfully")

        except Exception as e:
            # Update task status to failed
            if task_name in st.session_state.data_update_schedule:
                st.session_state.data_update_schedule[task_name]['status'] = 'Execution Failed'
            st.error(f"❌ Update task failed: {e}")
