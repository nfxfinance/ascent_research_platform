#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
import logging
import time
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
from streamlit_local_storage import LocalStorage
localS = LocalStorage()

# Configure Streamlit page
st.set_page_config(
    page_title="Quantitative Backtesting Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Authentication Configuration ---
UNIFIED_PASSWORD = "Fin_2024"  # Unified password, can be set as environment variable
SESSION_TIMEOUT = 2400000 * 60 * 60  # 24 hours timeout (seconds)

# --- System Configuration ---
ROOTPATH = os.getenv("CA_QROOT")
if ROOTPATH:
    sys.path.append(ROOTPATH + '/python')
else:
    project_python_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_python_path not in sys.path:
        sys.path.insert(0, project_python_path)

# Import modules - with error handling
try:
    from modules.factor_analysis import FactorAnalysisModule
    from modules.backtesting import BacktestingModule
    from modules.data_management import DataManagementModule
    from modules.result_manager import ResultManager
    from lib.utils import setup_logger
    from routes import (
        MODULE_ROUTES, ROUTE_MODULES, ROUTE_METADATA,
        get_module_name, get_route_name, generate_module_url,
        is_valid_route, ROUTE_PARAMS, DEFAULT_MODULE
    )

except ImportError as e:
    st.error(f"❌ Unable to import required modules: {e}")
    st.info("Please ensure all modules are correctly installed")
    st.stop()

# Configure logger
try:
    setup_logger()
except:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Persistent Session Management ---

def get_session_file_path(username: str):
    """Get session file path for a specific user"""
    session_dir = Path("sessions")
    session_dir.mkdir(exist_ok=True)
    return session_dir / f"session_{username}.json"

def create_persistent_session(username: str, login_time: float, session_token: str):
    """Create persistent session for a user"""
    session_data = {
        'username': username,
        'login_time': login_time,
        'session_token': session_token,
        'authenticated': True
    }
    try:
        session_file = get_session_file_path(username)
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Persistent session created for user: {username}")
    except Exception as e:
        logger.warning(f"Failed to create persistent session: {e}")

def get_persistent_session(username: str):
    """Get persistent session for a user"""
    try:
        session_file = get_session_file_path(username)
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                return session_data
        return None
    except Exception as e:
        logger.warning(f"Error getting persistent session: {e}")
        return None

def clear_persistent_session(username: str):
    """Clear persistent session for a user"""
    try:
        session_file = get_session_file_path(username)
        if session_file.exists():
            session_file.unlink()
        logger.debug(f"Persistent session cleared for user: {username}")
    except Exception as e:
        logger.warning(f"Error clearing persistent session: {e}")

def restore_session_from_persistent(username: str):
    """Restore session from persistent storage for a user"""
    try:
        # First check if session state is already valid
        if is_session_valid():
            return True
        # Try to restore from persistent storage
        persistent_data = get_persistent_session(username)
        if persistent_data:
            # Validate persistent session validity
            current_time = time.time()
            login_time = persistent_data.get('login_time', 0)
            # Check if expired
            if current_time - login_time <= SESSION_TIMEOUT:
                # Restore to session state
                st.session_state.authenticated = persistent_data.get('authenticated', False)
                st.session_state.username = persistent_data.get('username', '')
                st.session_state.login_time = login_time
                st.session_state.session_token = persistent_data.get('session_token', '')
                # Validate restored session
                if is_session_valid():
                    logger.info(f"Session restored for user: {st.session_state.username}")
                    return True
                else:
                    # If restored session is invalid, clear persistent data
                    clear_persistent_session(username)
            else:
                # If session expired, clear persistent data
                clear_persistent_session(username)
                logger.info(f"Persistent session expired and cleared for user: {username}")
        return False
    except Exception as e:
        logger.warning(f"Error restoring session: {e}")
        return False

# --- Authentication Functions ---

def generate_session_token(username: str, timestamp: str) -> str:
    """Generate session token"""
    secret_key = "quant_platform_secret_2024"  # Can be set as environment variable
    data = f"{username}:{timestamp}:{secret_key}"
    return hashlib.sha256(data.encode()).hexdigest()

def verify_session_token(username: str, timestamp: str, token: str) -> bool:
    """Verify session token"""
    expected_token = generate_session_token(username, timestamp)
    return hmac.compare_digest(expected_token, token)

def is_session_valid() -> bool:
    """Check if session is valid"""
    if not all(key in st.session_state for key in ['authenticated', 'username', 'login_time', 'session_token']):
        return False

    # Check if session expired
    current_time = time.time()
    if current_time - st.session_state.get('login_time', 0) > SESSION_TIMEOUT:
        return False

    # Verify session token
    login_time_str = str(st.session_state.get('login_time', ''))
    username = st.session_state.get('username', '')
    session_token = st.session_state.get('session_token', '')

    return verify_session_token(username, login_time_str, session_token)

def authenticate_user(username: str, password: str) -> bool:
    """Verify user credentials"""
    if not username or not username.strip():
        return False

    return password == UNIFIED_PASSWORD

def login_user(username: str):
    """Login user"""
    current_time = time.time()
    login_time_str = str(current_time)
    session_token = generate_session_token(username, login_time_str)

    st.session_state.authenticated = True
    st.session_state.username = username.strip()
    st.session_state.login_time = current_time
    st.session_state.session_token = session_token

    # Create persistent session
    try:
        create_persistent_session(username.strip(), current_time, session_token)
    except Exception as e:
        logger.warning(f"Failed to create persistent session: {e}")

    # 写入 localStorage
    localS.setItem("username", username.strip())

    logger.info(f"User {username} logged in successfully")

def logout_user():
    """Logout user"""
    username = st.session_state.get('username', 'Unknown')

    # Clear authentication-related session state
    auth_keys = ['authenticated', 'username', 'login_time', 'session_token']
    for key in auth_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Clear persistent session
    clear_persistent_session(username)

    # 清空 localStorage
    localS.setItem("username", "")

    logger.info(f"User {username} logged out")

def render_login_page():
    """Render login page"""
    st.markdown("""
    <div style='text-align: center; margin-top: 2rem;'>
        <h1>🚀 Quantitative Backtesting Platform</h1>
        <h3>🔐 User Login</h3>
    </div>
    """, unsafe_allow_html=True)

    # Center login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("---")

        with st.form("login_form"):
            st.markdown("### Please enter login information")

            username = st.text_input(
                "Username",
                placeholder="Enter any username as identifier",
                help="Username will be used for subsequent operation records and identification"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter unified password",
                help="All users use the unified password"
            )

            submitted = st.form_submit_button("🔑 Login", use_container_width=True)

            if submitted:
                if authenticate_user(username, password):
                    login_user(username)
                    st.success(f"✅ Welcome {username}! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    if not username or not username.strip():
                        st.error("❌ Please enter username")
                    else:
                        st.error("❌ Incorrect password, please try again")

        st.markdown("---")

        # Help information
        with st.expander("❓ Login Help"):
            st.markdown("""
            **Login Instructions:**
            - Username: Enter any username for subsequent operation identification
            - Password: All users use the unified password
            - Valid for 24 hours after login, no need to re-login

            **Function Modules:**
            - 🔍 Factor Analysis
            - 📊 Data Management
            - 📈 Strategy Backtesting
            - 📋 Result Management
            """)

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 3rem;'>
        <p>🔒 Secure Login | One-time authentication, valid for 24 hours</p>
    </div>
    """, unsafe_allow_html=True)

# --- URL and Route Management ---

def get_current_module_from_url():
    """Get current module from URL parameters"""
    try:
        query_params = st.query_params

        # Check all possible route parameters
        for param in ROUTE_PARAMS:
            if param in query_params:
                route_value = query_params[param].lower()
                if is_valid_route(route_value):
                    return get_module_name(route_value)
    except Exception as e:
        logger.warning(f"Error getting query params: {e}")

    return None

def update_url_for_module(module_name):
    """Update URL when module changes"""
    try:
        route = get_route_name(module_name)
        if route:
            st.query_params.page = route
    except Exception as e:
        logger.warning(f"Error updating URL: {e}")

# --- Session State Initialization ---

def initialize_session_state():
    """Initialize Session State variables"""

    # Check initial module from URL
    url_module = get_current_module_from_url()

    # Set default module
    try:
        default_module = DEFAULT_MODULE
    except:
        default_module = "Factor Analysis"

    if 'current_module' not in st.session_state:
        if url_module:
            st.session_state.current_module = url_module
        else:
            st.session_state.current_module = default_module
    elif url_module and url_module != st.session_state.current_module:
        st.session_state.current_module = url_module

    # Initialize other states
    defaults = {
        'user_data': {},
        'analysis_results': {},
        'saved_reports': []
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- UI Rendering Functions ---

def render_usage_guide_module():
    """Render the Usage Guide page with Markdown support"""
    st.markdown("## 📖 Usage Guide")
    md_path = Path("docs/usage.md")
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 处理图片路径，将相对路径转换为绝对路径
        import re
        # 查找markdown中的图片引用
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

        def replace_img_path(match):
            alt_text = match.group(1)
            img_path = match.group(2)

            # 如果是相对路径，转换为绝对路径
            if not img_path.startswith(('http://', 'https://', 'data:')):
                full_img_path = Path("docs") / img_path
                if full_img_path.exists():
                    # 使用streamlit的image组件显示图片
                    return f'<img src="data:image/png;base64,{get_base64_image(str(full_img_path))}" alt="{alt_text}" style="max-width: 100%; height: auto;">'
                else:
                    return f'*[image not found: {img_path}]*'
            return match.group(0)

        # 替换图片引用
        md_content = re.sub(img_pattern, replace_img_path, md_content)
        st.markdown(md_content, unsafe_allow_html=True)
    else:
        st.warning("Usage guide document not found (docs/usage.md)")

def get_base64_image(image_path):
    """将图片转换为base64编码"""
    import base64
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

def render_navigation():
    """Render main navigation sidebar"""
    with st.sidebar:
        st.markdown("# 🚀 Quantitative Platform")

        # Display current user information
        st.markdown("---")
        st.markdown("### 👤 User Information")
        st.markdown(f"**User:** {st.session_state.username}")

        # Calculate remaining time
        remaining_time = SESSION_TIMEOUT - (time.time() - st.session_state.login_time)
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        # st.markdown(f"**Session Remaining:** {hours}h {minutes}m")

        # Logout button
        if st.button("🚪 Logout", key="logout_btn"):
            logout_user()
            st.rerun()

        st.markdown("---")

        # Module navigation
        st.markdown("### 🧭 Module Navigation")

        # Available modules list
        available_modules = ["Factor Analysis", "Data Management", "Strategy Backtesting", "Result Management", "Usage Guide"]

        for module_name in available_modules:
            icon = {
                "Factor Analysis": "🔍",
                "Data Management": "📊",
                "Strategy Backtesting": "📈",
                "Result Management": "📋",
                "Usage Guide": "📖"
            }.get(module_name, "📄")

            # Highlight current module
            if st.session_state.current_module == module_name:
                st.markdown(f"🔸 **{icon} {module_name}** *(Current)*")
            else:
                if st.button(f"{icon} {module_name}", key=f"nav_{module_name}"):
                    st.session_state.current_module = module_name

                    try:
                        update_url_for_module(module_name)
                    except:
                        pass
                    st.rerun()

        st.markdown("---")

        # Quick statistics
        st.markdown("### 📊 Quick Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Datasets", len(st.session_state.user_data))
        with col2:
            st.metric("Saved Reports", len(st.session_state.saved_reports))

def render_current_module():
    """Render currently selected module"""
    module_name = st.session_state.current_module

    # Try to render specific module, display fallback page if module doesn't exist
    try:
        if module_name == "Factor Analysis":
            factor_module = FactorAnalysisModule()
            factor_module.render()

        elif module_name == "Data Management":
            data_module = DataManagementModule()
            data_module.render()

        elif module_name == "Strategy Backtesting":
            backtest_module = BacktestingModule()
            backtest_module.render()

        elif module_name == "Result Management":
            result_module = ResultManager()
            result_module.render()
        elif module_name == "Usage Guide":
            render_usage_guide_module()

        else:
            render_placeholder_module(module_name)

    except Exception as e:
        logger.error(f"Error rendering module {module_name}: {e}")
        render_placeholder_module(module_name, error=str(e))

def render_placeholder_module(module_name, error=None):
    """Render placeholder module page"""
    icon = {
        "Factor Analysis": "🔍",
        "Data Management": "📊",
        "Strategy Backtesting": "📈",
        "Result Management": "📋",
        "Usage Guide": "📖"
    }.get(module_name, "📄")

    st.markdown(f"## {icon} {module_name}")

    if error:
        st.error(f"❌ Module loading failed: {error}")
    else:
        st.info(f"🚧 {module_name} module is under development...")

    # Display module feature preview
    module_features = {
        "Factor Analysis": [
            "📈 Factor data loading and preprocessing",
            "🔍 Factor effectiveness analysis",
            "📊 Factor correlation analysis",
            "🎯 Factor IC analysis",
            "📋 Analysis report generation"
        ],
        "Data Management": [
            "📂 Data file management",
            "🔄 Data format conversion",
            "🧹 Data cleaning and processing",
            "💾 Data storage management",
            "📊 Data quality inspection"
        ],
        "Strategy Backtesting": [
            "📈 Strategy construction and configuration",
            "🔄 Historical data backtesting",
            "📊 Backtest result analysis",
            "🎯 Risk metric calculation",
            "📋 Backtest report generation"
        ],
        "Result Management": [
            "📋 Result report management",
            "📊 Result visualization display",
            "💾 Result export functionality",
            "🔍 Historical result query",
            "📈 Result comparison analysis"
        ],
        "Usage Guide": [
            "📖 Usage guide document",
        ]
    }

    if module_name in module_features:
        st.markdown("### 🎯 Module Features")
        for feature in module_features[module_name]:
            st.markdown(f"- {feature}")

    # Quick action buttons
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 View Sample Data", key=f"demo_data_{module_name}"):
            st.info("Sample data feature under development...")

    with col2:
        if st.button("📖 View Documentation", key=f"docs_{module_name}"):
            st.info("Documentation feature under development...")

    with col3:
        if st.button("🔧 Module Settings", key=f"settings_{module_name}"):
            st.info("Settings feature under development...")

def render_footer():
    """Render footer"""
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🚀 Quantitative Backtesting Platform v2.0 | User: {st.session_state.username} | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick navigation buttons
    st.markdown("### ⚡ Quick Navigation")
    col1, col2, col3, col4, col5 = st.columns(5)

    modules = [
        ("Factor Analysis", "🔍"),
        ("Data Management", "📊"),
        ("Strategy Backtesting", "📈"),
        ("Result Management", "📋"),
        ("Usage Guide", "📖")
    ]

    for i, (module_name, icon) in enumerate(modules):
        with [col1, col2, col3, col4, col5][i]:
            if st.button(f"{icon} {module_name}", key=f"footer_nav_{module_name}"):
                st.session_state.current_module = module_name
                try:
                    update_url_for_module(module_name)
                except:
                    pass
                st.rerun()

def optimize_for_print():
    """优化打印布局"""
    st.markdown("""
    <style>
    /* 全局打印优化 */
    @media print {
        .main .block-container {
            max-width: 100% !important;
            padding: 0 !important;
        }

        /* 确保图表完整显示 */
        .stPlotlyChart > div {
            width: 100% !important;
            height: auto !important;
        }

        /* 隐藏Streamlit控件 */
        .stSidebar, .stButton, .stSelectbox,
        .stNumberInput, .stCheckbox, .stExpander > summary {
            display: none !important;
        }

        /* 强制expander展开 */
        .stExpander > div[role="button"] + div {
            display: block !important;
        }

        /* 避免在图表中间分页 */
        .element-container {
            page-break-inside: avoid !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
# 在你的主应用中调用

# --- Main Application Function ---

def main():
    """Main application entry point"""

    # 优先从 localStorage 读取用户名
    username = localS.getItem("username")
    if not username:
        username = st.session_state.get('username', '')

    restore_session_from_persistent(username)

    # Check if session is valid
    if not is_session_valid():
        render_login_page()
        return

    # Initialize session state
    initialize_session_state()

    # Render navigation
    render_navigation()

    optimize_for_print()

    # Main content area
    with st.container():
        # Module title and route information
        try:
            current_route = get_route_name(st.session_state.current_module)
            current_metadata = ROUTE_METADATA.get(current_route, {}) if current_route else {}
            icon = current_metadata.get('icon', '📄')
        except:
            icon = "📄"
            current_route = None

        st.title(f"{icon} Quantitative Backtesting Platform - {st.session_state.current_module}")

        # Module breadcrumb navigation
        breadcrumb = f"**Current Location:** Home > {st.session_state.current_module}"
        if current_route:
            breadcrumb += f" (`?page={current_route}`)"
        st.markdown(breadcrumb)

        # Display category information
        try:
            category = current_metadata.get('category')
            if category:
                st.caption(f"🏷️ Category: {category}")
        except:
            pass

        st.markdown("---")

        # Render current module
        render_current_module()

        # Footer quick navigation
        render_footer()





# --- Application Entry Point ---

if __name__ == "__main__":
    # Run main application
    main()


