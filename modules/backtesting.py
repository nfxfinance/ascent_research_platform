#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

class BacktestingModule:
    """Strategy Backtesting Module - Strategy Backtesting, Performance Analysis, Risk Metrics"""

    def __init__(self):
        self.name = "Strategy Backtesting"
        self.description = "Strategy backtesting, performance analysis, risk metrics"
        self.initialize_state()

    def initialize_state(self):
        """Initialize module state"""
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = {}
        if 'strategy_configs' not in st.session_state:
            st.session_state.strategy_configs = {}
        if 'backtest_history' not in st.session_state:
            st.session_state.backtest_history = []
        if 'current_backtest_result' not in st.session_state:
            st.session_state.current_backtest_result = None
        if 'show_deep_analysis' not in st.session_state:
            st.session_state.show_deep_analysis = False

    def render(self):
        """Render strategy backtesting module interface"""
        st.markdown("## 📈 Strategy Backtesting Module")

        # Check if showing deep analysis
        if st.session_state.show_deep_analysis and st.session_state.current_backtest_result:
            self.render_deep_analysis()
        else:
            self.render_basic_interface()

    def render_basic_interface(self):
        """Render basic interface - Four-quadrant layout"""
        st.markdown("*Quantitative strategy development and backtesting platform*")

        # Initialize log state
        if 'backtest_logs' not in st.session_state:
            st.session_state.backtest_logs = []
        if 'strategy_code' not in st.session_state:
            st.session_state.strategy_code = self.get_default_strategy_code()

        # Initialize layout ratios
        if 'layout_ratios' not in st.session_state:
            st.session_state.layout_ratios = {
                'width_left': 4,
                'width_right': 6,
                'height_top': 5,
                'height_bottom': 5
            }

        # Layout control panel
        self.render_layout_controls()

        # Calculate actual ratios
        left_ratio = st.session_state.layout_ratios['width_left']
        right_ratio = st.session_state.layout_ratios['width_right']
        top_ratio = st.session_state.layout_ratios['height_top']
        bottom_ratio = st.session_state.layout_ratios['height_bottom']

        # Four-quadrant layout
        col1, col2 = st.columns([left_ratio, right_ratio])

        with col1:
            # Top left: Data selection and parameter configuration
            with st.container():
                st.markdown("### 📊 Data Configuration")
                # Dynamic height container
                data_container = st.container()
                with data_container:
                    self.render_data_config_panel()

                # Add placeholder space to control height ratio
                if top_ratio < bottom_ratio:
                    for _ in range(bottom_ratio - top_ratio):
                        st.write("")

            st.markdown("---")

            # Bottom left: Net value curve display
            with st.container():
                st.markdown("### 📈 Account Value")
                # Dynamic height container
                curve_container = st.container()
                with curve_container:
                    self.render_account_curve_panel()

                # Add placeholder space to control height ratio
                if bottom_ratio < top_ratio:
                    for _ in range(top_ratio - bottom_ratio):
                        st.write("")

        with col2:
            # Top right: Code editor
            with st.container():
                st.markdown("### 💻 Strategy Code")
                # Dynamic height container
                code_container = st.container()
                with code_container:
                    self.render_code_editor_panel()

                # Add placeholder space to control height ratio
                if top_ratio < bottom_ratio:
                    for _ in range(bottom_ratio - top_ratio):
                        st.write("")

            st.markdown("---")

            # Bottom right: Run log
            with st.container():
                st.markdown("### 📋 Execution Log")
                # Dynamic height container
                log_container = st.container()
                with log_container:
                    self.render_log_panel()

                # Add placeholder space to control height ratio
                if bottom_ratio < top_ratio:
                    for _ in range(top_ratio - bottom_ratio):
                        st.write("")

    def render_layout_controls(self):
        """Render layout control panel"""
        with st.expander("🎛️ Layout Adjustment", expanded=False):
            st.markdown("**Adjust four-quadrant layout ratios**")

            # Initialize custom layout presets
            if 'custom_layout_presets' not in st.session_state:
                st.session_state.custom_layout_presets = {}

            # Width ratio controls
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Width Ratios**")
                width_col1, width_col2 = st.columns(2)

                with width_col1:
                    left_width = st.slider(
                        "Left Width",
                        min_value=1,
                        max_value=9,
                        value=st.session_state.layout_ratios['width_left'],
                        key="width_left_slider"
                    )

                with width_col2:
                    right_width = st.slider(
                        "Right Width",
                        min_value=1,
                        max_value=9,
                        value=st.session_state.layout_ratios['width_right'],
                        key="width_right_slider"
                    )

            with col2:
                st.markdown("**Height Ratios**")
                height_col1, height_col2 = st.columns(2)

                with height_col1:
                    top_height = st.slider(
                        "Top Height",
                        min_value=1,
                        max_value=9,
                        value=st.session_state.layout_ratios['height_top'],
                        key="height_top_slider"
                    )

                with height_col2:
                    bottom_height = st.slider(
                        "Bottom Height",
                        min_value=1,
                        max_value=9,
                        value=st.session_state.layout_ratios['height_bottom'],
                        key="height_bottom_slider"
                    )

            # Preset layout buttons
            st.markdown("**Preset Layouts**")
            preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

            with preset_col1:
                if st.button("📊 Data Priority", use_container_width=True):
                    st.session_state.layout_ratios = {
                        'width_left': 6, 'width_right': 4,
                        'height_top': 7, 'height_bottom': 3
                    }
                    st.rerun()

            with preset_col2:
                if st.button("💻 Code Priority", use_container_width=True):
                    st.session_state.layout_ratios = {
                        'width_left': 4, 'width_right': 6,
                        'height_top': 7, 'height_bottom': 3
                    }
                    st.rerun()

            with preset_col3:
                if st.button("📈 Chart Priority", use_container_width=True):
                    st.session_state.layout_ratios = {
                        'width_left': 6, 'width_right': 4,
                        'height_top': 3, 'height_bottom': 7
                    }
                    st.rerun()

            with preset_col4:
                if st.button("📋 Log Priority", use_container_width=True):
                    st.session_state.layout_ratios = {
                        'width_left': 4, 'width_right': 6,
                        'height_top': 3, 'height_bottom': 7
                    }
                    st.rerun()

            # Update layout ratios
            st.session_state.layout_ratios.update({
                'width_left': left_width,
                'width_right': right_width,
                'height_top': top_height,
                'height_bottom': bottom_height
            })

            # Custom preset management
            st.markdown("---")
            st.markdown("**Custom Presets**")

            # Save current layout as preset
            preset_name = st.text_input("Preset Name", placeholder="Enter preset name")
            if st.button("💾 Save Current Layout") and preset_name:
                st.session_state.custom_layout_presets[preset_name] = st.session_state.layout_ratios.copy()
                st.success(f"✅ Layout preset '{preset_name}' saved")

            # Load custom presets
            if st.session_state.custom_layout_presets:
                selected_preset = st.selectbox(
                    "Load Custom Preset",
                    list(st.session_state.custom_layout_presets.keys()),
                    key="custom_preset_select"
                )

                load_col, delete_col = st.columns(2)

                with load_col:
                    if st.button("📥 Load Preset"):
                        st.session_state.layout_ratios = st.session_state.custom_layout_presets[selected_preset].copy()
                        st.rerun()

                with delete_col:
                    if st.button("🗑️ Delete Preset"):
                        del st.session_state.custom_layout_presets[selected_preset]
                        st.rerun()

    def render_data_config_panel(self):
        """Render data configuration panel"""
        # Check if there's data
        if 'user_data' not in st.session_state or not st.session_state.user_data:
            st.warning("⚠️ Please load data first")
            st.markdown("""
            ### 💡 Data Loading Guide

            Please load historical price data through the **Data Management Module**.

            **Required data format:**
            - Date column (index or column)
            - Price columns (Open, High, Low, Close)
            - Volume column (optional)
            """)
            return

        # Dataset selection
        dataset_names = list(st.session_state.user_data.keys())
        selected_dataset = st.selectbox(
            "Select Dataset",
            dataset_names,
            key="backtest_dataset",
            help="Choose the dataset for backtesting"
        )

        if selected_dataset:
            df = st.session_state.user_data[selected_dataset]

            # Data overview
            with st.expander("📋 Data Overview", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Records", len(df))
                    st.metric("Columns", len(df.columns))

                with col2:
                    if hasattr(df.index, 'min'):
                        st.metric("Start Date", df.index.min().strftime('%Y-%m-%d') if hasattr(df.index.min(), 'strftime') else str(df.index.min()))
                        st.metric("End Date", df.index.max().strftime('%Y-%m-%d') if hasattr(df.index.max(), 'strftime') else str(df.index.max()))

                # Show first few rows
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(3), use_container_width=True)

            # Strategy parameters
            with st.expander("⚙️ Strategy Parameters", expanded=True):
                # Strategy type selection
                strategy_type = st.selectbox(
                    "Strategy Type",
                    ["Momentum Strategy", "Mean Reversion", "Buy and Hold", "Custom Strategy"],
                    key="strategy_type",
                    help="Choose the type of strategy to backtest"
                )

                # Common parameters
                col1, col2 = st.columns(2)

                with col1:
                    initial_capital = st.number_input(
                        "Initial Capital",
                        value=100000.0,
                        min_value=1000.0,
                        step=1000.0,
                        key="initial_capital",
                        help="Starting capital for backtesting"
                    )

                    commission = st.number_input(
                        "Commission Rate (%)",
                        value=0.1,
                        min_value=0.0,
                        max_value=5.0,
                        step=0.01,
                        key="commission",
                        help="Transaction commission rate"
                    ) / 100

                with col2:
                    if strategy_type in ["Momentum Strategy", "Mean Reversion"]:
                        lookback_period = st.selectbox(
                            "Lookback Period",
                            ["5 days", "10 days", "20 days", "30 days", "60 days"],
                            index=2,
                            key="lookback_period",
                            help="Period for calculating technical indicators"
                        )

                    slippage = st.number_input(
                        "Slippage (%)",
                        value=0.05,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        key="slippage",
                        help="Market impact and slippage"
                    ) / 100

            # Quick backtest button
            st.markdown("---")
            if st.button("🚀 Quick Backtest", type="primary", use_container_width=True):
                self.add_log(f"Starting {strategy_type} backtest on {selected_dataset}", "INFO")

                try:
                    # Run simplified backtest
                    lookback_days = self.get_days_from_period(lookback_period) if strategy_type in ["Momentum Strategy", "Mean Reversion"] else 20

                    result = self.simulate_backtest_result(
                        df, strategy_type, lookback_days, initial_capital
                    )

                    if result:
                        st.session_state.current_backtest_result = result
                        self.add_log("✅ Backtest completed successfully!", "SUCCESS")

                        # Show quick results
                        with st.container():
                            st.markdown("**Quick Results:**")
                            metric_col1, metric_col2 = st.columns(2)

                            with metric_col1:
                                st.metric(
                                    "Total Return",
                                    f"{result['total_return']:.2%}",
                                    help="Total return over the backtesting period"
                                )

                            with metric_col2:
                                st.metric(
                                    "Sharpe Ratio",
                                    f"{result['sharpe_ratio']:.3f}",
                                    help="Risk-adjusted return measure"
                                )

                            # Deep analysis button
                            if st.button("🔍 Deep Analysis", use_container_width=True):
                                st.session_state.show_deep_analysis = True
                                st.rerun()

                    else:
                        self.add_log("❌ Backtest failed", "ERROR")

                except Exception as e:
                    self.add_log(f"❌ Backtest error: {str(e)}", "ERROR")

    def render_code_editor_panel(self):
        """Render code editor panel"""
        # Strategy template selection
        template_options = {
            "Default Strategy": "default",
            "Momentum Strategy": "momentum",
            "Value Strategy": "value",
            "Mean Reversion": "mean_reversion",
            "Custom Strategy": "custom"
        }

        selected_template = st.selectbox(
            "Strategy Template",
            list(template_options.keys()),
            key="strategy_template",
            help="Choose a strategy template to start with"
        )

        # Template loading
        if st.button("📜 Load Template", use_container_width=True):
            template_type = template_options[selected_template]

            if template_type == "momentum":
                st.session_state.strategy_code = self.get_momentum_template()
            elif template_type == "value":
                st.session_state.strategy_code = self.get_value_template()
            elif template_type == "default":
                st.session_state.strategy_code = self.get_default_strategy_code()
            else:
                st.session_state.strategy_code = self.get_default_strategy_code()

            self.add_log(f"Loaded {selected_template} template", "INFO")
            st.rerun()

        # Code editor
        st.markdown("---")
        new_code = st.text_area(
            "Strategy Code",
            value=st.session_state.strategy_code,
            height=300,
            key="code_editor",
            help="Edit your strategy code here"
        )

        # Update code if changed
        if new_code != st.session_state.strategy_code:
            st.session_state.strategy_code = new_code

        # Code action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Save Code", use_container_width=True):
                self.save_strategy_code()

        with col2:
            if st.button("✅ Validate", use_container_width=True):
                try:
                    # Simple syntax check
                    compile(st.session_state.strategy_code, '<string>', 'exec')
                    self.add_log("✅ Code syntax is valid", "SUCCESS")
                except SyntaxError as e:
                    self.add_log(f"❌ Syntax error: {e}", "ERROR")

        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.strategy_code = self.get_default_strategy_code()
                self.add_log("🔄 Code reset to default", "INFO")
                st.rerun()

        # Code help
        with st.expander("📖 Code Help", expanded=False):
            st.markdown("""
            **Available Variables:**
            - `data`: Current price data (DataFrame)
            - `position`: Current position (-1, 0, 1)
            - `cash`: Available cash
            - `portfolio_value`: Total portfolio value

            **Return Values:**
            - Return 1 for buy signal
            - Return -1 for sell signal
            - Return 0 for hold

            **Example:**
            ```python
            def strategy(data, position, cash, portfolio_value):
                # Simple momentum strategy
                if len(data) < 20:
                    return 0

                current_price = data['close'].iloc[-1]
                ma_20 = data['close'].rolling(20).mean().iloc[-1]

                if current_price > ma_20 and position == 0:
                    return 1  # Buy signal
                elif current_price < ma_20 and position == 1:
                    return -1  # Sell signal
                else:
                    return 0  # Hold
            ```
            """)

    def render_account_curve_panel(self):
        """Render account curve panel"""
        if not st.session_state.current_backtest_result:
            st.info("💡 Run a backtest to see the account value curve")

            # Show placeholder chart
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            placeholder_data = pd.DataFrame({
                'Date': dates,
                'Portfolio_Value': 100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.001)
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=placeholder_data['Date'],
                y=placeholder_data['Portfolio_Value'],
                mode='lines',
                name='Portfolio Value (Demo)',
                line=dict(color='lightgray', dash='dash')
            ))

            fig.update_layout(
                title="Account Value Curve (Demo)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=250
            )

            st.plotly_chart(fig, use_container_width=True)
            return

        # Display actual backtest results
        result = st.session_state.current_backtest_result

        # Performance metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Final Value",
                f"${result['final_value']:,.0f}",
                f"${result['final_value'] - result['initial_capital']:,.0f}"
            )

        with col2:
            st.metric(
                "Total Return",
                f"{result['total_return']:.2%}",
                help="Total return over the period"
            )

        with col3:
            st.metric(
                "Max Drawdown",
                f"{result['max_drawdown']:.2%}",
                help="Maximum decline from peak"
            )

        # Account value curve
        portfolio_data = result['portfolio_history']

        fig = go.Figure()

        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))

        # Benchmark (buy and hold) if available
        if 'benchmark_value' in portfolio_data.columns:
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['benchmark_value'],
                mode='lines',
                name='Benchmark',
                line=dict(color='#ff7f0e', width=1, dash='dash')
            ))

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=250,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_log_panel(self):
        """Render log panel"""
        # Log controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            log_level_filter = st.selectbox(
                "Filter by Level",
                ["ALL", "INFO", "SUCCESS", "WARNING", "ERROR"],
                key="log_level_filter",
                help="Filter logs by level"
            )

        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.backtest_logs = []
                st.rerun()

        with col3:
            auto_scroll = st.checkbox("Auto Scroll", value=True, key="auto_scroll")

        # Display logs
        if st.session_state.backtest_logs:
            # Filter logs based on level
            filtered_logs = self.filter_logs(st.session_state.backtest_logs, log_level_filter)

            # Create log display
            log_container = st.container()

            with log_container:
                # Reverse logs to show newest first
                display_logs = list(reversed(filtered_logs))

                if auto_scroll:
                    # Show only recent logs for auto-scroll
                    display_logs = display_logs[:20]

                for log_entry in display_logs:
                    timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
                    level = log_entry['level']
                    message = log_entry['message']

                    # Color coding based on level
                    if level == "ERROR":
                        st.error(f"[{timestamp}] {message}")
                    elif level == "WARNING":
                        st.warning(f"[{timestamp}] {message}")
                    elif level == "SUCCESS":
                        st.success(f"[{timestamp}] {message}")
                    else:
                        st.info(f"[{timestamp}] {message}")

        else:
            st.info("📝 Execution logs will appear here")

        # Log statistics
        if st.session_state.backtest_logs:
            total_logs = len(st.session_state.backtest_logs)
            error_count = len([log for log in st.session_state.backtest_logs if log['level'] == 'ERROR'])
            success_count = len([log for log in st.session_state.backtest_logs if log['level'] == 'SUCCESS'])

            st.markdown(f"**Statistics:** {total_logs} total | {success_count} success | {error_count} errors")

    def get_default_strategy_code(self):
        """Get default strategy code"""
        return '''def strategy(data, position, cash, portfolio_value):
    """
    Simple Moving Average Strategy

    Parameters:
    - data: Price data (DataFrame with OHLCV)
    - position: Current position (-1: short, 0: neutral, 1: long)
    - cash: Available cash
    - portfolio_value: Total portfolio value

    Returns:
    - 1: Buy signal
    - -1: Sell signal
    - 0: Hold
    """

    # Ensure we have enough data
    if len(data) < 20:
        return 0

    # Calculate moving averages
    short_ma = data['close'].rolling(window=5).mean().iloc[-1]
    long_ma = data['close'].rolling(window=20).mean().iloc[-1]
    current_price = data['close'].iloc[-1]

    # Generate signals
    if short_ma > long_ma and position <= 0:
        return 1  # Buy signal
    elif short_ma < long_ma and position >= 0:
        return -1  # Sell signal
    else:
        return 0  # Hold
'''

    def get_momentum_template(self):
        """Get momentum strategy template"""
        return '''def strategy(data, position, cash, portfolio_value):
    """
    Momentum Strategy Template
    Buy when price breaks above recent high, sell when below recent low
    """

    if len(data) < 20:
        return 0

    # Calculate price momentum indicators
    current_price = data['close'].iloc[-1]
    highest_20 = data['high'].rolling(window=20).max().iloc[-1]
    lowest_20 = data['low'].rolling(window=20).min().iloc[-1]

    # RSI calculation (simplified)
    price_changes = data['close'].diff()
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)
    avg_gains = gains.rolling(window=14).mean().iloc[-1]
    avg_losses = losses.rolling(window=14).mean().iloc[-1]

    if avg_losses != 0:
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 100

    # Generate signals
    if current_price > highest_20 * 0.98 and rsi < 70 and position <= 0:
        return 1  # Buy on momentum breakout
    elif current_price < lowest_20 * 1.02 and position >= 0:
        return -1  # Sell on momentum breakdown
    else:
        return 0  # Hold
'''

    def get_value_template(self):
        """Get value strategy template"""
        return '''def strategy(data, position, cash, portfolio_value):
    """
    Value Strategy Template
    Buy when price is below moving average, sell when above
    """

    if len(data) < 50:
        return 0

    # Calculate value indicators
    current_price = data['close'].iloc[-1]
    ma_50 = data['close'].rolling(window=50).mean().iloc[-1]
    ma_200 = data['close'].rolling(window=200).mean().iloc[-1]

    # Price relative to moving averages
    price_vs_ma50 = (current_price - ma_50) / ma_50
    price_vs_ma200 = (current_price - ma_200) / ma_200

    # Volume confirmation
    current_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 1
    avg_volume = data['volume'].rolling(window=20).mean().iloc[-1] if 'volume' in data.columns else 1
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

    # Generate signals
    if price_vs_ma50 < -0.05 and price_vs_ma200 < -0.1 and volume_ratio > 1.2 and position <= 0:
        return 1  # Buy when undervalued with volume confirmation
    elif price_vs_ma50 > 0.05 and position >= 0:
        return -1  # Sell when overvalued
    else:
        return 0  # Hold
'''

    def add_log(self, message, level="INFO"):
        """Add log entry"""
        log_entry = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }

        # Add to logs
        st.session_state.backtest_logs.append(log_entry)

        # Keep only recent logs (max 100)
        if len(st.session_state.backtest_logs) > 100:
            st.session_state.backtest_logs = st.session_state.backtest_logs[-100:]

    def filter_logs(self, logs, level):
        """Filter logs by level"""
        if level == "ALL":
            return logs
        else:
            return [log for log in logs if log['level'] == level]

    def save_strategy_code(self):
        """Save strategy code"""
        try:
            # Simple validation
            compile(st.session_state.strategy_code, '<string>', 'exec')

            # Save to session state (in a real app, you might save to file/database)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_name = f"strategy_{timestamp}"

            if 'saved_strategies' not in st.session_state:
                st.session_state.saved_strategies = {}

            st.session_state.saved_strategies[strategy_name] = {
                'code': st.session_state.strategy_code,
                'timestamp': timestamp,
                'description': 'User saved strategy'
            }

            self.add_log(f"✅ Strategy saved as {strategy_name}", "SUCCESS")

        except SyntaxError as e:
            self.add_log(f"❌ Cannot save: Syntax error - {e}", "ERROR")
        except Exception as e:
            self.add_log(f"❌ Save failed: {e}", "ERROR")

    def run_strategy_backtest(self, dataset_name):
        """Run strategy backtest"""
        try:
            self.add_log(f"🚀 Starting backtest on {dataset_name}", "INFO")

            # Get data
            if dataset_name not in st.session_state.user_data:
                raise ValueError(f"Dataset {dataset_name} not found")

            df = st.session_state.user_data[dataset_name]

            # Execute strategy code
            exec(st.session_state.strategy_code, globals())

            # Simulate backtest (simplified)
            initial_capital = 100000
            portfolio_value = initial_capital
            position = 0
            cash = initial_capital

            results = []

            for i in range(len(df)):
                if i < 20:  # Skip first 20 rows for indicators
                    continue

                # Get current data slice
                current_data = df.iloc[:i+1]

                # Execute strategy
                signal = strategy(current_data, position, cash, portfolio_value)

                # Process signal (simplified)
                current_price = df.iloc[i]['close'] if 'close' in df.columns else df.iloc[i].iloc[0]

                if signal == 1 and position == 0:  # Buy
                    shares = cash / current_price
                    position = shares
                    cash = 0
                elif signal == -1 and position > 0:  # Sell
                    cash = position * current_price
                    position = 0

                # Calculate portfolio value
                if position > 0:
                    portfolio_value = position * current_price
                else:
                    portfolio_value = cash

                results.append({
                    'date': df.index[i] if hasattr(df.index, 'strftime') else i,
                    'portfolio_value': portfolio_value,
                    'position': position,
                    'cash': cash,
                    'signal': signal
                })

            # Create results DataFrame
            results_df = pd.DataFrame(results)

            # Calculate performance metrics
            total_return = (portfolio_value - initial_capital) / initial_capital
            sharpe_ratio = self.calculate_sharpe_ratio(results_df)

            backtest_result = {
                'initial_capital': initial_capital,
                'final_value': portfolio_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self.calculate_max_drawdown(results_df),
                'portfolio_history': results_df,
                'strategy_code': st.session_state.strategy_code,
                'dataset_name': dataset_name,
                'backtest_date': datetime.now()
            }

            st.session_state.current_backtest_result = backtest_result
            self.add_log("✅ Backtest completed successfully!", "SUCCESS")

            return backtest_result

        except Exception as e:
            self.add_log(f"❌ Backtest failed: {e}", "ERROR")
            return None

    def calculate_sharpe_ratio(self, results_df):
        """Calculate Sharpe ratio"""
        try:
            returns = results_df['portfolio_value'].pct_change().dropna()
            if len(returns) == 0:
                return 0

            mean_return = returns.mean()
            std_return = returns.std()

            if std_return == 0:
                return 0

            # Annualize (assume daily data)
            sharpe = (mean_return / std_return) * np.sqrt(252)
            return sharpe

        except Exception:
            return 0

    def calculate_max_drawdown(self, results_df):
        """Calculate maximum drawdown"""
        try:
            portfolio_values = results_df['portfolio_value']
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak
            return drawdown.min()
        except Exception:
            return 0

    def get_days_from_period(self, period):
        """Convert period string to number of days"""
        period_map = {
            "5 days": 5,
            "10 days": 10,
            "20 days": 20,
            "30 days": 30,
            "60 days": 60,
            "1 month": 30,
            "3 months": 90,
            "6 months": 180,
            "1 year": 252,
            "2 years": 504
        }
        return period_map.get(period, 20)

    def simulate_backtest_result(self, df, strategy_type, lookback_days, initial_capital):
        """Simulate backtest result for quick testing"""
        try:
            # Generate simulated performance based on strategy type
            n_days = min(len(df), lookback_days)

            # Different performance characteristics for different strategies
            if strategy_type == "Momentum Strategy":
                base_return = 0.0008
                volatility = 0.025
            elif strategy_type == "Mean Reversion":
                base_return = 0.0005
                volatility = 0.015
            elif strategy_type == "Buy and Hold":
                base_return = 0.0004
                volatility = 0.018
            else:
                base_return = 0.0006
                volatility = 0.020

            # Generate portfolio history
            dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
            daily_returns = np.random.normal(base_return, volatility, n_days)
            portfolio_values = initial_capital * np.cumprod(1 + daily_returns)

            # Create portfolio history DataFrame
            portfolio_history = pd.DataFrame({
                'portfolio_value': portfolio_values,
                'daily_return': daily_returns,
                'date': dates
            }, index=dates)

            # Add benchmark (simple market return)
            benchmark_returns = np.random.normal(0.0003, 0.015, n_days)
            benchmark_values = initial_capital * np.cumprod(1 + benchmark_returns)
            portfolio_history['benchmark_value'] = benchmark_values

            # Calculate metrics
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_capital) / initial_capital

            # Calculate Sharpe ratio
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

            # Calculate max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)

            result = {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'portfolio_history': portfolio_history,
                'strategy_type': strategy_type,
                'n_days': n_days
            }

            return result

        except Exception as e:
            self.add_log(f"❌ Simulation failed: {e}", "ERROR")
            return None

    def render_deep_analysis(self):
        """Render deep analysis interface"""
        st.markdown("### 🔍 Deep Analysis")
        st.markdown("*Comprehensive backtesting analysis and diagnostics*")

        # Back button
        if st.button("← Back to Basic View", key="back_to_basic"):
            st.session_state.show_deep_analysis = False
            st.rerun()

        result = st.session_state.current_backtest_result

        if not result:
            st.error("No backtest result available for analysis")
            return

        # Performance overview
        st.markdown("## 📊 Performance Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Return",
                f"{result['total_return']:.2%}",
                help="Total return over the backtesting period"
            )

        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{result['sharpe_ratio']:.3f}",
                help="Risk-adjusted return measure"
            )

        with col3:
            st.metric(
                "Max Drawdown",
                f"{result['max_drawdown']:.2%}",
                help="Maximum decline from peak value"
            )

        with col4:
            st.metric(
                "Final Value",
                f"${result['final_value']:,.0f}",
                f"${result['final_value'] - result['initial_capital']:,.0f}"
            )

        # Detailed charts
        st.markdown("## 📈 Detailed Analysis")

        portfolio_data = result['portfolio_history']

        # Portfolio value and drawdown
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Portfolio Value")
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))

            if 'benchmark_value' in portfolio_data.columns:
                fig.add_trace(go.Scatter(
                    x=portfolio_data.index,
                    y=portfolio_data['benchmark_value'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#ff7f0e', width=1, dash='dash')
                ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Value ($)",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Drawdown Analysis")

            # Calculate drawdown
            portfolio_values = portfolio_data['portfolio_value']
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=drawdown * 100,
                mode='lines',
                name='Drawdown %',
                fill='tonexty',
                line=dict(color='red', width=1)
            ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        # Returns analysis
        st.markdown("### 📊 Returns Analysis")

        if 'daily_return' in portfolio_data.columns:
            returns = portfolio_data['daily_return'].dropna()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Return Distribution**")
                fig = px.histogram(
                    x=returns * 100,
                    nbins=30,
                    title="Daily Returns Distribution (%)"
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Return Statistics**")
                stats_data = {
                    'Metric': [
                        'Mean Daily Return',
                        'Std Daily Return',
                        'Skewness',
                        'Kurtosis',
                        'Best Day',
                        'Worst Day'
                    ],
                    'Value': [
                        f"{returns.mean():.4f}",
                        f"{returns.std():.4f}",
                        f"{returns.skew():.3f}",
                        f"{returns.kurtosis():.3f}",
                        f"{returns.max():.4f}",
                        f"{returns.min():.4f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        # Strategy information
        st.markdown("### 💻 Strategy Information")

        with st.expander("Strategy Code", expanded=False):
            st.code(result.get('strategy_code', 'No strategy code available'), language='python')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Backtest Configuration**")
            config_data = {
                'Parameter': [
                    'Strategy Type',
                    'Dataset',
                    'Initial Capital',
                    'Backtest Date',
                    'Number of Days'
                ],
                'Value': [
                    result.get('strategy_type', 'Unknown'),
                    result.get('dataset_name', 'Unknown'),
                    f"${result['initial_capital']:,.0f}",
                    result.get('backtest_date', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                    str(result.get('n_days', 'Unknown'))
                ]
            }
            st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**Performance Summary**")
            perf_data = {
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Volatility',
                    'Sharpe Ratio',
                    'Max Drawdown'
                ],
                'Value': [
                    f"{result['total_return']:.2%}",
                    f"{(1 + result['total_return']) ** (252 / result.get('n_days', 252)) - 1:.2%}",
                    f"{portfolio_data['daily_return'].std() * np.sqrt(252):.2%}",
                    f"{result['sharpe_ratio']:.3f}",
                    f"{result['max_drawdown']:.2%}"
                ]
            }
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
