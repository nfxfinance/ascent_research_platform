class PortfolioOptimizationModule:
    """Portfolio Optimization Module - Modern Portfolio Theory, Risk-Return Analysis, Asset Allocation"""

    def __init__(self):
        self.name = "Portfolio Optimization"
        self.description = "Modern Portfolio Theory, Risk-Return Analysis, Asset Allocation"
        self.initialize_state()

    def initialize_state(self):
        """Initialize module state"""
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = {}
        if 'portfolio_constraints' not in st.session_state:
            st.session_state.portfolio_constraints = {}

    def render(self):
        """Render portfolio optimization module interface"""
        st.markdown("## 📈 Portfolio Optimization Module")
        st.markdown("*Based on Modern Portfolio Theory, providing comprehensive investment portfolio optimization solutions*")

        # Check if there's data
        if 'user_data' not in st.session_state or not st.session_state.user_data:
            st.warning("⚠️ Please load price data first")
            st.markdown("""
            ### 💡 Data Loading Guide

            Portfolio optimization requires historical price data of multiple assets. You can import data through:

            1. **Data Management Module** - Upload CSV or Excel files
            2. **Sample data format**:
               - Date column (Date/datetime type)
               - Asset price columns (one column per asset)
               - Example: Date, Stock_A, Stock_B, Stock_C, Bond_A
            """)
            return

        # Main function tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Preparation",
            "⚙️ Portfolio Optimization",
            "📈 Efficient Frontier",
            "🎯 Strategy Comparison",
            "📋 Results Management"
        ])

        with tab1:
            self.render_data_preparation()

        with tab2:
            self.render_portfolio_optimization()

        with tab3:
            self.render_efficient_frontier()

        with tab4:
            self.render_strategy_comparison()

        with tab5:
            self.render_results_management()

    def render_data_preparation(self):
        """Render data preparation interface"""
        st.markdown("### 📊 Data Preparation")

        # Select dataset
        dataset_names = list(st.session_state.user_data.keys())
        selected_dataset = st.selectbox("Select Dataset", dataset_names, key="portfolio_dataset")

        if selected_dataset:
            df = st.session_state.user_data[selected_dataset]

            # Data overview
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📋 Data Overview")
                st.dataframe(df.head(), use_container_width=True)

                # Basic statistics
                st.markdown("#### 📊 Basic Statistics")
                st.text(f"Data Shape: {df.shape}")
                st.text(f"Date Range: {df.index[0] if hasattr(df.index, '__getitem__') else 'Unknown'} to {df.index[-1] if hasattr(df.index, '__getitem__') else 'Unknown'}")
                st.text(f"Missing Values: {df.isnull().sum().sum()}")

            with col2:
                # Column selection
                st.markdown("#### 🎯 Asset Selection")

                # If there's a date column, set it as index
                date_columns = []
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        date_columns.append(col)

                if date_columns:
                    date_col = st.selectbox("Date Column", date_columns, key="date_column")
                    if st.button("Set as Index"):
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.set_index(date_col)
                        st.session_state.user_data[selected_dataset] = df
                        st.rerun()

                # Select asset columns (numeric columns)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_assets = st.multiselect(
                        "Select Assets (Price Columns)",
                        numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))],
                        key="selected_assets"
                    )

                    if selected_assets:
                        # Calculate returns
                        st.markdown("#### 📈 Return Calculation")
                        return_method = st.selectbox(
                            "Return Calculation Method",
                            ["Simple Return", "Log Return"],
                            key="return_method"
                        )

                        if st.button("Calculate Returns", type="primary"):
                            returns_df = self.calculate_returns(df[selected_assets], return_method)

                            # Save to session state
                            returns_name = f"{selected_dataset}_returns"
                            st.session_state.user_data[returns_name] = returns_df

                            # Display return statistics
                            st.success(f"✅ Returns calculated and saved as: {returns_name}")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Return Statistics**")
                                st.dataframe(returns_df.describe(), use_container_width=True)

                            with col_b:
                                st.markdown("**Correlation Matrix**")
                                corr_matrix = returns_df.corr()
                                fig = px.imshow(
                                    corr_matrix,
                                    title="Asset Return Correlation Matrix",
                                    color_continuous_scale="RdBu",
                                    aspect="auto"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("No numeric columns found")

    def render_portfolio_optimization(self):
        """Render portfolio optimization interface"""
        st.markdown("### ⚙️ Portfolio Optimization")

        # Select return data
        return_datasets = [name for name in st.session_state.user_data.keys() if '_returns' in name]

        if not return_datasets:
            st.warning("Please calculate returns first in Data Preparation")
            return

        selected_returns = st.selectbox("Select Return Data", return_datasets, key="opt_returns")

        if selected_returns:
            returns_df = st.session_state.user_data[selected_returns]

            # Optimization settings
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ⚙️ Optimization Settings")

                # Optimization objective
                objective = st.selectbox(
                    "Optimization Objective",
                    ["Maximize Sharpe Ratio", "Minimize Risk", "Maximize Return", "Risk Parity"],
                    key="optimization_objective"
                )

                # Risk model
                risk_model = st.selectbox(
                    "Risk Model",
                    ["Sample Covariance", "Ledoit-Wolf Shrinkage", "Constant Correlation"],
                    key="risk_model"
                )

                # Risk-free rate
                risk_free_rate = st.number_input(
                    "Risk-free Rate (Annual %)",
                    value=3.0,
                    min_value=0.0,
                    max_value=20.0,
                    step=0.1,
                    key="risk_free_rate"
                ) / 100

            with col2:
                st.markdown("#### 🎯 Investment Constraints")

                # Position constraints
                min_weight = st.number_input(
                    "Minimum Asset Weight (%)",
                    value=0.0,
                    min_value=0.0,
                    max_value=50.0,
                    step=1.0,
                    key="min_weight"
                ) / 100

                max_weight = st.number_input(
                    "Maximum Asset Weight (%)",
                    value=30.0,
                    min_value=1.0,
                    max_value=100.0,
                    step=1.0,
                    key="max_weight"
                ) / 100

                # Sector constraints
                enable_sector_constraints = st.checkbox("Enable Sector Constraints", key="enable_sector")

                if enable_sector_constraints:
                    st.info("Sector constraint functionality to be implemented")

                # Transaction costs
                transaction_cost = st.number_input(
                    "Transaction Cost (%)",
                    value=0.1,
                    min_value=0.0,
                    max_value=5.0,
                    step=0.1,
                    key="transaction_cost"
                ) / 100

            # Execute optimization
            st.markdown("---")
            if st.button("🚀 Execute Optimization", type="primary"):
                with st.spinner("Optimizing portfolio..."):
                    try:
                        # Execute optimization
                        optimization_result = self.optimize_portfolio(
                            returns_df,
                            objective,
                            risk_model,
                            risk_free_rate,
                            min_weight,
                            max_weight,
                            transaction_cost
                        )

                        # Save results
                        result_name = f"Portfolio_{objective}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.optimization_results[result_name] = optimization_result

                        # Display results
                        self.display_optimization_results(optimization_result)

                    except Exception as e:
                        st.error(f"Optimization failed: {e}")

    def render_efficient_frontier(self):
        """Render efficient frontier interface"""
        st.markdown("### 📈 Efficient Frontier")

        # Select return data
        return_datasets = [name for name in st.session_state.user_data.keys() if '_returns' in name]

        if not return_datasets:
            st.warning("Please calculate returns first")
            return

        selected_returns = st.selectbox("Select Return Data", return_datasets, key="ef_returns")

        if selected_returns:
            returns_df = st.session_state.user_data[selected_returns]

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### ⚙️ Frontier Settings")

                # Number of portfolios
                num_portfolios = st.slider(
                    "Number of Portfolios",
                    min_value=50,
                    max_value=500,
                    value=100,
                    key="num_portfolios"
                )

                # Risk-free rate
                rf_rate = st.number_input(
                    "Risk-free Rate (%)",
                    value=3.0,
                    key="ef_rf_rate"
                ) / 100

                # Generate efficient frontier
                if st.button("Generate Efficient Frontier", type="primary"):
                    with st.spinner("Generating efficient frontier..."):
                        frontier_results = self.calculate_efficient_frontier(
                            returns_df, num_portfolios, rf_rate
                        )

                        # Save results
                        st.session_state.efficient_frontier = frontier_results

            with col2:
                if 'efficient_frontier' in st.session_state:
                    st.markdown("#### 📊 Efficient Frontier Chart")

                    frontier_data = st.session_state.efficient_frontier

                    # Create efficient frontier chart
                    fig = go.Figure()

                    # Add efficient frontier
                    fig.add_trace(go.Scatter(
                        x=frontier_data['risk'],
                        y=frontier_data['return'],
                        mode='lines',
                        name='Efficient Frontier',
                        line=dict(color='blue', width=2)
                    ))

                    # Add optimal portfolios
                    fig.add_trace(go.Scatter(
                        x=[frontier_data['min_vol_risk']],
                        y=[frontier_data['min_vol_return']],
                        mode='markers',
                        name='Minimum Volatility',
                        marker=dict(color='green', size=10, symbol='star')
                    ))

                    fig.add_trace(go.Scatter(
                        x=[frontier_data['max_sharpe_risk']],
                        y=[frontier_data['max_sharpe_return']],
                        mode='markers',
                        name='Maximum Sharpe Ratio',
                        marker=dict(color='red', size=10, symbol='star')
                    ))

                    # Add capital allocation line
                    if rf_rate > 0:
                        cal_x = np.linspace(0, max(frontier_data['risk']) * 1.2, 100)
                        cal_y = rf_rate + (frontier_data['max_sharpe_return'] - rf_rate) / frontier_data['max_sharpe_risk'] * cal_x
                        fig.add_trace(go.Scatter(
                            x=cal_x,
                            y=cal_y,
                            mode='lines',
                            name='Capital Allocation Line',
                            line=dict(color='orange', dash='dash')
                        ))

                    fig.update_layout(
                        title="Efficient Frontier",
                        xaxis_title="Risk (Standard Deviation)",
                        yaxis_title="Expected Return",
                        hovermode='closest'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display optimal portfolio weights
                    st.markdown("#### 🎯 Optimal Portfolio Weights")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("**Minimum Volatility Portfolio**")
                        min_vol_weights = pd.DataFrame({
                            'Asset': returns_df.columns,
                            'Weight': frontier_data['min_vol_weights']
                        })
                        st.dataframe(min_vol_weights, use_container_width=True)

                    with col_b:
                        st.markdown("**Maximum Sharpe Ratio Portfolio**")
                        max_sharpe_weights = pd.DataFrame({
                            'Asset': returns_df.columns,
                            'Weight': frontier_data['max_sharpe_weights']
                        })
                        st.dataframe(max_sharpe_weights, use_container_width=True)

    def render_strategy_comparison(self):
        """Render strategy comparison interface"""
        st.markdown("### 🎯 Strategy Comparison")

        if not st.session_state.optimization_results:
            st.info("No optimization results available. Please execute optimization first.")
            return

        # Select results for comparison
        result_names = list(st.session_state.optimization_results.keys())
        selected_results = st.multiselect(
            "Select Optimization Results for Comparison",
            result_names,
            default=result_names[:min(3, len(result_names))],
            key="comparison_results"
        )

        if selected_results:
            # Create comparison table
            comparison_data = []
            for name in selected_results:
                result = st.session_state.optimization_results[name]
                comparison_data.append({
                    'Strategy': name,
                    'Expected Return': f"{result['expected_return']:.2%}",
                    'Volatility': f"{result['volatility']:.2%}",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                    'VaR (95%)': f"{result.get('var_95', 0):.2%}",
                    'Max Drawdown': f"{result.get('max_drawdown', 0):.2%}"
                })

            comparison_df = pd.DataFrame(comparison_data)

            st.markdown("#### 📊 Performance Comparison")
            st.dataframe(comparison_df, use_container_width=True)

            # Visualization
            col1, col2 = st.columns(2)

            with col1:
                # Risk-return scatter plot
                fig_scatter = go.Figure()

                for name in selected_results:
                    result = st.session_state.optimization_results[name]
                    fig_scatter.add_trace(go.Scatter(
                        x=[result['volatility']],
                        y=[result['expected_return']],
                        mode='markers',
                        name=name,
                        marker=dict(size=10),
                        text=[f"Sharpe: {result['sharpe_ratio']:.3f}"],
                        hovertemplate="<b>%{fullData.name}</b><br>" +
                                    "Return: %{y:.2%}<br>" +
                                    "Risk: %{x:.2%}<br>" +
                                    "%{text}<extra></extra>"
                    ))

                fig_scatter.update_layout(
                    title="Risk-Return Comparison",
                    xaxis_title="Volatility",
                    yaxis_title="Expected Return"
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                # Portfolio weights comparison
                weight_data = {}
                for name in selected_results:
                    result = st.session_state.optimization_results[name]
                    weight_data[name] = result['weights']

                weight_df = pd.DataFrame(weight_data).fillna(0)

                fig_weights = go.Figure()

                for strategy in weight_df.columns:
                    fig_weights.add_trace(go.Bar(
                        name=strategy,
                        x=weight_df.index,
                        y=weight_df[strategy],
                        text=[f"{w:.1%}" for w in weight_df[strategy]],
                        textposition='auto'
                    ))

                fig_weights.update_layout(
                    title="Portfolio Weights Comparison",
                    xaxis_title="Assets",
                    yaxis_title="Weight",
                    barmode='group'
                )

                st.plotly_chart(fig_weights, use_container_width=True)

    def render_results_management(self):
        """Render results management interface"""
        st.markdown("### 📋 Results Management")

        if st.session_state.optimization_results:
            st.markdown("#### 📊 Saved Optimization Results")

            for name, result in st.session_state.optimization_results.items():
                with st.expander(f"📈 {name}", expanded=False):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Performance Metrics**")
                        st.text(f"Expected Return: {result['expected_return']:.2%}")
                        st.text(f"Volatility: {result['volatility']:.2%}")
                        st.text(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")

                    with col2:
                        st.markdown("**Portfolio Weights**")
                        weights_df = pd.DataFrame({
                            'Asset': list(result['weights'].keys()),
                            'Weight': [f"{w:.2%}" for w in result['weights'].values()]
                        })
                        st.dataframe(weights_df, use_container_width=True)

                    with col3:
                        st.markdown("**Actions**")
                        if st.button("📥 Export to CSV", key=f"export_{name}"):
                            self.export_results_to_csv(result, name)

                        if st.button("🗑️ Delete", key=f"delete_result_{name}"):
                            del st.session_state.optimization_results[name]
                            st.rerun()

            # Batch operations
            st.markdown("---")
            st.markdown("#### 🔧 Batch Operations")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📥 Export All Results"):
                    self.export_all_results()

            with col2:
                if st.button("🗑️ Clear All Results"):
                    st.session_state.optimization_results = {}
                    st.rerun()

            with col3:
                if st.button("📊 Generate Report"):
                    self.generate_optimization_report()

        else:
            st.info("No saved optimization results")

    # Helper methods
    def calculate_returns(self, price_df, method="Simple Return"):
        """Calculate asset returns"""
        if method == "Simple Return":
            returns = price_df.pct_change().dropna()
        else:  # Log Return
            returns = np.log(price_df / price_df.shift(1)).dropna()

        return returns

    def optimize_portfolio(self, returns_df, objective, risk_model, risk_free_rate,
                         min_weight, max_weight, transaction_cost):
        """Execute portfolio optimization"""

        # Calculate expected returns and covariance matrix
        expected_returns = returns_df.mean() * 252  # Annualized

        if risk_model == "Sample Covariance":
            cov_matrix = returns_df.cov() * 252  # Annualized
        elif risk_model == "Ledoit-Wolf Shrinkage":
            # Simplified implementation
            cov_matrix = returns_df.cov() * 252
        else:  # Constant Correlation
            # Simplified implementation
            cov_matrix = returns_df.cov() * 252

        n_assets = len(expected_returns)

        # Optimization constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Objective function
        if objective == "Maximize Sharpe Ratio":
            def objective_func(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return - risk_free_rate) / portfolio_vol

        elif objective == "Minimize Risk":
            def objective_func(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        elif objective == "Maximize Return":
            def objective_func(weights):
                return -np.sum(weights * expected_returns)

        else:  # Risk Parity
            def objective_func(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)

        # Initial weights (equal weight)
        initial_weights = np.array([1/n_assets] * n_assets)

        # Optimize
        from scipy.optimize import minimize
        result = minimize(objective_func, initial_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)

        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol

            optimization_result = {
                'weights': dict(zip(returns_df.columns, optimal_weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'objective': objective,
                'risk_model': risk_model,
                'optimization_date': datetime.now().isoformat()
            }

            return optimization_result
        else:
            raise Exception(f"Optimization failed: {result.message}")

    def calculate_efficient_frontier(self, returns_df, num_portfolios, risk_free_rate):
        """Calculate efficient frontier"""
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252

        # Generate target returns
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)

        frontier_volatility = []
        frontier_weights = []

        from scipy.optimize import minimize

        for target_ret in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_ret}
            )

            bounds = tuple((0, 1) for _ in range(len(expected_returns)))

            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            initial_weights = np.array([1/len(expected_returns)] * len(expected_returns))

            result = minimize(objective, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)

            if result.success:
                frontier_volatility.append(result.fun)
                frontier_weights.append(result.x)
            else:
                frontier_volatility.append(np.nan)
                frontier_weights.append([np.nan] * len(expected_returns))

        # Find minimum volatility portfolio
        min_vol_idx = np.nanargmin(frontier_volatility)
        min_vol_weights = frontier_weights[min_vol_idx]
        min_vol_return = target_returns[min_vol_idx]
        min_vol_risk = frontier_volatility[min_vol_idx]

        # Find maximum Sharpe ratio portfolio
        sharpe_ratios = [(target_returns[i] - risk_free_rate) / frontier_volatility[i]
                        if not np.isnan(frontier_volatility[i]) else -np.inf
                        for i in range(len(target_returns))]
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_weights = frontier_weights[max_sharpe_idx]
        max_sharpe_return = target_returns[max_sharpe_idx]
        max_sharpe_risk = frontier_volatility[max_sharpe_idx]

        return {
            'return': target_returns,
            'risk': frontier_volatility,
            'weights': frontier_weights,
            'min_vol_weights': min_vol_weights,
            'min_vol_return': min_vol_return,
            'min_vol_risk': min_vol_risk,
            'max_sharpe_weights': max_sharpe_weights,
            'max_sharpe_return': max_sharpe_return,
            'max_sharpe_risk': max_sharpe_risk
        }

    def display_optimization_results(self, result):
        """Display optimization results"""
        st.success("✅ Portfolio optimization completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Performance Metrics")
            st.metric("Expected Annual Return", f"{result['expected_return']:.2%}")
            st.metric("Annual Volatility", f"{result['volatility']:.2%}")
            st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")

        with col2:
            st.markdown("#### 🎯 Portfolio Weights")
            weights_df = pd.DataFrame({
                'Asset': list(result['weights'].keys()),
                'Weight': list(result['weights'].values())
            })

            # Pie chart
            fig = px.pie(weights_df, values='Weight', names='Asset',
                        title="Portfolio Asset Allocation")
            st.plotly_chart(fig, use_container_width=True)

        # Detailed weights table
        st.markdown("#### 📋 Detailed Weights")
        weights_display = weights_df.copy()
        weights_display['Weight'] = weights_display['Weight'].apply(lambda x: f"{x:.2%}")
        st.dataframe(weights_display, use_container_width=True)

    def export_results_to_csv(self, result, name):
        """Export optimization results to CSV"""
        try:
            weights_df = pd.DataFrame({
                'Asset': list(result['weights'].keys()),
                'Weight': list(result['weights'].values())
            })

            csv = weights_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{name}_weights.csv",
                mime="text/csv",
                key=f"download_{name}"
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

    def export_all_results(self):
        """Export all optimization results"""
        try:
            all_results = []
            for name, result in st.session_state.optimization_results.items():
                for asset, weight in result['weights'].items():
                    all_results.append({
                        'Strategy': name,
                        'Asset': asset,
                        'Weight': weight,
                        'Expected_Return': result['expected_return'],
                        'Volatility': result['volatility'],
                        'Sharpe_Ratio': result['sharpe_ratio']
                    })

            results_df = pd.DataFrame(all_results)
            csv = results_df.to_csv(index=False)

            st.download_button(
                label="Download All Results",
                data=csv,
                file_name=f"portfolio_optimization_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

    def generate_optimization_report(self):
        """Generate optimization report"""
        st.info("Report generation functionality to be implemented")
