import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import numpy as np

# --- Crypto Data Fetcher Classes ---
class CryptoDataFetcher:
    def __init__(self, symbols, start_time, end_time):
        self.symbols = symbols
        self.start_time = start_time
        self.end_time = end_time
        self.base_url = "https://api.binance.com"

    def fetch_klines(self, symbol, start_time):
        endpoint = f"/api/v3/klines?symbol={symbol}&interval=1d&limit=1000&startTime={start_time}"
        url = self.base_url + endpoint
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None

    def make_request(self):
        df = pd.DataFrame()
        for symbol in self.symbols:
            symbol_klines = []
            current_time = self.start_time

            while current_time < self.end_time:
                klines = self.fetch_klines(symbol, current_time)
                if not klines:
                    st.warning(f"No data received for timestamp {current_time} for {symbol}")
                    break

                filtered_klines = [kline for kline in klines if int(kline[0]) <= self.end_time]
                if not filtered_klines:
                    break

                symbol_klines.extend(filtered_klines)
                current_time = int(filtered_klines[-1][0]) + 1
                time.sleep(0.1)

            symbol_klines = np.array(symbol_klines)

            if df.empty:
                df = pd.DataFrame()
                df['Timestamp'] = symbol_klines[:, 0]
                df[f'{symbol}'] = symbol_klines[:, 4]
                df[f'{symbol}'] = df[f'{symbol}'].apply(pd.to_numeric).pct_change()
            else:
                df[f'{symbol}'] = symbol_klines[:, 4]
                df[f'{symbol}'] = df[f'{symbol}'].apply(pd.to_numeric).pct_change()

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        df = df.dropna()
        return df


class PortfolioAllocator:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.returns = self.data_fetcher.make_request()

    def calculate_portfolio(self, weights):
        weights = np.array(weights)
        if len(weights) != len(self.returns.columns):
            raise ValueError("Weights length must match number of assets")
        if not np.isclose(np.sum(weights), 1):
            raise ValueError("Weights must sum to 1")

        portfolio_returns = self.returns.dot(weights)
        return portfolio_returns

    def calculate_individual_returns(self):
        return self.returns


class CryptoVaRCalculator:
    def __init__(self, portfolio_returns, individual_returns):
        self.returns = portfolio_returns
        self.ind_returns = individual_returns

    def historical_var(self, alpha):
        returns = self.returns
        if isinstance(returns, pd.Series):
            return np.percentile(returns, alpha)
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(self.historical_var, alpha=alpha)
        else:
            raise TypeError("Expected returns to be dataframe or series")

    def historical_cvar(self, alpha):
        returns = self.returns
        if isinstance(returns, pd.Series):
            belowVaR = returns <= self.historical_var(alpha)
            return returns[belowVaR].mean()
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(self.historical_cvar, alpha)
        else:
            raise TypeError("Expected returns to be dataframe or series")

    def eff_frontier(self, num_portfolios=10000, risk_free_rate=0.04):
        r = self.ind_returns
        weights = np.random.dirichlet(np.ones(len(r.columns)), size=num_portfolios)
        assert np.allclose(np.sum(weights, axis=1), 1)

        eff_front_dict = {}
        cov_matrix_ret = r.cov()
        expected_returns = r.mean()

        for w in weights:
            port_ret = expected_returns @ w.T
            port_std = np.sqrt(w.T @ cov_matrix_ret @ w)
            sharpe_ratio = (port_ret - risk_free_rate) / port_std if port_std != 0 else 0
            eff_front_dict[str(list(w))] = [port_ret, port_std, sharpe_ratio]

        eff_frontier_dataframe = pd.DataFrame(eff_front_dict, index=['Returns', 'Standard Deviation', 'Sharpe Ratio']).T
        highest_sharpe_ratio = eff_frontier_dataframe.sort_values(by='Sharpe Ratio', ascending=False)
        return highest_sharpe_ratio


def parse_date_to_timestamp_ms(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

# Initialize session state
if 'calculated' not in st.session_state:
    st.session_state.calculated = False

# Streamlit app
st.title("Crypto Portfolio Efficient Frontier and VaR Dashboard")

# Input widgets
symbols_input = st.text_input("Enter symbols separated by commas (e.g. BTCUSDT,ETHUSDT,SOLUSDT):", "BTCUSDT,ETHUSDT,SOLUSDT")
start_date_str = st.text_input("Enter start date (YYYY-MM-DD):", "2024-01-01")
end_date_str = st.text_input("Enter end date (YYYY-MM-DD):", "2024-12-31")
risk_free_rate = st.number_input("Enter risk-free rate (e.g. 0.04 for 4%):", min_value=0.0, max_value=1.0, value=0.04, step=0.001, key='risk_free_rate')
alpha = st.number_input("Enter VaR confidence level in percent (e.g. 5 for 5%):", min_value=0.1, max_value=100.0, value=5.0, step=0.1, key='alpha')

# Centered button layout
col1, col2, _ = st.columns([1.5, 1, 1])  # Adjust column widths as needed

with col2:
    calculate_button = st.button("Calculate")

# Perform calculations and display results
if calculate_button:
    try:
        SYMBOLS = [sym.strip().upper() for sym in symbols_input.split(",")]
        START_TIME = parse_date_to_timestamp_ms(start_date_str)
        END_TIME = parse_date_to_timestamp_ms(end_date_str)

        data_fetcher = CryptoDataFetcher(SYMBOLS, START_TIME, END_TIME)
        portfolio_allocator = PortfolioAllocator(data_fetcher)
        individual_returns = portfolio_allocator.calculate_individual_returns()

        var_calculator = CryptoVaRCalculator(None, individual_returns)
        eff_frontier_df = var_calculator.eff_frontier(num_portfolios=10000, risk_free_rate=risk_free_rate)

        # Extract best portfolio info first
        best_portfolio = eff_frontier_df.iloc[0]
        best_weights_str = eff_frontier_df.index[0]
        best_weights = [float(w.strip()) for w in best_weights_str.strip('[]').split(',')]

        # Show best weights
        best_weights_percent = [round(w * 100, 4) for w in best_weights]
        asset_columns = individual_returns.columns.tolist()
        formatted_weights = [f"{asset}: {weight}%" for asset, weight in zip(asset_columns, best_weights_percent)]

        st.subheader(f"Most Efficient Weights for Highest Sharpe Ratio ({best_portfolio['Sharpe Ratio']:.4f}):")
        st.write(", ".join(formatted_weights))

        # Efficient Frontier Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(eff_frontier_df['Standard Deviation'], eff_frontier_df['Returns'], 
                             c=eff_frontier_df['Sharpe Ratio'], cmap='viridis', marker='o')
        ax.set_xlabel('Standard Deviation')
        ax.set_ylabel('Returns')
        ax.set_title('Efficient Frontier')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio')
        st.pyplot(fig)

        # Calculate portfolio returns and VaR/CVaR
        portfolio_returns = portfolio_allocator.calculate_portfolio(best_weights)
        var_calculator.returns = portfolio_returns

        time_horizon = 1
        hVaR = -var_calculator.historical_var(alpha) * np.sqrt(time_horizon)
        historical_cvar = -var_calculator.historical_cvar(alpha) * np.sqrt(time_horizon)

        # Display VaR value and histogram
        st.subheader(f"Historical VaR ({alpha}%)")
        st.write(f"{hVaR:.4f}")

        fig_var, ax_var = plt.subplots()
        ax_var.hist(portfolio_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        var_threshold = np.percentile(portfolio_returns, alpha)
        ax_var.axvline(x=var_threshold, color='red', linestyle='--', linewidth=2, label=f'VaR ({alpha}%) = {var_threshold:.4f}')
        ax_var.set_title('Portfolio Returns Histogram with VaR')
        ax_var.set_xlabel('Returns')
        ax_var.set_ylabel('Frequency')
        ax_var.legend()
        st.pyplot(fig_var)

        # Display CVaR value and histogram
        st.subheader(f"Historical CVaR ({alpha}%)")
        st.write(f"{historical_cvar:.4f}")

        fig_cvar, ax_cvar = plt.subplots()
        counts, bins, patches = ax_cvar.hist(portfolio_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')

        var_threshold = np.percentile(portfolio_returns, alpha)
        cvar_value = var_calculator.historical_cvar(alpha)

        # Highlight bars below VaR threshold in red
        for patch, bin_left in zip(patches, bins):
            if bin_left <= var_threshold:
                patch.set_facecolor('red')
                patch.set_alpha(0.4)

        # Draw vertical line for CVaR value
        ax_cvar.axvline(x=cvar_value, color='darkred', linestyle='-', linewidth=2, label=f'CVaR ({alpha}%) = {cvar_value:.4f}')

        ax_cvar.set_title('Portfolio Returns Histogram with CVaR')
        ax_cvar.set_xlabel('Returns')
        ax_cvar.set_ylabel('Frequency')
        ax_cvar.legend()
        st.pyplot(fig_cvar)

        st.session_state.calculated = True

    except Exception as e:
        st.error(f"Error during calculation: {e}")

# Show reset button only after calculation is done
# Show reset button only after calculation is done at the bottom of the page
if st.session_state.calculated:
    # Create a new set of columns at the end of the page
    st.write("")  # Add some space
    st.write("")  # Add more space
    reset_col1, reset_col2, reset_col3 = st.columns([1.5, 1, 1])
    
    with reset_col2:
        if st.button("Reset"):
            st.session_state.calculated = False
            st.rerun()



