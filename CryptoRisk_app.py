import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# --- Crypto Data Fetcher Classes ---
class CryptoDataFetcher:
    def __init__(self, symbols, start_time, end_time):
        self.symbols = symbols
        self.start_time = start_time
        self.end_time = end_time
        self.base_url = "https://api.binance.us"
        self.headers = {'User-Agent': 'MyStreamlitApp/1.0'}  # Add User-Agent header

    def fetch_klines(self, symbol, start_time):
        endpoint = f"/api/v3/klines?symbol={symbol}&interval=1d&limit=1000&startTime={start_time}"
        url = self.base_url + endpoint
        retries = 3
        delay = 1  # Initial delay in seconds

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                return response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Attempt {attempt + 1}: Error fetching data for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    st.error(f"Max retries reached for {symbol}. Unable to fetch data.")
                    return None
        return None  # Return None if all retries fail

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

    def eff_frontier(self, num_portfolios=10000, risk_free_rate=0.04, include_cash=False):
        r = self.ind_returns
        num_assets = len(r.columns)

        if include_cash:
            weights = np.random.dirichlet(np.ones(num_assets + 1), size=num_portfolios)  # +1 for cash
            extended_returns = np.append(r.mean(), risk_free_rate)
            extended_cov = np.zeros((num_assets + 1, num_assets + 1))
            extended_cov[:num_assets, :num_assets] = r.cov()
        else:
            weights = np.random.dirichlet(np.ones(num_assets), size=num_portfolios)
            extended_returns = r.mean()
            extended_cov = r.cov()

        eff_front_dict = {}

        for w in weights:
            port_ret = np.dot(w, extended_returns)
            port_std = np.sqrt(np.dot(w.T, np.dot(extended_cov, w)))
            sharpe_ratio = (port_ret - risk_free_rate) / port_std if port_std != 0 else 0
            eff_front_dict[str(list(map(float, w)))] = [port_ret, port_std, sharpe_ratio]

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
available_tickers = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "XLMUSDT", "AVAXUSDT"]
symbols_input = st.multiselect(
    "Select tickers to include in the portfolio:",
    options=available_tickers,
    default=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
)

include_cash = st.radio(
    "Include cash (risk-free asset) in portfolio optimization?",
    options=["No", "Yes"],
    index=0
)

if not symbols_input:
    st.warning("Please select at least one ticker.")
else:
    SYMBOLS = [sym.upper() for sym in symbols_input]

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
        SYMBOLS = [sym.upper() for sym in symbols_input]
        START_TIME = parse_date_to_timestamp_ms(start_date_str)
        END_TIME = parse_date_to_timestamp_ms(end_date_str)

        data_fetcher = CryptoDataFetcher(SYMBOLS, START_TIME, END_TIME)
        portfolio_allocator = PortfolioAllocator(data_fetcher)
        individual_returns = portfolio_allocator.calculate_individual_returns()

        var_calculator = CryptoVaRCalculator(None, individual_returns)
        eff_frontier_df = var_calculator.eff_frontier(num_portfolios=10000, risk_free_rate=risk_free_rate, include_cash=(include_cash == "Yes"))

        # Extract best portfolio info first
        best_portfolio = eff_frontier_df.iloc[0]
        best_weights_str = eff_frontier_df.index[0]
        best_weights = [float(w.strip()) for w in best_weights_str.strip('[]').split(',')]

        # Show best weights
        best_weights_percent = [round(w * 100, 4) for w in best_weights]
        asset_columns = individual_returns.columns.tolist()
        if include_cash == "Yes":
            asset_columns = asset_columns + ['Cash']
        formatted_weights = [f"{asset}: {weight}%" for asset, weight in zip(asset_columns, best_weights_percent)]

        st.subheader(f"Most Efficient Weights for Highest Sharpe Ratio ({best_portfolio['Sharpe Ratio']:.4f}):")
        st.write(", ".join(formatted_weights))

        # Efficient Frontier Plot using Plotly
        fig = px.scatter(
            eff_frontier_df,
            x='Standard Deviation',
            y='Returns',
            color='Sharpe Ratio',
            hover_data=['Sharpe Ratio'],
            title='Efficient Frontier'
        )
        fig.update_layout(
            xaxis_title='Standard Deviation',
            yaxis_title='Returns'
        )
        st.plotly_chart(fig)

        # Calculate portfolio returns and VaR/CVaR
        # Handle portfolio returns calculation differently when cash is included
        if include_cash == "Yes":
            weights_risky = np.array(best_weights[:-1])
            weight_cash = best_weights[-1]

            sum_weights_risky = np.sum(weights_risky)
            if sum_weights_risky > 0:
                normalized_weights_risky = weights_risky / sum_weights_risky
                portfolio_returns_risky = portfolio_allocator.calculate_portfolio(normalized_weights_risky)
            else:
                portfolio_returns_risky = pd.Series(0, index=portfolio_allocator.returns.index)

            portfolio_returns = portfolio_returns_risky * sum_weights_risky + weight_cash * risk_free_rate
        else:
            portfolio_returns = portfolio_allocator.calculate_portfolio(best_weights)


        var_calculator.returns = portfolio_returns

        time_horizon = 1
        hVaR = -var_calculator.historical_var(alpha) * np.sqrt(time_horizon)
        historical_cvar = -var_calculator.historical_cvar(alpha) * np.sqrt(time_horizon)

        # Display VaR value
        st.subheader(f"1 Day Historical VaR ({alpha}%) - {hVaR:.4f}")

        # Compute histogram data for VaR and CVaR
        num_bins = 21
        counts, bin_edges = np.histogram(portfolio_returns, bins=num_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        var_threshold = np.percentile(portfolio_returns, alpha)

        # Determine colors based on VaR threshold
        colors = ['red' if center <= var_threshold else 'blue' for center in bin_centers]

        # Create VaR Histogram
        fig_var = go.Figure(data=[go.Bar(
            x=bin_centers,
            y=counts,
            marker_color=colors,
            opacity=0.7
        )])
        fig_var.add_vline(
            x=var_threshold,
            line_color='red',
            line_dash='dash',
            annotation_text=f'VaR ({alpha}%) = {var_threshold:.4f}',
            annotation_position="top right"
        )
        fig_var.update_layout(
            title='Portfolio Returns Histogram with VaR',
            xaxis_title='Returns',
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig_var)

        # Display CVaR value
        st.subheader(f"1 Day Historical CVaR ({alpha}%) - {historical_cvar:.4f}")

        # Create CVaR Histogram
        fig_cvar = go.Figure(data=[go.Bar(
            x=bin_centers,
            y=counts,
            marker_color=colors,
            opacity=0.7
        )])

        cvar_value = var_calculator.historical_cvar(alpha)

        fig_cvar.add_vline(
            x=cvar_value,
            line_color='darkred',
            line_dash='dash',
            annotation_text=f'CVaR ({alpha}%) = {cvar_value:.4f}',
            annotation_position="top right"
        )

        fig_cvar.update_layout(
            title='Portfolio Returns Histogram with CVaR',
            xaxis_title='Returns',
            yaxis_title='Frequency'
        )

        st.plotly_chart(fig_cvar)

        st.session_state.calculated = True

        # Time Series Graph
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Create DataFrame for the cumulative returns
        cumulative_returns_df = pd.DataFrame(cumulative_returns, index=individual_returns.index, columns=['Cumulative Portfolio Value'])

        # Plot the cumulative returns
        fig_cumulative_returns = px.line(cumulative_returns_df, x=cumulative_returns_df.index, y='Cumulative Portfolio Value', title='Portfolio Value Over Time')
        fig_cumulative_returns.update_layout(xaxis_title='Date', yaxis_title='Cumulative Portfolio Value')
        st.plotly_chart(fig_cumulative_returns)

        st.session_state.calculated = True

    except Exception as e:
        st.error(f"Error during calculation: {e}")

if st.session_state.calculated:
    # Create a new set of columns at the end of the page
    st.write("")  # Add some space
    st.write("")  # Add more space
    reset_col1, reset_col2, reset_col3 = st.columns([1.5, 1, 1])
    
    with reset_col2:
        if st.button("Reset"):
            st.session_state.calculated = False
            st.rerun()