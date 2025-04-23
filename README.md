# Crypto Portfolio Dashboard

This streamlit application provides a quick snapshot of the performance and risk of a collection of crypto assets. The efficient frontier outlines the best weighting based on the excess returns and volatilities, the VaR and CVaR charts provide quick risk management snapshots, and the Portfolio Over Time charts shows the cumulative returns of the portfolio over the chosen time range.

## Features

-   **Efficient Frontier:** Outputs the most efficient weights for each chosen asset based on the excess returns and volatility of each asset. This is done by generating a set of random weights, in which the one with the highest Sharpe Ratio is chosen.
-   **1 Day Historical Value-at-Risk (VaR):** The 1 day historical VaR represents the loss threshold for a given confidence level, where holding the most efficient portfolio for one day will not exceed this amount. This is calculated by using the percentile of the historical returns corresponding to the confidence level.
-   **1 Day Historical Conditional Value-at-Risk (CVaR):** The 1 day historical CVaR (also known as Expected Shortfall) represents the average amount that would be lost given that the loss exceeds the VaR threshold at the specified confidence level. It is often used as the more conservative risk metric compared to VaR.
-   **Portfolio Value Over Time:** Shows the cumulative returns of the most efficient portfolio based on the efficient frontier.

## How to Use

1.  **Select Cryptocurrencies:** Use the multiselect widget to choose the cryptocurrencies for your portfolio.
2.  **Enter Date Range:** Input the start and end dates in YYYY-MM-DD format.
3.  **Specify Risk-Free Rate:** Enter the risk-free rate as a decimal (e.g., 0.04 for 4%).
4.  **Set VaR Confidence Level:** Enter the desired VaR confidence level as a percentage (e.g., 5 for 5%).
5.  **Click Calculate:** Press the "Calculate" button to perform the analysis.
6.  **View Results:** The app will display the efficient frontier plot, optimal portfolio weights, VaR, CVaR, and corresponding histograms.
7.  **Reset:** You can reset the calculation by clicking the "Reset" button.

## Code Overview

The code is structured into the following main components:

*   **CryptoDataFetcher Class:** Fetches historical kline data from the Binance API for the selected cryptocurrencies.
*   **PortfolioAllocator Class:** Calculates portfolio returns based on given weights.
*   **CryptoVaRCalculator Class:** Computes historical VaR, CVaR, and generates the efficient frontier.
*   **Streamlit Interface:** Provides the user interface for input and output.

### CryptoDataFetcher

This class is responsible for fetching cryptocurrency data from the Binance API.

*   `__init__(self, symbols, start_time, end_time)`: Initializes the data fetcher with a list of cryptocurrency symbols, a start time, and an end time.
*   `fetch_klines(self, symbol, start_time)`: Fetches kline data for a given symbol starting from a specific time. It handles API requests with error retries and exponential backoff.
*   `make_request(self)`: Makes requests for all symbols, processes the data, and returns a Pandas DataFrame containing the returns for each cryptocurrency.

### PortfolioAllocator

This class is responsible for calculating portfolio returns based on specified weights.

*   `__init__(self, data_fetcher)`: Initializes the allocator with a `CryptoDataFetcher` instance.
*   `calculate_portfolio(self, weights)`: Calculates the portfolio returns given a set of weights.
*   `calculate_individual_returns(self)`: Returns the returns of the individual assets.

### CryptoVaRCalculator

This class computes VaR, CVaR, and the efficient frontier for a given portfolio.

*   `__init__(self, portfolio_returns, individual_returns)`: Initializes the calculator with portfolio and individual returns.
*   `historical_var(self, alpha)`: Calculates historical Value at Risk (VaR) for a given confidence level.
*   `historical_cvar(self, alpha)`: Calculates historical Conditional Value at Risk (CVaR) for a given confidence level.
*   `eff_frontier(self, num_portfolios=10000, risk_free_rate=0.04, include_cash=False)`: Generates the efficient frontier by simulating multiple portfolios with random weights.

### Streamlit Interface

The Streamlit interface allows users to interact with the application.

*   **Input Widgets:** Provides widgets for selecting cryptocurrencies, entering dates, and setting parameters.
*   **Button:** Triggers the calculation when clicked.
*   **Output Display:** Displays the efficient frontier plot, portfolio weights, VaR, CVaR, and related charts.

## Dependencies

*   streamlit
*   requests
*   pandas
*   datetime
*   time
*   numpy
*   plotly.express
*   plotly.graph_objects
*   scipy.interpolate

You can install these dependencies using pip:
pip install streamlit requests pandas numpy plotly scipy


## Disclaimer

This application is for informational purposes only and should not be considered financial advice.
