# Crypto Portfolio Dashboard

This Streamlit application calculates and displays the efficient frontier of a crypto portfolio, along with Value at Risk (VaR) and Conditional Value at Risk (CVaR) metrics. It allows users to select cryptocurrencies, specify a time period, and set parameters like risk-free rate and VaR confidence level.

## Features

-   **Cryptocurrency Selection:** Choose from a list of cryptocurrencies to include in the portfolio.
-   **Date Range Input:** Specify the start and end dates for the analysis.
-   **Risk-Free Rate:** Set the risk-free rate for efficient frontier calculation.
-   **VaR Confidence Level:** Adjust the confidence level for VaR and CVaR calculations.
-   **Efficient Frontier Visualization:** Interactive plot displaying the efficient frontier.
-   **Portfolio Optimization:** Determines the portfolio with the highest Sharpe ratio.
-   **VaR and CVaR Calculation:** Computes and displays VaR and CVaR for the optimized portfolio.
-   **Interactive Charts:** Displays portfolio returns and risk metrics using Plotly.

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
