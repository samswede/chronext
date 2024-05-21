import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical market data from Yahoo Finance for a given ticker between specified dates.

    Args:
    ticker (str): Ticker symbol of the stock or index.
    start_date (str): Start date in YYYY-MM-DD format.
    end_date (str): End date in YYYY-MM-DD format.

    Returns:
    pandas.DataFrame: Dataframe containing the historical market data.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def save_data(data, filename):
    """
    Saves the DataFrame to a CSV file.

    Args:
    data (pandas.DataFrame): Dataframe to save.
    filename (str): Filename for the CSV file.
    """
    data.to_csv(filename)

def main():
    """
    Main function to fetch and save data with predefined parameters.
    """
    ticker = '^MXX'  
    # Examples: 
    #   S&P 500 index: ^GSPC
    #   LVMH Moët Hennessy Louis Vuitton SE: MC.PA
    #   Swiss Market Index: ^SSMI
    #   Swatch Group (Swiss Company): UHR.SW
    #   Richemont (Swiss Company): CFR.SW
    #   
    #   Monaco Market Index: ^MXX


    start_date = '2020-05-13'
    end_date = '2024-05-11'
    filename = 'MXX.csv'

    data = fetch_data(ticker, start_date, end_date)
    save_data(data, filename)
    print(f'Data for {ticker} has been saved to {filename}')

if __name__ == "__main__":
    main()
