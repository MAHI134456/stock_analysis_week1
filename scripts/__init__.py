import pandas as pd

def load_stock_data(filepath):
    """
    Load stock price data from a CSV file.
    Returns a pandas DataFrame .
    """
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df = df.sort_values('Date')
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"File is empty: {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None