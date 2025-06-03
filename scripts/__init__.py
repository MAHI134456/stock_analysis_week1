import pandas as pd
import talib
import os
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


def calculate_technical_indicators(ticker, filepath):
    """
    Calculate technical indicators (SMA, RSI, MACD, Daily Return) for a single company.
    
    Returns:
    None (saves results to data/processed/technical_indicators_{ticker}.csv)
    """
    # Load data
    df = load_stock_data(filepath)
    
    if df is not None:
        # Calculate technical indicators
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Print first few rows
        print(f"\nData for {ticker} with technical indicators:")
        print(df[['Date', 'Close', 'SMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'Daily_Return']].head())
        
        # Ensure output directory exists
        output_dir = '../data/processed'  # Adjusted path to match the original script
        os.makedirs(output_dir, exist_ok=True)  # Creates folder if it doesn't exist
        
    
        output_path = f'{output_dir}/technical_indicators_{ticker}.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved indicators for {ticker} to {output_path}")
    else:
        print(f"Failed to load data for {ticker}")   