import pandas as pd
import talib
import pynance as pn
import matplotlib.pyplot as plt
import seaborn as sns
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


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_indicators(ticker, filepath):
    """
    Load precomputed indicators from CSV and calculate missing financial metrics.
    
    Returns:
    None (saves plots to reports/figures/)
    """
    # Ensuring output directory exists
    os.makedirs('reports/figures', exist_ok=True)

    # Load data
    df = pd.read_csv(filepath, parse_dates=['Date'])

    if not df.empty:
        print(f"Loaded data for {ticker}")

        # Calculateing metrics
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change()

        if 'Volatility' not in df.columns:
            df['Volatility'] = df['Daily_Return'].rolling(window=252).std() * np.sqrt(252)

        if 'Cumulative_Return' not in df.columns:
            df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1

        
        # Plot 1: Close Price and SMA_20
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='Close Price')
        plt.plot(df['Date'], df['SMA_20'], label='SMA_20', linestyle='--')
        plt.title(f'{ticker} Close Price and 20-Day SMA')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'reports/figures/{ticker}_price_sma.png')
        plt.close()
        
        # Plot 2: RSI_14
        plt.figure(figsize=(12, 4))
        plt.plot(df['Date'], df['RSI_14'], label='RSI_14')
        plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        plt.title(f'{ticker} RSI (14-Day)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'reports/figures/{ticker}_rsi.png')
        plt.close()
        
        # Plot 3: MACD and Signal Line
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['MACD'], label='MACD')
        plt.plot(df['Date'], df['MACD_signal'], label='Signal Line', linestyle='--')
        plt.title(f'{ticker} MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'reports/figures/{ticker}_macd.png')
        plt.close()
        
        # Plot 4: Daily Return and Volatility
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(df['Date'], df['Daily_Return'], label='Daily Return')
        plt.title(f'{ticker} Daily Return')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(df['Date'], df['Volatility'], label='Annualized Volatility')
        plt.title(f'{ticker} Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'reports/figures/{ticker}_return_volatility.png')
        plt.close()
        
        # Plot 5: Cumulative Return
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Cumulative_Return'], label='Cumulative Return')
        plt.title(f'{ticker} Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'reports/figures/{ticker}_cumulative_return.png')
        plt.close()

        print(f"Saved visualizations for {ticker} to reports/figures/")
    else:
        print(f"Failed to load data for {ticker}")