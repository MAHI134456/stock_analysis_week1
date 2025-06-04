import pandas as pd
import talib
import pynance as pn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

# Ensure NLTK VADER is downloaded
import nltk
nltk.download('vader_lexicon')

def perform_sentiment_correlation(news_filepath, data_dir, target_tickers):
    """
    Align news and stock data, perform sentiment analysis,
    calculate daily returns, and analyze correlation.

    Returns:
    None (saves results to data/processed/ and reports/figures/)
    """
    # Setup output directories
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Load and clean news data
    try:
        news_df = pd.read_csv(news_filepath, on_bad_lines='skip')
        print("✅ News dataset loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading news file: {e}")
        return

    # Normalize ticker names
    news_df['stock'] = news_df['stock'].replace({'FB': 'META', 'MFT': 'MSFT'})

    # Filter only target tickers
    news_df = news_df[news_df['stock'].isin(target_tickers)]

    # Convert date column to datetime
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce').dt.date

    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Build companies list dynamically
    companies = []
    for ticker in target_tickers:
        filepath = os.path.join(data_dir, f'technical_indicators_{ticker}.csv')
        if os.path.exists(filepath):
            companies.append({
                'ticker': ticker,
                'filepath': filepath
            })
        else:
            print(f"⚠️ File not found for {ticker}: {filepath}")

    # Process each company
    for company in companies:
        ticker = company['ticker']
        filepath = company['filepath']

        # Load stock data
        stock_df = pd.read_csv(filepath, parse_dates=['Date'])
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
        stock_df['date'] = stock_df['Date'].dt.date

        # Calculate daily returns
        stock_df['Daily_Return'] = stock_df['Close'].pct_change()

        # Filter news for this ticker
        ticker_news = news_df[news_df['stock'] == ticker].copy()

        # Perform sentiment analysis
        ticker_news['sentiment'] = ticker_news['headline'].apply(
            lambda x: sid.polarity_scores(str(x))['compound']
        )

        # Aggregate sentiment by date (mean per day)
        daily_sentiment = ticker_news.groupby('date')['sentiment'].mean().reset_index()
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

        # Ensure both date columns are same type before merging
        stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.date

        # Align with stock data
        aligned_df = pd.merge(
        stock_df[['date', 'Close', 'Daily_Return']],
        daily_sentiment,
        on='date',
        how='inner'
        )

        # Skip if no alignment
        if aligned_df.empty:
            print(f"⚠️ No aligned data for {ticker}")
            continue

        # Calculate correlation
        correlation = aligned_df['sentiment'].corr(aligned_df['Daily_Return'])
        print(f"\n📈 Correlation for {ticker}: {correlation:.4f}")

        # Save aligned data
        aligned_df.to_csv(f'data/processed/aligned_sentiment_returns_{ticker}.csv', index=False)
        print(f"💾 Saved aligned data for {ticker}")

        # Plot scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='sentiment', y='Daily_Return', data=aligned_df)
        plt.title(f'{ticker}: Sentiment vs Daily Return\n(Correlation: {correlation:.4f})')
        plt.xlabel('Sentiment Score (Compound)')
        plt.ylabel('Daily Return')
        plt.grid(True)
        plt.savefig(f'reports/figures/{ticker}_sentiment_return_correlation.png')
        plt.close()

        print(f"📊 Saved plot for {ticker}")