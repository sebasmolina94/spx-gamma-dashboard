import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import pandas as pd

# Fetch market data from Yahoo Finance
def get_data(ticker="SPY", period="1mo", interval="1h"):
    data = yf.download(ticker, period=period, interval=interval)
    
    # Check if data is empty
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} with period {period} and interval {interval}.")
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]  # Extract the first level of the MultiIndex
    
    # Ensure all columns are numeric and handle non-numeric values
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            print(f"Warning: Column '{col}' is missing in the data.")
    
    # Drop rows with invalid numeric values in essential columns
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data.index.name = "Date"
    return data

# Fetch options data and identify IV walls
def identify_iv_walls(ticker):
    ticker_obj = yf.Ticker(ticker)
    expirations = ticker_obj.options  # Get all expiration dates
    nearest_expiration = expirations[0]  # Use the nearest expiration date
    options = ticker_obj.option_chain(nearest_expiration)

    # Combine calls and puts into a single DataFrame
    calls = options.calls
    puts = options.puts
    options_data = pd.concat([calls, puts])

    # Calculate IV walls based on open interest
    iv_walls = options_data.groupby("strike").agg({
        "openInterest": "sum",
        "volume": "sum"
    }).reset_index()

    # Identify key levels
    support = iv_walls.loc[iv_walls["openInterest"].idxmax(), "strike"]  # Strike with max open interest
    resistance = iv_walls.loc[iv_walls["volume"].idxmax(), "strike"]  # Strike with max volume
    midpoint = (support + resistance) / 2  # Midpoint between support and resistance

    return {
        "support": support,
        "resistance": resistance,
        "midpoint": midpoint
    }

# Function to plot market data with candlesticks and key levels
def plot_market_map(data, levels, ticker="SPY"):
    mc = mpf.make_marketcolors(up="lime", down="red", edge="inherit", wick="gray", volume="gray")
    s = mpf.make_mpf_style(base_mpl_style="dark_background", marketcolors=mc, gridcolor="gray")

    # Highlight support and resistance zones
    hlines = [levels["support"], levels["resistance"], levels["midpoint"]]
    colors = ["blue", "red", "yellow"]

    # Create the plot
    fig, axlist = mpf.plot(
        data,
        type="candle",
        style=s,
        title=f"{ticker} Market Map",
        ylabel="Price",
        hlines=dict(hlines=hlines, colors=colors, linewidths=[1.5, 1.5, 1]),
        volume=True,  # Show volume bars
        returnfig=True,  # Return the figure and axes for further customization
    )

    # Add labels for the levels on the right axis
    ax = axlist[0]  # Main candlestick chart axis
    for level, color, label in zip(hlines, colors, ["Support", "Resistance", "Midpoint"]):
        ax.annotate(
            f"{label}: {level:.2f}",
            xy=(1, level),
            xycoords=("axes fraction", "data"),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="black", alpha=0.8),
        )

    # Show the plot
    plt.show()

if __name__ == "__main__":
    ticker = "QQQ"  # Change to "QQQ" or any other ticker as needed

    # Fetch market data
    data = get_data(ticker)

    # Identify IV walls based on options data
    levels = identify_iv_walls(ticker)

    # Plot the market map with IV walls
    plot_market_map(data, levels, ticker)
