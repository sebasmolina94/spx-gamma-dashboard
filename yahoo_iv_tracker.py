import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class YahooFinanceIVTracker:
    def __init__(self):
        """Initialize the Yahoo Finance IV Tracker."""
        pass

    def get_option_chain(self, symbol):
        """Fetch the option chain for a given symbol."""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options  # List of expiration dates
            option_chain = []

            for exp_date in expirations:
                options = ticker.option_chain(exp_date)
                for option_type, data in [("CALL", options.calls), ("PUT", options.puts)]:
                    for _, row in data.iterrows():
                        option_chain.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'option_type': option_type,
                            'expiration': exp_date,
                            'strike': row['strike'],
                            'implied_volatility': row['impliedVolatility'],
                            'bid': row['bid'],
                            'ask': row['ask'],
                            'last_price': row['lastPrice'],
                            'volume': row['volume'],
                            'open_interest': row['openInterest']
                        })

            return pd.DataFrame(option_chain)
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return None

    def calculate_weekly_iv_peaks(self, iv_data):
        """Calculate weekly IV peaks from the option chain data."""
        # Ensure timestamp is in datetime format
        iv_data['timestamp'] = pd.to_datetime(iv_data['timestamp'])

        # Add a week column for grouping
        iv_data['week'] = iv_data['timestamp'].dt.isocalendar().week
        iv_data['year'] = iv_data['timestamp'].dt.isocalendar().year

        # Group by year, week, and option type to find max IV for each week
        weekly_peaks = iv_data.groupby(['year', 'week', 'option_type'])['implied_volatility'].max().reset_index()

        # Create a more readable date column representing the week
        weekly_peaks['week_date'] = weekly_peaks.apply(
            lambda row: f"{row['year']}-W{row['week']}", axis=1
        )

        return weekly_peaks

    def plot_weekly_iv_peaks(self, weekly_peaks, symbol):
        """Plot weekly IV peaks for visualization."""
        plt.figure(figsize=(12, 6))

        # Filter for PUTs and CALLs
        put_data = weekly_peaks[weekly_peaks['option_type'] == 'PUT']
        call_data = weekly_peaks[weekly_peaks['option_type'] == 'CALL']

        # Plot
        plt.plot(put_data['week_date'], put_data['implied_volatility'], 'r-', label='PUT IV')
        plt.plot(call_data['week_date'], call_data['implied_volatility'], 'b-', label='CALL IV')

        plt.title(f'Weekly IV Peaks for {symbol}')
        plt.xlabel('Week')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{symbol}_weekly_iv_peaks.png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"Plot saved as {plot_filename}")
        return plot_filename

    def calculate_iv_walls(self, option_chain, level_type='weekly'):
        """
        Calculate IV walls based on open interest or volume.

        Args:
            option_chain (pd.DataFrame): The option chain data.
            level_type (str): 'weekly' or 'daily' to calculate IV walls for the respective timeframes.

        Returns:
            pd.DataFrame: DataFrame containing IV wall levels.
        """
        # Filter the option chain for the desired level type
        if level_type == 'weekly':
            expiration_filter = option_chain['expiration'].unique()[:1]  # Use the nearest expiration
        elif level_type == 'daily':
            expiration_filter = option_chain['expiration'].unique()[:5]  # Use the next 5 expirations
        else:
            raise ValueError("Invalid level_type. Use 'weekly' or 'daily'.")

        filtered_chain = option_chain[option_chain['expiration'].isin(expiration_filter)]

        # Group by strike price and calculate total open interest and volume
        iv_walls = filtered_chain.groupby('strike').agg({
            'open_interest': 'sum',
            'volume': 'sum'
        }).reset_index()

        # Sort by open interest and volume to identify key levels
        iv_walls = iv_walls.sort_values(by=['open_interest', 'volume'], ascending=False)

        return iv_walls

    def plot_iv_walls(self, iv_walls, symbol, level_type):
        """
        Plot IV walls for visualization.

        Args:
            iv_walls (pd.DataFrame): DataFrame containing IV wall levels.
            symbol (str): The symbol being analyzed.
            level_type (str): 'weekly' or 'daily' to indicate the type of IV walls.
        """
        plt.figure(figsize=(12, 6))

        # Plot open interest and volume as bar charts
        plt.bar(iv_walls['strike'], iv_walls['open_interest'], color='blue', alpha=0.6, label='Open Interest')
        plt.bar(iv_walls['strike'], iv_walls['volume'], color='orange', alpha=0.6, label='Volume')

        plt.title(f'{level_type.capitalize()} IV Walls for {symbol}')
        plt.xlabel('Strike Price')
        plt.ylabel('Open Interest / Volume')
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{symbol}_{level_type}_iv_walls.png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"IV Wall plot saved as {plot_filename}")
        return plot_filename

    def track_weekly_iv_peaks(self, symbol):
        """
        Main method to track weekly IV peaks for a symbol.

        This method:
        1. Gets the current option chain
        2. Extracts IV data
        3. Calculates weekly peaks
        4. Plots and saves the results
        """
        # Get option chain
        print(f"Fetching option chain for {symbol}...")
        option_chain = self.get_option_chain(symbol)

        if option_chain is None or option_chain.empty:
            print("Failed to fetch option chain or no data available.")
            return

        # Calculate weekly peaks
        print("Calculating weekly IV peaks...")
        weekly_peaks = self.calculate_weekly_iv_peaks(option_chain)

        # Save weekly peaks
        weekly_peaks_file = f"{symbol}_weekly_iv_peaks.csv"
        weekly_peaks.to_csv(weekly_peaks_file, index=False)
        print(f"Weekly peaks saved to {weekly_peaks_file}")

        # Plot results
        print("Generating plot...")
        plot_file = self.plot_weekly_iv_peaks(weekly_peaks, symbol)

        return {
            'weekly_peaks_file': weekly_peaks_file,
            'plot_file': plot_file,
            'weekly_peaks': weekly_peaks
        }

    def track_iv_walls(self, symbol):
        """
        Main method to track IV walls for a symbol.

        This method:
        1. Gets the current option chain
        2. Calculates weekly and daily IV walls
        3. Plots and saves the results
        """
        # Get option chain
        print(f"Fetching option chain for {symbol}...")
        option_chain = self.get_option_chain(symbol)

        if option_chain is None or option_chain.empty:
            print("Failed to fetch option chain or no data available.")
            return

        # Calculate weekly IV walls
        print("Calculating weekly IV walls...")
        weekly_iv_walls = self.calculate_iv_walls(option_chain, level_type='weekly')
        weekly_iv_walls_file = f"{symbol}_weekly_iv_walls.csv"
        weekly_iv_walls.to_csv(weekly_iv_walls_file, index=False)
        print(f"Weekly IV walls saved to {weekly_iv_walls_file}")

        # Plot weekly IV walls
        print("Generating weekly IV wall plot...")
        weekly_plot_file = self.plot_iv_walls(weekly_iv_walls, symbol, level_type='weekly')

        # Calculate daily IV walls
        print("Calculating daily IV walls...")
        daily_iv_walls = self.calculate_iv_walls(option_chain, level_type='daily')
        daily_iv_walls_file = f"{symbol}_daily_iv_walls.csv"
        daily_iv_walls.to_csv(daily_iv_walls_file, index=False)
        print(f"Daily IV walls saved to {daily_iv_walls_file}")

        # Plot daily IV walls
        print("Generating daily IV wall plot...")
        daily_plot_file = self.plot_iv_walls(daily_iv_walls, symbol, level_type='daily')

        return {
            'weekly_iv_walls_file': weekly_iv_walls_file,
            'weekly_plot_file': weekly_plot_file,
            'daily_iv_walls_file': daily_iv_walls_file,
            'daily_plot_file': daily_plot_file
        }

# Example usage
if __name__ == "__main__":
    # Create an instance of the tracker
    tracker = YahooFinanceIVTracker()

    # Track weekly IV peaks for SPY ETF
    results_peaks = tracker.track_weekly_iv_peaks('SPY')

    if results_peaks:
        print("\nWeekly IV peaks for SPY (most recent first):")
        recent_peaks = results_peaks['weekly_peaks'].sort_values(by=['year', 'week'], ascending=False).head(5)
        print(recent_peaks[['week_date', 'option_type', 'implied_volatility']])

    # Track IV walls for QQQ ETF
    results_walls = tracker.track_iv_walls('QQQ')

    if results_walls:
        print("\nIV wall tracking completed. Files generated:")
        print(results_walls)