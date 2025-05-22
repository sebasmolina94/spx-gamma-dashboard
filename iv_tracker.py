import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from configparser import ConfigParser
import time

class TDAmeritradeIVTracker:
    def __init__(self, config_file='config.ini'):
        """Initialize the TD Ameritrade IV Tracker with configuration."""
        self.config = ConfigParser()
        if os.path.exists(config_file):
            self.config.read(config_file)
        else:
            self._create_default_config(config_file)
            
        self.client_id = self.config.get('API', 'client_id')
        self.redirect_uri = self.config.get('API', 'redirect_uri')
        self.token_path = self.config.get('API', 'token_path')
        
        self.base_url = "https://api.tdameritrade.com/v1"
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        
        # Load tokens if they exist
        self._load_tokens()
    
    def _create_default_config(self, config_file):
        """Create a default configuration file."""
        self.config['API'] = {
            'client_id': 'GnhaaTIat2rZn5RTdzOaDHeVNUs72fSf@AMER.OAUTHAP',
            'redirect_uri': 'https://ebf7-2607-fb90-79f5-42bd-3d0c-58a5-9c21-b251.ngrok-free.app',
            'token_path': 'td_token.json'
        }
        
        with open(config_file, 'w') as f:
            self.config.write(f)
            
        print(f"Created default config file at {config_file}. Please edit it with your TD Ameritrade API credentials.")
    
    def _load_tokens(self):
        """Load access and refresh tokens if they exist."""
        if os.path.exists(self.token_path):
            try:
                token_data = pd.read_json(self.token_path, typ='series')
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                self.token_expiry = pd.to_datetime(token_data.get('expiry_time'))
                
                # Check if token is expired and refresh if needed
                if self.token_expiry and datetime.now() >= self.token_expiry:
                    self._refresh_access_token()
            except Exception as e:
                print(f"Error loading tokens: {e}")
    
    def authenticate(self, auth_code=None):
        """Authenticate with TD Ameritrade API using authorization code flow."""
        if not auth_code:
            # Generate the authorization URL for the user to visit
            auth_url = (
                f"https://auth.tdameritrade.com/auth?"
                f"response_type=code&redirect_uri={self.redirect_uri}&"
                f"client_id={self.client_id}"
            )
            
            print(f"Please visit the following URL to authenticate:")
            print(auth_url)
            print("\nAfter authorization, you'll be redirected to your redirect URI.")
            print("Copy the full redirect URL and extract the authorization code.")
            
            auth_code = input("\nEnter the authorization code: ")
        
        # Convert the authorization code (it comes URL-encoded)
        auth_code = auth_code.replace("%2F", "/")
        
        # Exchange authorization code for tokens
        resp = requests.post(
            f"{self.base_url}/oauth2/token",
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data={
                'grant_type': 'authorization_code',
                'access_type': 'offline',
                'code': auth_code,
                'client_id': f"{self.client_id}",
                'redirect_uri': self.redirect_uri
            }
        )
        
        if resp.status_code == 200:
            token_data = resp.json()
            self.access_token = token_data['access_token']
            self.refresh_token = token_data['refresh_token']
            # Set expiry to 30 minutes from now (access token validity period)
            self.token_expiry = datetime.now() + timedelta(minutes=30)
            
            # Save tokens
            self._save_tokens()
            print("Authentication successful!")
            return True
        else:
            print(f"Authentication failed: {resp.status_code} - {resp.text}")
            return False
    
    def _refresh_access_token(self):
        """Refresh the access token using the refresh token."""
        if not self.refresh_token:
            print("No refresh token available. Please authenticate first.")
            return False
        
        resp = requests.post(
            f"{self.base_url}/oauth2/token",
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data={
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': f"{self.client_id}@AMER.OAUTHAP"
            }
        )
        
        if resp.status_code == 200:
            token_data = resp.json()
            self.access_token = token_data['access_token']
            # Some implementations also return a new refresh token
            if 'refresh_token' in token_data:
                self.refresh_token = token_data['refresh_token']
            
            # Set expiry to 30 minutes from now
            self.token_expiry = datetime.now() + timedelta(minutes=30)
            
            # Save tokens
            self._save_tokens()
            return True
        else:
            print(f"Token refresh failed: {resp.status_code} - {resp.text}")
            return False
    
    def _save_tokens(self):
        """Save access and refresh tokens to a file."""
        token_data = {
            'access_token': self.access_token,
            'refresh_token': self.refresh_token,
            'expiry_time': self.token_expiry.isoformat() if self.token_expiry else None
        }
        
        pd.Series(token_data).to_json(self.token_path)
    
    def _ensure_authenticated(self):
        """Ensure that we have a valid access token."""
        if not self.access_token or (self.token_expiry and datetime.now() >= self.token_expiry):
            return self._refresh_access_token()
        return True
    
    def get_option_chain(self, symbol):
        """Get the full option chain for a symbol."""
        if not self._ensure_authenticated():
            print("Authentication required. Please call authenticate() first.")
            return None
        
        url = f"{self.base_url}/marketdata/chains"
        params = {
            'apikey': self.client_id,
            'symbol': symbol,
            'contractType': 'ALL',
            'strikeCount': 50,  # Adjust as needed
            'includeQuotes': 'TRUE',
            'strategy': 'SINGLE',
            'range': 'ALL',
            'optionType': 'ALL'
        }
        
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            resp = requests.get(url, params=params, headers=headers)
            
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"API request failed: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return None
    
    def extract_iv_data(self, option_chain):
        """Extract implied volatility data from the option chain response."""
        if not option_chain or 'putExpDateMap' not in option_chain or 'callExpDateMap' not in option_chain:
            print("Invalid option chain data")
            return None
        
        # Initialize an empty list to store IV data
        iv_data = []
        
        # Extract IV data from puts
        for exp_date, strikes in option_chain['putExpDateMap'].items():
            for strike, options in strikes.items():
                for option in options:
                    iv_data.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'option_type': 'PUT',
                        'expiration': exp_date.split(':')[0],
                        'strike': float(strike),
                        'implied_volatility': option['volatility'] / 100,  # TDA returns IV as percentage
                        'delta': option.get('delta', None),
                        'gamma': option.get('gamma', None),
                        'theta': option.get('theta', None),
                        'vega': option.get('vega', None),
                        'bid': option.get('bid', None),
                        'ask': option.get('ask', None),
                        'last': option.get('last', None),
                        'mark': option.get('mark', None),
                        'volume': option.get('totalVolume', None),
                        'open_interest': option.get('openInterest', None),
                        'days_to_expiration': option.get('daysToExpiration', None)
                    })
        
        # Extract IV data from calls
        for exp_date, strikes in option_chain['callExpDateMap'].items():
            for strike, options in strikes.items():
                for option in options:
                    iv_data.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'option_type': 'CALL',
                        'expiration': exp_date.split(':')[0],
                        'strike': float(strike),
                        'implied_volatility': option['volatility'] / 100,  # TDA returns IV as percentage
                        'delta': option.get('delta', None),
                        'gamma': option.get('gamma', None),
                        'theta': option.get('theta', None),
                        'vega': option.get('vega', None),
                        'bid': option.get('bid', None),
                        'ask': option.get('ask', None),
                        'last': option.get('last', None),
                        'mark': option.get('mark', None),
                        'volume': option.get('totalVolume', None),
                        'open_interest': option.get('openInterest', None),
                        'days_to_expiration': option.get('daysToExpiration', None)
                    })
        
        return pd.DataFrame(iv_data)
    
    def calculate_weekly_iv_peaks(self, iv_history_df):
        """Calculate weekly IV peaks from historical IV data."""
        # Ensure timestamp is in datetime format
        iv_history_df['timestamp'] = pd.to_datetime(iv_history_df['timestamp'])
        
        # Add a week column for grouping
        iv_history_df['week'] = iv_history_df['timestamp'].dt.isocalendar().week
        iv_history_df['year'] = iv_history_df['timestamp'].dt.isocalendar().year
        
        # Group by year, week, and option type to find max IV for each week
        weekly_peaks = iv_history_df.groupby(['year', 'week', 'option_type'])['implied_volatility'].max().reset_index()
        
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
    
    def save_iv_data(self, iv_data, symbol):
        """Save IV data to a CSV file."""
        filename = f"{symbol}_iv_data.csv"
        
        # Check if file exists to append or create new
        if os.path.exists(filename):
            # Read existing data
            existing_data = pd.read_csv(filename)
            # Append new data
            combined_data = pd.concat([existing_data, iv_data], ignore_index=True)
            # Remove duplicates if any
            combined_data.drop_duplicates(inplace=True)
            # Save
            combined_data.to_csv(filename, index=False)
        else:
            # Create new file
            iv_data.to_csv(filename, index=False)
        
        print(f"IV data saved to {filename}")
        return filename
    
    def track_weekly_iv_peaks(self, symbol, days_back=30):
        """
        Main method to track weekly IV peaks for a symbol.
        
        This method:
        1. Gets the current option chain
        2. Extracts IV data
        3. Combines with historical data if available
        4. Calculates weekly peaks
        5. Plots and saves the results
        """
        # Get option chain
        print(f"Fetching option chain for {symbol}...")
        option_chain = self.get_option_chain(symbol)
        
        if not option_chain:
            print("Failed to fetch option chain.")
            return
        
        # Extract IV data
        print("Extracting IV data...")
        iv_data = self.extract_iv_data(option_chain)
        
        if iv_data is None or iv_data.empty:
            print("No IV data found.")
            return
        
        # Save new IV data
        iv_data_file = self.save_iv_data(iv_data, symbol)
        
        # Load combined historical data
        historical_iv = pd.read_csv(iv_data_file)
        
        # Calculate weekly peaks
        print("Calculating weekly IV peaks...")
        weekly_peaks = self.calculate_weekly_iv_peaks(historical_iv)
        
        # Save weekly peaks
        weekly_peaks_file = f"{symbol}_weekly_iv_peaks.csv"
        weekly_peaks.to_csv(weekly_peaks_file, index=False)
        print(f"Weekly peaks saved to {weekly_peaks_file}")
        
        # Plot results
        print("Generating plot...")
        plot_file = self.plot_weekly_iv_peaks(weekly_peaks, symbol)
        
        return {
            'iv_data_file': iv_data_file,
            'weekly_peaks_file': weekly_peaks_file,
            'plot_file': plot_file,
            'weekly_peaks': weekly_peaks
        }

# Example usage
if __name__ == "__main__":
    # Create an instance of the tracker
    tracker = TDAmeritradeIVTracker()
    
    # Check if authentication is needed
    if not tracker.access_token:
        tracker.authenticate()
    
    # Track weekly IV peaks for SPY ETF
    results = tracker.track_weekly_iv_peaks('SPY')
    
    if results:
        print("\nWeekly IV peaks for SPY (most recent first):")
        recent_peaks = results['weekly_peaks'].sort_values(by=['year', 'week'], ascending=False).head(5)
        print(recent_peaks[['week_date', 'option_type', 'implied_volatility']])