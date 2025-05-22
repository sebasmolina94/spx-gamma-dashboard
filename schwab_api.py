import json
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from authlib.integrations.httpx_client import OAuth2Client
import config

background_color = '#161a25'

class Client:
    def __init__(self) -> None:
        self.filepath = 'schwab_client.json'
        self.TOKEN_ENDPOINT = 'https://api.schwabapi.com/v1/oauth/token'
        self.session: OAuth2Client = None
        self.config = {
            'client': {
                'api_key': config.API_KEY,
                'app_secret': config.APP_SECRET,
                'callback': config.CALLBACK_URL,
                'setup': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            },
            'token': {}
        }
        self.load()

    def setup(self) -> bool:
        try:
            oauth = OAuth2Client(self.config['client']['api_key'], redirect_uri=self.config['client']['callback'])
            authorization_url, state = oauth.create_authorization_url('https://api.schwabapi.com/v1/oauth/authorize')
            print('Click the link below:')
            print(authorization_url)
            redirected_url = input('Paste URL: ').strip()
            self.config['token'] = oauth.fetch_token(
                self.TOKEN_ENDPOINT,
                authorization_response=redirected_url,
                client_id=self.config['client']['api_key'],
                auth=(self.config['client']['api_key'], self.config['client']['app_secret'])
            )
            self.save()
            self.load_session()
            return True
        except Exception as e:
            print(f'Setup failed: {e}')
            return False

    def get_Quote(self, underlying: str) -> dict:
        try:
            if not self.check_session():
                raise Exception('No valid session')
            underlying = self.clean_symbol(underlying)
            endpoint = f'https://api.schwabapi.com/marketdata/v1/quotes'
            params = {'symbols': underlying}
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f'Could not get quote for {underlying}: {e}')
            return None

    def get_Option(self, underlying: str, fromDate: datetime, toDate: datetime, strike_count: int) -> dict:
        try:
            if not self.check_session():
                raise Exception('No valid session')
            underlying = self.clean_symbol(underlying)
            print(f'Getting options chain for {underlying} from {fromDate.strftime("%Y-%m-%d")} ' + 
                  (f'to {toDate.strftime("%Y-%m-%d")}' if toDate else ''))
            endpoint = 'https://api.schwabapi.com/marketdata/v1/chains'
            params = {
                'symbol': underlying,
                'contractType': 'ALL',
                'fromDate': fromDate.strftime('%Y-%m-%d'),
                'toDate': toDate.strftime('%Y-%m-%d'),
                'strikeCount': strike_count
            }
            if toDate:
                params['toDate'] = toDate.strftime('%Y-%m-%d')
            else:
                params['toDate'] = fromDate.strftime('%Y-%m-%d')
                
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f'Could not get options chain for {underlying}: {e}')
            return None

    
    def get_intraday_ohlc(self, symbol, interval='5min', hours=6):
        import pandas as pd
        from datetime import datetime, timedelta

        end = datetime.now()
        start = end - timedelta(hours=hours)

        endpoint = "https://api.schwabapi.com/v1/marketdata/pricehistory"

        params = {
            "symbol": symbol,
            "frequencyType": "minute",
            "frequency": int(interval.replace("min", "")),
            "startDate": int(start.timestamp() * 1000),
            "endDate": int(end.timestamp() * 1000),
            "needExtendedHoursData": "false"
        }

        response = self.session.get(endpoint, params=params)

        if response.status_code != 200:
            raise Exception(f"Schwab intraday OHLC fetch failed: {response.text}")

        data = response.json()

        if "candles" not in data:
            raise Exception(f"No candlestick data returned: {data}")

        candles = data["candles"]
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

        return df[["datetime", "open", "high", "low", "close"]]

    def check_session(self) -> bool:
            try:
                if self.session is None:
                    self.load_session()
                expires = datetime.fromtimestamp(int(self.session.token['expires_at']), timezone.utc)
                current = datetime.now(timezone.utc)
                if (expires - current).total_seconds() <= 0:
                    self.refresh_token()
                return True
            except Exception as e:
                print(f'Checking session failed: {e}')
                return False

    def write_token(self, token, *args, **kwargs):
        try:
            self.config['token'] = token
            self.save()
            self.load_session()
        except Exception as e:
            print(f'Token could not be loaded: {e}')

    def save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.config, f)
        except Exception as e:
            print(f'Configuration could not be saved: {e}')

    def load(self):
        try:
            if not os.path.exists(self.filepath):
                print('Config file not found, run setup!')
                return
            with open(self.filepath, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f'Configuration could not be loaded: {e}')

    def refresh_token(self):
        try:
            if self.session is None:
                self.load_session()
            token = self.config['token']
            new_token = self.session.fetch_token(
                self.TOKEN_ENDPOINT,
                grant_type='refresh_token',
                refresh_token=token['refresh_token'],
                access_type='offline'
            )
            self.config['token'] = new_token
            self.save()
            print('Token refreshed')
        except Exception as e:
            print(f'Token could not be refreshed: {e}')

    def load_session(self):
        try:
            if 'client' not in self.config or 'api_key' not in self.config['client'] or 'app_secret' not in self.config['client']:
                raise Exception('API Key or App Secret missing in configuration')
            token = self.config['token']
            self.session = OAuth2Client(
                self.config['client']['api_key'],
                self.config['client']['app_secret'],
                token=token,
                token_endpoint=self.TOKEN_ENDPOINT,
                update_token=self.write_token
            )
        except Exception as e:
            print(f'Could not load session: {e}')
            raise e

    def clean_symbol(self, symbol: str) -> str:
        # Clean the symbol format
        return symbol.replace('.X', '')

    def options_chain_to_dataframe(self, options_chain: dict) -> tuple:
        try:
            spotprice = options_chain.get('underlyingPrice', None)
            if spotprice is None:
                raise ValueError("underlyingPrice not found in options_chain")
            calls = options_chain.get('callExpDateMap', {})
            puts = options_chain.get('putExpDateMap', {})
                        
            call_data = [option for exp_date in calls for strike in calls[exp_date] for option in calls[exp_date][strike]]
            put_data = [option for exp_date in puts for strike in puts[exp_date] for option in puts[exp_date][strike]]
            
            
            calls_df = pd.DataFrame(call_data)
            puts_df = pd.DataFrame(put_data)
        
            calls_df['spotprice'] = spotprice
            puts_df['spotprice'] = spotprice
            options_df = pd.concat([calls_df, puts_df], ignore_index=True)
            
            # Convert and verify data types
            options_df['strike'] = pd.to_numeric(options_df['strikePrice'])
            options_df['openInterest'] = pd.to_numeric(options_df['openInterest'])
            options_df['volatility'] = pd.to_numeric(options_df['volatility'])
            options_df['gamma'] = pd.to_numeric(options_df['gamma'])
            options_df['daysToExpiration'] = pd.to_numeric(options_df['daysToExpiration'])
                        
            return options_df, spotprice
        except Exception as e:
            print(f'Could not convert options chain to DataFrame: {e}')
            return pd.DataFrame(), None

    import pandas as pd
from datetime import datetime, timedelta

def naive_gamma(quote, options, spotPrice, fromStrike, toStrike):
    # Reset index and ensure we have necessary columns
    df = options.reset_index(drop=True).copy()
    
    # Get unique strike prices
    unique_strikes = sorted(df['strike'].unique())
    
    # Initialize arrays for gamma values
    totalGamma = []
    
    # Process each strike level
    for strike in unique_strikes:
        # Calculate gamma at this strike price
        calls = df[(df['strike'] == strike) & (df['putCall'] == 'CALL')]
        puts = df[(df['strike'] == strike) & (df['putCall'] == 'PUT')]
        
        # Sum the gamma values
        call_gamma_sum = (calls['gamma'] * calls['openInterest'] * 100 * spotPrice * spotPrice * 0.01 / 1_000_000).sum()
        put_gamma_sum = (puts['gamma'] * puts['openInterest'] * 100 * spotPrice * spotPrice * 0.01 * -1 / 1_000_000).sum()
        total_gamma = call_gamma_sum + put_gamma_sum
        #print(f"Strike: {strike}, Call Gamma: {calls['gamma'].sum()}, Call Open Interest: {calls['openInterest'].sum()}, Call Gex: {call_gamma_sum}, Put Gamma: {put_gamma_sum}, Total Gamma: {total_gamma}")
        
        totalGamma.append(total_gamma)
    
    # Print debug info
    #print(f"Total gamma sum: {sum(totalGamma)}")
    
    # Convert to billions
    totalGamma = np.array(totalGamma) #/ 10**9
    
    return quote, unique_strikes, totalGamma

def plot_gamma_exposure(todayDate, quote, levels, totalGamma, spotPrice, fromStrike, toStrike):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    # Set grid with lower z-order to appear behind bars
    plt.grid(True, color='white', alpha=0.1, zorder=0)
    
    # Plot bars with conditional colors (green for positive, red for negative)
    bar_width = (levels[1] - levels[0]) if len(levels) > 1 else 1  # Calculate width based on strike spacing
    colors = ['green' if gamma >= 0 else 'red' for gamma in totalGamma]
    plt.bar(levels, totalGamma, width=bar_width * 0.8, color=colors, alpha=0.7)
    
    chartTitle = f"Gamma Exposure for {quote['symbol']}: {todayDate.strftime('%d %b %Y')}"
    plt.title(chartTitle, fontweight="bold", fontsize=20, color='white')
    plt.xlabel('Index Price', fontweight="bold", color='white')
    plt.ylabel('Gamma Exposure ($ millions/1% move)', fontweight="bold", color='white')
    plt.axvline(x=spotPrice, color='yellow', lw=1, label=f"{quote['symbol']} Spot: {spotPrice:,.0f}")
    plt.axhline(y=0, color='grey', lw=1)
    plt.xlim([fromStrike, toStrike])
    trans = ax.get_xaxis_transform()
    plt.legend(facecolor=background_color, edgecolor='white', fontsize=12, loc='upper left', framealpha=1, labelcolor='white')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=25))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.show()

def calculate_vanna(quote, options_df, spot_price, min_strike, max_strike):
    """
    Safe vanna calculation with extensive error handling and fallback mechanisms
    
    Args:
        quote (dict): Quote information with current price
        options_df (pd.DataFrame): Options chain data
        spot_price (float): Current spot price
        min_strike (float): Minimum strike to consider
        max_strike (float): Maximum strike to consider
    
    Returns:
        tuple: (total_vanna, strike_prices, vanna_values)
    """
    from scipy.stats import norm
    import numpy as np
    import pandas as pd
    import math
    
    # Validate input data
    if options_df is None or options_df.empty:
        print("Warning: Options dataframe is empty or None")
        return 0, [], []

    # Attempt to map columns if they have different names
    column_mapping = {
        'putCall': ['putCall', 'optionType', 'type'],
        'strike': ['strike', 'strikePrice', 'strikeValue'],
        'openInterest': ['openInterest', 'openInt', 'open_interest'],
        'impliedVolatility': ['impliedVolatility', 'iv', 'implied_volatility']
    }

    def find_column(possible_names):
        for name in possible_names:
            if name in options_df.columns:
                return name
        return None

    # Map columns
    put_call_col = find_column(column_mapping['putCall'])
    strike_col = find_column(column_mapping['strike'])
    oi_col = find_column(column_mapping['openInterest'])
    iv_col = find_column(column_mapping['impliedVolatility'])

    # Check mapped columns
    required_columns = [put_call_col, strike_col, oi_col]
    if any(col is None for col in required_columns):
        print("Missing critical columns:")
        print(f"Columns available: {list(options_df.columns)}")
        print(f"Put/Call Column: {put_call_col}")
        print(f"Strike Column: {strike_col}")
        print(f"Open Interest Column: {oi_col}")
        return 0, [], []

    # Prepare a working copy of the dataframe
    working_df = options_df.copy()

    # Add default implied volatility if missing
    if iv_col is None:
        print("No implied volatility column found. Using default 30% IV.")
        working_df['impliedVolatility'] = 0.3
        iv_col = 'impliedVolatility'

    # Prepare arrays for vanna calculation
    strikes = []
    vanna_values = []

    # Group by strike price to consolidate options
    grouped = working_df[
        (working_df[strike_col] >= min_strike) & 
        (working_df[strike_col] <= max_strike)
    ].groupby(strike_col)

    for strike, group in grouped:
        try:
            # Calculate vanna for each contract type at the strike
            call_group = group[group[put_call_col] == 'CALL']
            put_group = group[group[put_call_col] == 'PUT']

            def calculate_strike_vanna(sub_group, flag):
                if sub_group.empty:
                    return 0

                # Use default 0.3 if IV is missing or invalid
                sigma = sub_group[iv_col].fillna(0.3).mean()
                oi = sub_group[oi_col].sum()

                # Constrain volatility
                sigma = max(0.01, min(sigma, 2.0))
                
                # Time to expiry (default to 30 days)
                t = 30 / 365.0  # Default to 30 days

                # Black-Scholes vanna calculation
                d1 = (math.log(spot_price / strike) + (0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
                
                # Vanna calculation 
                vanna_val = -norm.pdf(d1) * d1 / sigma
                
                # Adjust for contract direction
                vanna_val *= flag
                
                # Scale by open interest and convert to millions
                vanna_dollars = vanna_val * oi * 100 * spot_price * 0.01 / 1_000_000
                
                return vanna_dollars

            # Calculate vanna for calls (positive) and puts (negative)
            call_vanna = calculate_strike_vanna(call_group, 1)
            put_vanna = calculate_strike_vanna(put_group, -1)

            # Combine vanna values
            total_strike_vanna = call_vanna + put_vanna

            # Store results
            strikes.append(strike)
            vanna_values.append(total_strike_vanna)

        except Exception as e:
            print(f"Error calculating vanna for strike {strike}: {e}")
            continue

    # Calculate total vanna
    total_vanna = sum(vanna_values) if vanna_values else 0

    # # Diagnostic information
    # print(f"Vanna Calculation Summary:")
    # print(f"  Total Strikes Processed: {len(strikes)}")
    # print(f"  Total Vanna: {total_vanna:.4f} million")
    # print(f"  Min Vanna: {min(vanna_values) if vanna_values else 0:.4f}")
    # print(f"  Max Vanna: {max(vanna_values) if vanna_values else 0:.4f}")

    return total_vanna, strikes, vanna_values
def vanna_safe_calculation(quote, options_df, spot_price, min_strike, max_strike):
    """
    Safe vanna calculation with extensive error handling and fallback mechanisms
    
    Args:
        quote (dict): Quote information with current price
        options_df (pd.DataFrame): Options chain data
        spot_price (float): Current spot price
        min_strike (float): Minimum strike to consider
        max_strike (float): Maximum strike to consider
    
    Returns:
        tuple: (total_vanna, strike_prices, vanna_values)
    """
    from scipy.stats import norm
    import numpy as np
    import pandas as pd
    import math
    
    # First, diagnose the dataframe
    diagnostic_info = diagnose_options_dataframe(options_df)
    print("Options DataFrame Diagnostic:")
    for key, value in diagnostic_info.items():
        print(f"{key}: {value}")
    
    # Validate input data
    if options_df is None or options_df.empty:
        print("Warning: Options dataframe is empty or None")
        return 0, [], []

    # Attempt to map columns if they have different names
    column_mapping = {
        'putCall': ['putCall', 'optionType', 'type'],
        'strike': ['strike', 'strikePrice', 'strikeValue'],
        'openInterest': ['openInterest', 'openInt', 'open_interest'],
        'impliedVolatility': ['impliedVolatility', 'iv', 'implied_volatility']
    }

    def find_column(possible_names):
        for name in possible_names:
            if name in options_df.columns:
                return name
        return None

    # Map columns
    put_call_col = find_column(column_mapping['putCall'])
    strike_col = find_column(column_mapping['strike'])
    oi_col = find_column(column_mapping['openInterest'])
    iv_col = find_column(column_mapping['impliedVolatility'])

    # Check mapped columns
    required_columns = [put_call_col, strike_col, oi_col]
    if any(col is None for col in required_columns):
        print("Missing critical columns:")
        print(f"Columns available: {list(options_df.columns)}")
        print(f"Put/Call Column: {put_call_col}")
        print(f"Strike Column: {strike_col}")
        print(f"Open Interest Column: {oi_col}")
        return 0, [], []

    # Prepare a working copy of the dataframe
    working_df = options_df.copy()

    # Add default implied volatility if missing
    if iv_col is None:
        print("No implied volatility column found. Using default 30% IV.")
        working_df['impliedVolatility'] = 0.3
        iv_col = 'impliedVolatility'

    # Prepare arrays for vanna calculation
    strikes = []
    vanna_values = []

    # Group by strike price to consolidate options
    grouped = working_df[
        (working_df[strike_col] >= min_strike) & 
        (working_df[strike_col] <= max_strike)
    ].groupby(strike_col)

    for strike, group in grouped:
        try:
            # Calculate vanna for each contract type at the strike
            call_group = group[group[put_call_col] == 'CALL']
            put_group = group[group[put_call_col] == 'PUT']

            def calculate_strike_vanna(sub_group, flag):
                if sub_group.empty:
                    return 0

                # Use default 0.3 if IV is missing or invalid
                sigma = sub_group[iv_col].fillna(0.3).mean()
                oi = sub_group[oi_col].sum()

                # Constrain volatility
                sigma = max(0.01, min(sigma, 2.0))
                
                # Time to expiry (default to 30 days)
                t = 30 / 365.0  # Default to 30 days

                # Black-Scholes vanna calculation
                d1 = (math.log(spot_price / strike) + (0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
                
                # Vanna calculation 
                vanna_val = -norm.pdf(d1) * d1 / sigma
                
                # Adjust for contract direction
                vanna_val *= flag
                
                # Scale by open interest and convert to millions
                vanna_dollars = vanna_val * oi * 100 * spot_price * 0.01 / 1_000_000
                
                return vanna_dollars

            # Calculate vanna for calls (positive) and puts (negative)
            call_vanna = calculate_strike_vanna(call_group, 1)
            put_vanna = calculate_strike_vanna(put_group, -1)

            # Combine vanna values
            total_strike_vanna = call_vanna + put_vanna

            # Store results
            strikes.append(strike)
            vanna_values.append(total_strike_vanna)

        except Exception as e:
            print(f"Error calculating vanna for strike {strike}: {e}")
            continue

    # Calculate total vanna
    total_vanna = sum(vanna_values) if vanna_values else 0

    # # Diagnostic information
    # print(f"Vanna Calculation Summary:")
    # print(f"  Total Strikes Processed: {len(strikes)}")
    # print(f"  Total Vanna: {total_vanna:.4f} million")
    # print(f"  Min Vanna: {min(vanna_values) if vanna_values else 0:.4f}")
    # print(f"  Max Vanna: {max(vanna_values) if vanna_values else 0:.4f}")

    return total_vanna, strikes, vanna_values
def calculate_charm(quote, options_df, spot_price, min_strike, max_strike):
    """
    Calculate charm exposure across strikes
    
    Args:
        quote (dict): Quote information with current price
        options_df (pd.DataFrame): Options chain data
        spot_price (float): Current spot price
        min_strike (float): Minimum strike to consider
        max_strike (float): Maximum strike to consider
    
    Returns:
        tuple: (total_charm, strike_prices, charm_values)
    """
    from scipy.stats import norm
    import numpy as np
    import pandas as pd
    import math
    
    # Validate input data
    if options_df is None or options_df.empty:
        print("Warning: Options dataframe is empty or None")
        return 0, [], []

    # Attempt to map columns if they have different names
    column_mapping = {
        'putCall': ['putCall', 'optionType', 'type'],
        'strike': ['strike', 'strikePrice', 'strikeValue'],
        'openInterest': ['openInterest', 'openInt', 'open_interest'],
        'impliedVolatility': ['impliedVolatility', 'iv', 'implied_volatility'],
        'daysToExpiration': ['daysToExpiration', 'dte', 'days_to_expiry', 'expiryDays']
    }

    def find_column(possible_names):
        for name in possible_names:
            if name in options_df.columns:
                return name
        return None

    # Map columns
    put_call_col = find_column(column_mapping['putCall'])
    strike_col = find_column(column_mapping['strike'])
    oi_col = find_column(column_mapping['openInterest'])
    iv_col = find_column(column_mapping['impliedVolatility'])
    dte_col = find_column(column_mapping['daysToExpiration'])

    # Check mapped columns
    required_columns = [put_call_col, strike_col, oi_col]
    if any(col is None for col in required_columns):
        print("Missing critical columns for charm calculation")
        return 0, [], []

    # Prepare a working copy of the dataframe
    working_df = options_df.copy()

    # Add default implied volatility if missing
    if iv_col is None:
        print("No implied volatility column found. Using default 30% IV.")
        working_df['impliedVolatility'] = 0.3
        iv_col = 'impliedVolatility'
    
    # Set default DTE if missing
    if dte_col is None:
        print("No DTE column found. Using default 7 days.")
        working_df['daysToExpiration'] = 7
        dte_col = 'daysToExpiration'

    # Prepare arrays for charm calculation
    strikes = []
    charm_values = []

    # Group by strike price to consolidate options
    grouped = working_df[
        (working_df[strike_col] >= min_strike) & 
        (working_df[strike_col] <= max_strike)
    ].groupby(strike_col)

    for strike, group in grouped:
        try:
            # Calculate charm for each contract type at the strike
            call_group = group[group[put_call_col] == 'CALL']
            put_group = group[group[put_call_col] == 'PUT']

            def calculate_strike_charm(sub_group, flag):
                if sub_group.empty:
                    return 0

                # Use default 0.3 if IV is missing or invalid
                sigma = sub_group[iv_col].fillna(0.3).mean()
                oi = sub_group[oi_col].sum()
                
                # Get DTE, default to 7 if not available
                dte = sub_group[dte_col].fillna(7).mean()
                
                # Convert DTE to years
                t = max(dte / 365.0, 0.001)  # Minimum of 0.001 to avoid division by zero

                # Constrain volatility
                sigma = max(0.01, min(sigma, 2.0))
                
                # Black-Scholes charm calculation
                d1 = (math.log(spot_price / strike) + (0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
                d2 = d1 - sigma * math.sqrt(t)
                
                # Charm calculation (dDelta/dt) - simplified formula
                charm_val = -norm.pdf(d1) * (2 * (0.5 * sigma / math.sqrt(t)) - 
                                            ((0.5 * sigma**2) / (sigma * math.sqrt(t))))
                
                # Adjust for contract direction
                charm_val *= flag
                
                # Scale by open interest and convert to millions
                charm_dollars = charm_val * oi * 100 * spot_price * 0.01 / 1_000_000
                
                return charm_dollars

            # Calculate charm for calls (positive) and puts (negative)
            call_charm = calculate_strike_charm(call_group, 1)
            put_charm = calculate_strike_charm(put_group, -1)

            # Combine charm values
            total_strike_charm = call_charm + put_charm

            # Store results
            strikes.append(strike)
            charm_values.append(total_strike_charm)

        except Exception as e:
            print(f"Error calculating charm for strike {strike}: {e}")
            continue

    # Calculate total charm
    total_charm = sum(charm_values) if charm_values else 0

    return total_charm, strikes, charm_values

def calculate_delta_exposure(quote, options_df, spot_price, min_strike, max_strike):
    """
    Calculate delta exposure (DEx) across strikes
    
    Args:
        quote (dict): Quote information with current price
        options_df (pd.DataFrame): Options chain data
        spot_price (float): Current spot price
        min_strike (float): Minimum strike to consider
        max_strike (float): Maximum strike to consider
    
    Returns:
        tuple: (total_dex, strike_prices, dex_values)
    """
    import pandas as pd
    
    # Validate input data
    if options_df is None or options_df.empty:
        print("Warning: Options dataframe is empty or None")
        return 0, [], []

    # Attempt to map columns if they have different names
    column_mapping = {
        'putCall': ['putCall', 'optionType', 'type'],
        'strike': ['strike', 'strikePrice', 'strikeValue'],
        'openInterest': ['openInterest', 'openInt', 'open_interest'],
        'delta': ['delta', 'Delta']
    }

    def find_column(possible_names):
        for name in possible_names:
            if name in options_df.columns:
                return name
        return None

    # Map columns
    put_call_col = find_column(column_mapping['putCall'])
    strike_col = find_column(column_mapping['strike'])
    oi_col = find_column(column_mapping['openInterest'])
    delta_col = find_column(column_mapping['delta'])

    # Check mapped columns
    required_columns = [put_call_col, strike_col, oi_col]
    if any(col is None for col in required_columns):
        print("Missing critical columns for delta exposure calculation")
        return 0, [], []

    # Prepare a working copy of the dataframe
    working_df = options_df.copy()

    # If delta is missing, calculate it or use a placeholder
    if delta_col is None:
        print("No delta column found. Using a placeholder based on strike relative to spot.")
        # Simple delta approximation (not accurate but better than nothing)
        working_df['delta'] = working_df.apply(
            lambda row: max(0, min(1, 1 - (row[strike_col] - spot_price) / spot_price)) if row[put_call_col] == 'CALL' 
            else max(-1, min(0, -(row[strike_col] - spot_price) / spot_price)), 
            axis=1
        )
        delta_col = 'delta'

    # Prepare arrays for delta exposure calculation
    strikes = []
    dex_values = []

    # Group by strike price to consolidate options
    grouped = working_df[
        (working_df[strike_col] >= min_strike) & 
        (working_df[strike_col] <= max_strike)
    ].groupby(strike_col)

    for strike, group in grouped:
        try:
            call_group = group[group[put_call_col] == 'CALL']
            put_group = group[group[put_call_col] == 'PUT']
            
            # Calculate DEx for each option type
            def calculate_strike_dex(sub_group):
                if sub_group.empty:
                    return 0
                    
                # Calculate weighted delta
                delta_sum = (sub_group[delta_col] * sub_group[oi_col]).sum()
                
                # Scale by contract size and convert to millions
                dex_dollars = delta_sum * 100 * spot_price / 1_000_000
                
                return dex_dollars
            
            # Calculate DEx for calls and puts
            call_dex = calculate_strike_dex(call_group)
            put_dex = calculate_strike_dex(put_group)
            
            # Combine DEx values
            total_strike_dex = call_dex + put_dex
            
            # Store results
            strikes.append(strike)
            dex_values.append(total_strike_dex)
            
        except Exception as e:
            print(f"Error calculating delta exposure for strike {strike}: {e}")
            continue
    
    # Calculate total DEx
    total_dex = sum(dex_values) if dex_values else 0
    
    return total_dex, strikes, dex_values
def diagnose_options_dataframe(options_df):
    """
    Diagnose the structure and contents of the options dataframe
    
    Args:
        options_df (pd.DataFrame): Options chain dataframe to analyze
    
    Returns:
        dict: Diagnostic information about the dataframe
    """
    import pandas as pd
    
    # Check if dataframe is None or empty
    if options_df is None:
        return {
            "is_none": True,
            "error": "Dataframe is None"
        }
    
    if options_df.empty:
        return {
            "is_empty": True,
            "error": "Dataframe is empty"
        }
    
    # Get column information
    return {
        "total_columns": len(options_df.columns),
        "columns": list(options_df.columns),
        "sample_data": options_df.head().to_dict(),
        "column_types": {col: str(dtype) for col, dtype in options_df.dtypes.items()},
        "basic_stats": {
            "rows": len(options_df),
            "non_null_counts": options_df.count().to_dict()
        }
    }
def plot_index_gamma_report(quote, options, snapshot_time):
    spot_price = quote['current_price']
    print(f"Spot Price: {spot_price}")
    
    # Extract min and max strike - handle as numeric values
    min_strike = options['strike'].min()
    if isinstance(min_strike, pd.Series):
        fromStrike = float(min_strike.iloc[0])
    else:
        fromStrike = float(min_strike)
    print(f"fromStrike: {fromStrike}")

    max_strike = options['strike'].max()
    if isinstance(max_strike, pd.Series):
        toStrike = float(max_strike.iloc[0])
    else:
        toStrike = float(max_strike)
    print(f"toStrike: {toStrike}")

    gamma_params = naive_gamma(quote, options, spot_price, fromStrike, toStrike)
    #print(f"Gamma Params: {gamma_params}")
    plot_gamma_exposure(snapshot_time, *gamma_params, spot_price, fromStrike, toStrike)

if __name__ == '__main__':
    client = Client()
    
    if not os.path.exists(client.filepath):
        if client.setup():
            print("Setup completed successfully. Configuration saved.")
        else:
            print("Setup failed. Please check the logs for more details.")
            exit()

    symbol = 'SPY'  # Example symbol (no $ needed for Schwab API)
    fromDate = datetime(2025, 3, 24)  # The expiration date we want to analyze
    toDate = datetime(2025, 3, 24)  # The expiration date we want to analyze
    strike_count = 40

    # Fetch quote and options chain from Schwab API
    quote_data = client.get_Quote(symbol)
    options_chain = client.get_Option(symbol, fromDate, toDate, strike_count)

    if quote_data and options_chain:
        # Process quote data
        quote = {
            'symbol': symbol,
            'current_price': quote_data[symbol]['quote']['lastPrice']
        }

        # Process options chain data
        options_df, spotprice = client.options_chain_to_dataframe(options_chain)
        
        # Rename columns to match expected fields
        options_df.drop(columns=['strikePrice'], inplace=True)

        # Rename other columns if needed
        options_df = options_df.rename(columns={
            'putCall': 'putCall',
            'volatility': 'volatility',  # Schwab returns this as a percentage
            'openInterest': 'openInterest',
            'expirationDate': 'expirationDate',
            'daysToExpiration': 'daysToExpiration',
            'gamma': 'gamma'
        })
        #print(f"Options DataFrame:\n{options_df.head()}")

        # Use expiration_date instead of current time for the snapshot
        snapshot_time = fromDate
        plot_index_gamma_report(quote, options_df, snapshot_time)