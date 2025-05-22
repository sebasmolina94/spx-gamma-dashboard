import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import seaborn as sns

class OptionsAnalyzer:
    """
    A class for analyzing options data using yfinance
    with a focus on implied volatility walls and key trading metrics
    """
    
    def __init__(self, ticker):
        """Initialize with a ticker symbol"""
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.stock_price = self.stock.history(period="1d")['Close'][-1]
        self.expirations = self.stock.options
        print(f"Loaded {ticker} with current price: ${self.stock_price:.2f}")
        print(f"Available expiration dates: {len(self.expirations)}")
        
    def get_options_chain(self, expiry_idx=0):
        """Get options chain for a specific expiration date"""
        if expiry_idx >= len(self.expirations):
            print("Expiration index out of range, using first available")
            expiry_idx = 0
            
        self.current_expiry = self.expirations[expiry_idx]
        print(f"Analyzing options for expiration: {self.current_expiry}")
        
        options = self.stock.option_chain(self.current_expiry)
        self.calls = options.calls.sort_values('strike')
        self.puts = options.puts.sort_values('strike')
        
        # Calculate IV differences between adjacent strikes
        self.calls['iv_diff'] = self.calls['impliedVolatility'].diff()
        self.puts['iv_diff'] = self.puts['impliedVolatility'].diff()
        
        # Calculate distance from current price
        self.calls['price_distance'] = (self.calls['strike'] - self.stock_price) / self.stock_price
        self.puts['price_distance'] = (self.puts['strike'] - self.stock_price) / self.stock_price
        
        return self.calls, self.puts
    
    def find_iv_walls(self, threshold=0.05):
        """Find IV walls based on a threshold of IV change"""
        if not hasattr(self, 'calls') or not hasattr(self, 'puts'):
            self.get_options_chain()
            
        self.iv_wall_threshold = threshold
        self.call_iv_walls = self.calls[abs(self.calls['iv_diff']) > threshold]
        self.put_iv_walls = self.puts[abs(self.puts['iv_diff']) > threshold]
        
        print(f"\nFound {len(self.call_iv_walls)} Call IV walls:")
        if not self.call_iv_walls.empty:
            for _, wall in self.call_iv_walls.iterrows():
                print(f"  Strike: ${wall['strike']:.2f}, IV: {wall['impliedVolatility']:.2f}, IV Change: {wall['iv_diff']:.2f}")
        
        print(f"\nFound {len(self.put_iv_walls)} Put IV walls:")
        if not self.put_iv_walls.empty:
            for _, wall in self.put_iv_walls.iterrows():
                print(f"  Strike: ${wall['strike']:.2f}, IV: {wall['impliedVolatility']:.2f}, IV Change: {wall['iv_diff']:.2f}")
                
        return self.call_iv_walls, self.put_iv_walls
    
    def plot_iv_surface(self):
        """Plot the implied volatility surface"""
        if not hasattr(self, 'calls') or not hasattr(self, 'puts'):
            self.get_options_chain()
            
        plt.figure(figsize=(14, 8))
        plt.plot(self.calls['strike'], self.calls['impliedVolatility'], 'b-', label='Calls IV')
        plt.plot(self.puts['strike'], self.puts['impliedVolatility'], 'r-', label='Puts IV')
        
        # Add vertical line for current stock price
        plt.axvline(x=self.stock_price, color='k', linestyle='--', label=f'Current Price (${self.stock_price:.2f})')
        
        # Highlight IV walls if we've found them
        if hasattr(self, 'call_iv_walls') and not self.call_iv_walls.empty:
            plt.scatter(self.call_iv_walls['strike'], self.call_iv_walls['impliedVolatility'], 
                        color='blue', s=80, alpha=0.7, label='Call IV Walls')
            
        if hasattr(self, 'put_iv_walls') and not self.put_iv_walls.empty:
            plt.scatter(self.put_iv_walls['strike'], self.put_iv_walls['impliedVolatility'], 
                        color='red', s=80, alpha=0.7, label='Put IV Walls')
        
        plt.title(f'{self.ticker} Implied Volatility Surface - {self.current_expiry}')
        plt.xlabel('Strike Price ($)')
        plt.ylabel('Implied Volatility')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def calculate_trading_metrics(self):
        """Calculate important trading metrics from options data"""
        if not hasattr(self, 'calls') or not hasattr(self, 'puts'):
            self.get_options_chain()
            
        # 1. Volume-weighted IV
        vw_call_iv = np.average(self.calls['impliedVolatility'], weights=self.calls['volume'])
        vw_put_iv = np.average(self.puts['impliedVolatility'], weights=self.puts['volume'])
        
        # 2. Put/Call ratio based on volume and open interest
        put_call_volume_ratio = self.puts['volume'].sum() / max(1, self.calls['volume'].sum())
        put_call_oi_ratio = self.puts['openInterest'].sum() / max(1, self.calls['openInterest'].sum())
        
        # 3. Strikes with highest volume and open interest
        high_volume_calls = self.calls.nlargest(5, 'volume')
        high_volume_puts = self.puts.nlargest(5, 'volume')
        high_oi_calls = self.calls.nlargest(5, 'openInterest')
        high_oi_puts = self.puts.nlargest(5, 'openInterest')
        
        # 4. ATM IV (closest to current price)
        atm_call = self.calls.iloc[(self.calls['strike'] - self.stock_price).abs().argsort()[:1]]
        atm_put = self.puts.iloc[(self.puts['strike'] - self.stock_price).abs().argsort()[:1]]
        atm_call_iv = atm_call['impliedVolatility'].values[0] if len(atm_call) > 0 else 0
        atm_put_iv = atm_put['impliedVolatility'].values[0] if len(atm_put) > 0 else 0
        
        # 5. IV Skew (25-delta call IV minus 25-delta put IV)
        # This is a simplification - typically would find actual 25-delta options
        otm_calls = self.calls[self.calls['strike'] > self.stock_price]
        otm_puts = self.puts[self.puts['strike'] < self.stock_price]
        skew_measure = vw_put_iv - vw_call_iv if len(otm_calls) > 0 and len(otm_puts) > 0 else 0
        
        metrics = {
            'volume_weighted_call_iv': vw_call_iv,
            'volume_weighted_put_iv': vw_put_iv,
            'put_call_volume_ratio': put_call_volume_ratio,
            'put_call_oi_ratio': put_call_oi_ratio,
            'highest_volume_call_strikes': high_volume_calls['strike'].tolist(),
            'highest_volume_put_strikes': high_volume_puts['strike'].tolist(),
            'highest_oi_call_strikes': high_oi_calls['strike'].tolist(),
            'highest_oi_put_strikes': high_oi_puts['strike'].tolist(),
            'atm_call_iv': atm_call_iv,
            'atm_put_iv': atm_put_iv,
            'iv_skew': skew_measure
        }
        
        self.metrics = metrics
        
        # Print metrics
        print("\n=== Options Trading Metrics ===")
        print(f"Volume-weighted Call IV: {vw_call_iv:.4f}")
        print(f"Volume-weighted Put IV: {vw_put_iv:.4f}")
        print(f"ATM Call IV: {atm_call_iv:.4f}")
        print(f"ATM Put IV: {atm_put_iv:.4f}")
        print(f"IV Skew (Put IV - Call IV): {skew_measure:.4f}")
        print(f"Put/Call Volume Ratio: {put_call_volume_ratio:.2f}")
        print(f"Put/Call Open Interest Ratio: {put_call_oi_ratio:.2f}")
        print(f"Top 5 Call Strike Prices by Volume: {', '.join([f'${x:.2f}' for x in high_volume_calls['strike'].tolist()])}")
        print(f"Top 5 Put Strike Prices by Volume: {', '.join([f'${x:.2f}' for x in high_volume_puts['strike'].tolist()])}")
        
        return metrics
    
    def plot_volume_profile(self):
        """Plot option volume and open interest by strike"""
        if not hasattr(self, 'calls') or not hasattr(self, 'puts'):
            self.get_options_chain()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot volume
        ax1.bar(self.calls['strike'], self.calls['volume'], width=1, alpha=0.6, color='green', label='Calls Volume')
        ax1.bar(self.puts['strike'], self.puts['volume'], width=1, alpha=0.6, color='red', label='Puts Volume')
        ax1.set_title(f'{self.ticker} Options Volume by Strike - {self.current_expiry}')
        ax1.set_ylabel('Volume')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=self.stock_price, color='k', linestyle='--', label=f'Current Price (${self.stock_price:.2f})')
        ax1.legend()
        
        # Plot open interest
        ax2.bar(self.calls['strike'], self.calls['openInterest'], width=1, alpha=0.6, color='green', label='Calls OI')
        ax2.bar(self.puts['strike'], self.puts['openInterest'], width=1, alpha=0.6, color='red', label='Puts OI')
        ax2.set_title(f'{self.ticker} Options Open Interest by Strike - {self.current_expiry}')
        ax2.set_xlabel('Strike Price ($)')
        ax2.set_ylabel('Open Interest')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.stock_price, color='k', linestyle='--', label=f'Current Price (${self.stock_price:.2f})')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_iv_term_structure(self, num_expirations=4):
        """Analyze IV term structure across different expirations"""
        # Limit to the specified number of expirations
        expirations_to_analyze = self.expirations[:min(num_expirations, len(self.expirations))]
        
        term_structure = []
        expiry_dates = []
        
        print("\n=== IV Term Structure Analysis ===")
        for expiry in expirations_to_analyze:
            options = self.stock.option_chain(expiry)
            calls = options.calls
            puts = options.puts
            
            # Find ATM options
            atm_call = calls.iloc[(calls['strike'] - self.stock_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - self.stock_price).abs().argsort()[:1]]
            
            if len(atm_call) > 0 and len(atm_put) > 0:
                atm_call_iv = atm_call['impliedVolatility'].values[0]
                atm_put_iv = atm_put['impliedVolatility'].values[0]
                avg_atm_iv = (atm_call_iv + atm_put_iv) / 2
                term_structure.append(avg_atm_iv)
                
                # Convert expiry string to date object
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                expiry_dates.append(expiry_date)
                
                print(f"Expiry {expiry}: ATM IV = {avg_atm_iv:.4f}")
        
        # Plot term structure
        if term_structure:
            plt.figure(figsize=(12, 6))
            plt.plot(expiry_dates, term_structure, 'o-', linewidth=2)
            plt.title(f'{self.ticker} IV Term Structure')
            plt.xlabel('Expiration Date')
            plt.ylabel('ATM Implied Volatility')
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
    def plot_iv_skew(self):
        """Plot IV skew relative to strike price distance from current price"""
        if not hasattr(self, 'calls') or not hasattr(self, 'puts'):
            self.get_options_chain()
            
        plt.figure(figsize=(12, 6))
        
        # Filter to reasonable ranges (e.g., within ±30% of current price)
        calls_filtered = self.calls[(self.calls['price_distance'] > -0.3) & (self.calls['price_distance'] < 0.3)]
        puts_filtered = self.puts[(self.puts['price_distance'] > -0.3) & (self.puts['price_distance'] < 0.3)]
        
        plt.scatter(calls_filtered['price_distance'], calls_filtered['impliedVolatility'], 
                   label='Calls', color='green', alpha=0.7)
        plt.scatter(puts_filtered['price_distance'], puts_filtered['impliedVolatility'], 
                   label='Puts', color='red', alpha=0.7)
        
        # Add trend lines
        if len(calls_filtered) > 1:
            z = np.polyfit(calls_filtered['price_distance'], calls_filtered['impliedVolatility'], 1)
            p = np.poly1d(z)
            plt.plot(calls_filtered['price_distance'], p(calls_filtered['price_distance']), 
                    "g--", alpha=0.8, linewidth=1)
            
        if len(puts_filtered) > 1:
            z = np.polyfit(puts_filtered['price_distance'], puts_filtered['impliedVolatility'], 1)
            p = np.poly1d(z)
            plt.plot(puts_filtered['price_distance'], p(puts_filtered['price_distance']), 
                    "r--", alpha=0.8, linewidth=1)
        
        plt.axvline(x=0, color='k', linestyle='--', label='Current Price')
        plt.title(f'{self.ticker} IV Skew - {self.current_expiry}')
        plt.xlabel('Distance from Current Price (% of price)')
        plt.ylabel('Implied Volatility')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self, save_to_file=False):
        """Generate a comprehensive summary report of the analysis"""
        if not hasattr(self, 'metrics'):
            self.calculate_trading_metrics()
            
        report = f"""
        ===============================================
        OPTIONS ANALYSIS REPORT FOR {self.ticker}
        ===============================================
        
        Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Current Stock Price: ${self.stock_price:.2f}
        Options Expiration: {self.current_expiry}
        
        --- IV WALLS ---
        Call IV Walls: {len(self.call_iv_walls)} walls found (threshold: {self.iv_wall_threshold})
        {'Wall Strikes: ' + ', '.join([f'${x:.2f}' for x in self.call_iv_walls['strike'].tolist()]) if not self.call_iv_walls.empty else 'No walls detected'}
        
        Put IV Walls: {len(self.put_iv_walls)} walls found (threshold: {self.iv_wall_threshold})
        {'Wall Strikes: ' + ', '.join([f'${x:.2f}' for x in self.put_iv_walls['strike'].tolist()]) if not self.put_iv_walls.empty else 'No walls detected'}
        
        --- OPTIONS METRICS ---
        Volume-weighted Call IV: {self.metrics['volume_weighted_call_iv']:.4f}
        Volume-weighted Put IV: {self.metrics['volume_weighted_put_iv']:.4f}
        ATM Call IV: {self.metrics['atm_call_iv']:.4f}
        ATM Put IV: {self.metrics['atm_put_iv']:.4f}
        IV Skew (Put - Call): {self.metrics['iv_skew']:.4f}
        Put/Call Volume Ratio: {self.metrics['put_call_volume_ratio']:.2f}
        Put/Call Open Interest Ratio: {self.metrics['put_call_oi_ratio']:.2f}
        
        --- KEY LEVELS ---
        Top Call Strikes by Volume: {', '.join([f'${x:.2f}' for x in self.metrics['highest_volume_call_strikes']])}
        Top Put Strikes by Volume: {', '.join([f'${x:.2f}' for x in self.metrics['highest_volume_put_strikes']])}
        Top Call Strikes by OI: {', '.join([f'${x:.2f}' for x in self.metrics['highest_oi_call_strikes']])}
        Top Put Strikes by OI: {', '.join([f'${x:.2f}' for x in self.metrics['highest_oi_put_strikes']])}
        
        --- TRADING INSIGHTS ---
        {self._generate_trading_insights()}
        
        ===============================================
        """
        
        print(report)
        
        if save_to_file:
            filename = f"{self.ticker}_options_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Report saved to {filename}")
            
        return report
    
    def _generate_trading_insights(self):
        """Generate trading insights based on the analysis"""
        insights = []
        
        # Check put/call ratio for sentiment
        if self.metrics['put_call_volume_ratio'] > 1.5:
            insights.append("- High put/call ratio indicates bearish sentiment or potential hedging activity")
        elif self.metrics['put_call_volume_ratio'] < 0.7:
            insights.append("- Low put/call ratio suggests bullish sentiment")
            
        # Check IV skew
        if self.metrics['iv_skew'] > 0.05:
            insights.append("- Steep IV skew indicates market is pricing in downside risk")
        elif self.metrics['iv_skew'] < -0.05:
            insights.append("- Negative IV skew suggests market is pricing in upside potential")
            
        # Volume and OI insights
        call_max_volume_strike = self.metrics['highest_volume_call_strikes'][0] if self.metrics['highest_volume_call_strikes'] else 0
        put_max_volume_strike = self.metrics['highest_volume_put_strikes'][0] if self.metrics['highest_volume_put_strikes'] else 0
        
        if call_max_volume_strike > self.stock_price:
            insights.append(f"- High call volume at ${call_max_volume_strike:.2f} may indicate a resistance level or upside target")
            
        if put_max_volume_strike < self.stock_price:
            insights.append(f"- High put volume at ${put_max_volume_strike:.2f} may indicate a support level or downside target")
            
        # IV wall insights
        if not self.call_iv_walls.empty:
            insights.append(f"- Call IV walls may represent key resistance levels worth monitoring")
            
        if not self.put_iv_walls.empty:
            insights.append(f"- Put IV walls may represent key support levels worth monitoring")
            
        if not insights:
            insights.append("- No significant options-based signals detected with current parameters")
            
        return '\n'.join(insights)


def main():
    """Main function to demonstrate the OptionsAnalyzer class"""
    print("Options Analysis Tool using yfinance")
    print("====================================")
    
    # Default ticker
    default_ticker = "SPY"
    
    # Get ticker from user input or use default
    ticker_input = input(f"Enter ticker symbol (default: {default_ticker}): ").strip().upper()
    ticker = ticker_input if ticker_input else default_ticker
    
    try:
        # Create analyzer instance
        analyzer = OptionsAnalyzer(ticker)
        
        # Get options chain (default to first expiration)
        calls, puts = analyzer.get_options_chain()
        
        # Find IV walls
        analyzer.find_iv_walls(threshold=0.05)
        
        # Calculate trading metrics
        analyzer.calculate_trading_metrics()
        
        # Plot visualizations
        analyzer.plot_iv_surface()
        analyzer.plot_volume_profile()
        analyzer.plot_iv_skew()
        analyzer.analyze_iv_term_structure()
        
        # Generate summary report
        analyzer.generate_summary_report(save_to_file=True)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        

if __name__ == "__main__":
    main()