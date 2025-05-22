import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

class OptionsFlowAnalyzer:
    """
    A class to analyze options flow data and identify key support/resistance levels
    with enhanced dark mode visualizations
    """
    
    def __init__(self, symbol="SPY", risk_free_rate=0.05, dark_mode=True):
        """Initialize with ticker and params"""
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        self.current_price = None
        self.exp_date = None
        self.dark_mode = dark_mode
        
        # Set color scheme based on dark mode
        self.setup_colors()
    
    def setup_colors(self):
        """Setup color scheme based on dark mode preference"""
        if self.dark_mode:
            self.bg_color = '#121212'
            self.grid_color = '#333333'
            self.text_color = '#E0E0E0'
            self.accent_color = '#BB86FC'  # Purple accent
            
            # Define color palette
            self.call_color = '#4CAF50'  # Green
            self.put_color = '#F44336'   # Red
            self.gamma_color = '#2196F3' # Blue
            self.delta_color = '#4CAF50' # Green
            self.vanna_color = '#BB86FC' # Purple
            self.charm_color = '#FF9800' # Orange
            
            # Support and resistance colors
            self.support_color = '#4CAF50'     # Green
            self.resistance_color = '#F44336'  # Red
            self.current_line_color = '#FFFFFF'  # White
            self.zero_line_color = '#FFEB3B'   # Yellow
        else:
            self.bg_color = 'white'
            self.grid_color = '#CCCCCC'
            self.text_color = 'black'
            self.accent_color = '#673AB7'
            
            # Define color palette
            self.call_color = 'green'
            self.put_color = 'red'
            self.gamma_color = 'blue'
            self.delta_color = 'green'
            self.vanna_color = 'purple'
            self.charm_color = 'orange'
            
            # Support and resistance colors
            self.support_color = 'green'
            self.resistance_color = 'red'
            self.current_line_color = 'black'
            self.zero_line_color = '#FFCC00'

    def fetch_data(self, exp_date_index=0):
        """Fetch and process options data"""
        try:
            # Get ticker data
            ticker = yf.Ticker(self.symbol)
            self.current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Get expiration dates
            expiration_dates = ticker.options
            if not expiration_dates:
                print("No options data found.")
                return None
            
            # Select expiration date
            self.exp_date = expiration_dates[exp_date_index]
            options_chain = ticker.option_chain(self.exp_date)
            
            # Calculate time to expiry
            exp_date_obj = datetime.strptime(self.exp_date, '%Y-%m-%d').date()
            days_to_expiry = max(1, (exp_date_obj - datetime.today().date()).days)
            T = days_to_expiry / 365.0  # Time to expiry in years
            
            # Process options data and calculate Greeks
            df = self._process_options_chain(options_chain, T)
            return df
            
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return None
    
    def _black_scholes_greeks(self, S, K, T, r, sigma, option_type="call"):
        """Calculate all option Greeks using Black-Scholes formulas"""
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Basic Greeks
        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
            
        # Shared Greeks
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01
        vanna = -norm.pdf(d1) * d2 / sigma
        
        # Charm (delta decay)
        charm_factor = -norm.pdf(d1) * (
            (2 * (r + 0.5 * sigma**2) * T - d2 * sigma * np.sqrt(T)) / 
            (2 * T * sigma * np.sqrt(T))
        )
        if option_type == "put":
            charm_factor = -charm_factor
            
        return delta, gamma, vega, theta, vanna, charm_factor

    def _process_options_chain(self, options_chain, T):
        """Process options chain data and calculate Greeks"""
        # Extract calls and puts
        calls = options_chain.calls[['strike', 'openInterest', 'impliedVolatility', 'volume', 'lastPrice']].copy()
        puts = options_chain.puts[['strike', 'openInterest', 'impliedVolatility', 'volume', 'lastPrice']].copy()
        
        # Calculate Greeks for calls (vectorized when possible)
        S = self.current_price
        r = self.risk_free_rate
        
        # Process calls
        for idx, row in calls.iterrows():
            K = row['strike']
            sigma = max(0.01, min(2.0, row['impliedVolatility']))
            delta, gamma, vega, theta, vanna, charm = self._black_scholes_greeks(S, K, T, r, sigma, "call")
            
            calls.loc[idx, 'Delta_Calls'] = delta
            calls.loc[idx, 'Gamma_Calls'] = gamma
            calls.loc[idx, 'Vega_Calls'] = vega
            calls.loc[idx, 'Theta_Calls'] = theta
            calls.loc[idx, 'Vanna_Calls'] = vanna
            calls.loc[idx, 'Charm_Calls'] = charm
        
        # Process puts
        for idx, row in puts.iterrows():
            K = row['strike']
            sigma = max(0.01, min(2.0, row['impliedVolatility']))
            delta, gamma, vega, theta, vanna, charm = self._black_scholes_greeks(S, K, T, r, sigma, "put")
            
            puts.loc[idx, 'Delta_Puts'] = delta
            puts.loc[idx, 'Gamma_Puts'] = gamma
            puts.loc[idx, 'Vega_Puts'] = vega
            puts.loc[idx, 'Theta_Puts'] = theta
            puts.loc[idx, 'Vanna_Puts'] = vanna
            puts.loc[idx, 'Charm_Puts'] = charm
        
        # Rename basic columns
        calls.columns = ['Strike', 'OI_Calls', 'IV_Calls', 'Volume_Calls', 'Price_Calls', 
                        'Delta_Calls', 'Gamma_Calls', 'Vega_Calls', 'Theta_Calls', 'Vanna_Calls', 'Charm_Calls']
        puts.columns = ['Strike', 'OI_Puts', 'IV_Puts', 'Volume_Puts', 'Price_Puts',
                       'Delta_Puts', 'Gamma_Puts', 'Vega_Puts', 'Theta_Puts', 'Vanna_Puts', 'Charm_Puts']
        
        # Merge data
        df = pd.merge(calls, puts, on="Strike", how="outer").fillna(0)
        
        # Calculate all metrics in one go
        df["Call_Notional"] = df["OI_Calls"] * df["Price_Calls"] * 100
        df["Put_Notional"] = df["OI_Puts"] * df["Price_Puts"] * 100
        df["Net_Gamma"] = df["Gamma_Calls"] * df["OI_Calls"] - df["Gamma_Puts"] * df["OI_Puts"]
        df["Net_Delta"] = df["Delta_Calls"] * df["OI_Calls"] - df["Delta_Puts"] * df["OI_Puts"]
        df["Net_Vega"] = df["Vega_Calls"] * df["OI_Calls"] - df["Vega_Puts"] * df["OI_Puts"]
        df["Net_Theta"] = df["Theta_Calls"] * df["OI_Calls"] - df["Theta_Puts"] * df["OI_Puts"]
        df["Net_Vanna"] = df["Vanna_Calls"] * df["OI_Calls"] - df["Vanna_Puts"] * df["OI_Puts"]
        df["Net_Charm"] = df["Charm_Calls"] * df["OI_Calls"] - df["Charm_Puts"] * df["OI_Puts"]
        df["Net_OI"] = df["OI_Calls"] - df["OI_Puts"]
        df["Net_Volume"] = df["Volume_Calls"] - df["Volume_Puts"]
        df["Put_Call_Ratio"] = np.where(df["OI_Calls"] > 0, df["OI_Puts"] / df["OI_Calls"], 0)
        df["Dollar_Weighted_Gamma"] = df["Net_Gamma"] * df["Strike"] * self.current_price
        df["GEX"] = df["Net_Gamma"] * 10000  # Scale for readability
        df["DEX"] = df["Net_Delta"] * 100     # Delta exposure
        df["VEX"] = df["Net_Vanna"] * 100     # Vanna exposure
        df["Distance_From_Current"] = ((df["Strike"] / self.current_price) - 1) * 100
        
        return df

    def identify_key_levels(self, df):
        """Identify key support and resistance levels"""
        # Filter strikes within reasonable range
        min_strike = self.current_price * 0.80
        max_strike = self.current_price * 1.20
        filtered_df = df[(df["Strike"] >= min_strike) & (df["Strike"] <= max_strike)]
        
        # Initialize levels dictionary with optimized queries
        resistance_oi = filtered_df[filtered_df["Strike"] > self.current_price].nlargest(3, "Net_OI")
        support_oi = filtered_df[filtered_df["Strike"] < self.current_price].nlargest(3, "Net_OI")
        nearest_strikes = filtered_df.iloc[filtered_df["Strike"].sub(self.current_price).abs().argsort()[:5]]
        
        # Identify resistance (negative gamma) levels
        negative_gamma_df = filtered_df[filtered_df["GEX"] < 0]
        resistance_gamma = negative_gamma_df.nsmallest(3, "GEX") if not negative_gamma_df.empty else pd.DataFrame()
        
        # Identify support (positive gamma) levels
        positive_gamma_df = filtered_df[filtered_df["GEX"] > 0]
        support_gamma = positive_gamma_df.nlargest(3, "GEX") if not positive_gamma_df.empty else pd.DataFrame()
        
        # Dollar-weighted gamma levels
        resistance_dollar = filtered_df[filtered_df["Strike"] > self.current_price].nlargest(3, "Dollar_Weighted_Gamma")
        support_dollar = filtered_df[filtered_df["Strike"] < self.current_price].nlargest(3, "Dollar_Weighted_Gamma")
        
        # Create results structure
        levels = {
            "resistance": {
                "gamma": resistance_gamma,
                "oi": resistance_oi,
                "dollar_gamma": resistance_dollar
            },
            "support": {
                "gamma": support_gamma,
                "oi": support_oi,
                "dollar_gamma": support_dollar
            },
            "current": {
                "nearest_strikes": nearest_strikes,
                "overall_sentiment": "Bullish" if filtered_df["Net_Delta"].sum() > 0 else "Bearish"
            },
            "gex_summary": {
                "net_gamma": filtered_df["GEX"].sum(),
                "positive_gamma": positive_gamma_df["GEX"].sum() if not positive_gamma_df.empty else 0,
                "negative_gamma": negative_gamma_df["GEX"].sum() if not negative_gamma_df.empty else 0,
                "zero_gamma_line": self._find_zero_gamma_line(filtered_df)
            }
        }
        
        return levels
    
    def _find_zero_gamma_line(self, df):
        """Find the strike where gamma crosses from positive to negative or vice versa"""
        if df.empty:
            return self.current_price
        
        # Sort by strike and find sign changes
        sorted_df = df.sort_values("Strike")
        gamma_values = sorted_df["GEX"].values
        strikes = sorted_df["Strike"].values
        
        zero_gamma_strikes = []
        
        # Find zero crossings
        for i in range(1, len(gamma_values)):
            if (gamma_values[i-1] > 0 and gamma_values[i] < 0) or (gamma_values[i-1] < 0 and gamma_values[i] > 0):
                # Linear interpolation for zero crossing
                t = abs(gamma_values[i-1]) / (abs(gamma_values[i-1]) + abs(gamma_values[i]))
                zero_strike = strikes[i-1] + t * (strikes[i] - strikes[i-1])
                zero_gamma_strikes.append((zero_strike, abs(zero_strike - self.current_price)))
        
        # Return the zero crossing closest to current price
        if zero_gamma_strikes:
            return min(zero_gamma_strikes, key=lambda x: x[1])[0]
        return self.current_price

    def visualize_standard_flow(self, df):
        """Create standard visualizations of options flow"""
        # Apply dark mode style if selected
        if self.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
            
        # Filter strikes for visualization
        min_strike = self.current_price * 0.85
        max_strike = self.current_price * 1.15
        filtered_df = df[(df["Strike"] >= min_strike) & (df["Strike"] <= max_strike)]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 14))
        fig.patch.set_facecolor(self.bg_color)
        
        # Add a title with styled text
        fig.suptitle(f"{self.symbol} Options Flow Analysis - Expiry: {self.exp_date}", 
                    fontsize=18, color=self.text_color, fontweight='bold', y=0.98)
        
        # Create GridSpec for more flexible layout
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.5])
        
        # 1. Net Gamma Profile
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(self.bg_color)
        
        # Plot bars with gradient based on value
        bars = ax1.bar(filtered_df["Strike"], filtered_df["GEX"], 
                      alpha=0.8, width=1.0)
        
        # Color the bars based on positive or negative values
        for i, bar in enumerate(bars):
            if bar.get_height() >= 0:
                bar.set_color(self.support_color)
            else:
                bar.set_color(self.resistance_color)
        
        ax1.axvline(self.current_price, color=self.current_line_color, linestyle="--", 
                    linewidth=1.5, label=f"Current Price: ${self.current_price:.2f}")
        ax1.axhline(0, color=self.zero_line_color, linestyle="-", alpha=0.4)
        
        # Highlight significant gamma levels
        top_gamma = filtered_df.nlargest(3, "GEX")
        bottom_gamma = filtered_df.nsmallest(3, "GEX")
        
        for _, row in top_gamma.iterrows():
            ax1.annotate(f"${row['Strike']:.0f}", 
                        xy=(row['Strike'], row['GEX']),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center',
                        color=self.support_color,
                        fontweight="bold")
        
        for _, row in bottom_gamma.iterrows():
            ax1.annotate(f"${row['Strike']:.0f}", 
                        xy=(row['Strike'], row['GEX']),
                        xytext=(0, -15),
                        textcoords="offset points",
                        ha='center',
                        color=self.resistance_color,
                        fontweight="bold")
        
        # Style the axes
        ax1.set_title("GEX by Strike", fontsize=14, color=self.text_color, pad=10)
        ax1.set_xlabel("Strike Price", fontsize=12, color=self.text_color)
        ax1.set_ylabel("Gamma Exposure", fontsize=12, color=self.text_color)
        ax1.tick_params(colors=self.text_color)
        ax1.grid(True, alpha=0.2, color=self.grid_color)
        
        # Add legend
        ax1.legend(facecolor=self.bg_color, edgecolor=self.grid_color, 
                  labelcolor=self.text_color, framealpha=0.8)

        # 2. Open Interest Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(self.bg_color)
        
        ax2.bar(filtered_df["Strike"], filtered_df["OI_Calls"], 
               color=self.call_color, alpha=0.7, label="Calls OI", width=1.0)
        ax2.bar(filtered_df["Strike"], -filtered_df["OI_Puts"], 
               color=self.put_color, alpha=0.7, label="Puts OI", width=1.0)
        
        ax2.axvline(self.current_price, color=self.current_line_color, linestyle="--", 
                   linewidth=1.5, label=f"Current Price: ${self.current_price:.2f}")
        
        # Style the axes
        ax2.set_title("Call vs Put Open Interest", fontsize=14, color=self.text_color, pad=10)
        ax2.set_xlabel("Strike Price", fontsize=12, color=self.text_color)
        ax2.set_ylabel("Open Interest", fontsize=12, color=self.text_color)
        ax2.tick_params(colors=self.text_color)
        ax2.grid(True, alpha=0.2, color=self.grid_color)
        
        # Add legend
        ax2.legend(facecolor=self.bg_color, edgecolor=self.grid_color, 
                  labelcolor=self.text_color, framealpha=0.8)
        
        # 3. Greeks Combined Chart
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(self.bg_color)
        
        # Normalize data for comparison
        metrics = filtered_df[["Strike", "DEX", "GEX", "VEX", "Net_Charm"]].copy()
        for col in metrics.columns[1:]:
            max_abs = max(abs(metrics[col].max()), abs(metrics[col].min()))
            if max_abs > 0:
                metrics[col] = metrics[col] / max_abs
        
        # Plot lines with improved styling
        ax3.plot(metrics["Strike"], metrics["DEX"], label="Delta Exp", 
                color=self.delta_color, linestyle="-", linewidth=2)
        ax3.plot(metrics["Strike"], metrics["GEX"], label="Gamma Exp", 
                color=self.gamma_color, linestyle="--", linewidth=2)
        ax3.plot(metrics["Strike"], metrics["VEX"], label="Vanna Exp", 
                color=self.vanna_color, linestyle="-.", linewidth=2)
        ax3.plot(metrics["Strike"], metrics["Net_Charm"], label="Charm Exp", 
                color=self.charm_color, linestyle=":", linewidth=2)
        
        ax3.axvline(self.current_price, color=self.current_line_color, linestyle="--", 
                   linewidth=1.5, label=f"Current: ${self.current_price:.2f}")
        ax3.axhline(0, color=self.zero_line_color, linestyle="-", alpha=0.4)
        
        # Style the axes
        ax3.set_title("Normalized Greeks Profile", fontsize=14, color=self.text_color, pad=10)
        ax3.set_xlabel("Strike Price", fontsize=12, color=self.text_color)
        ax3.set_ylabel("Normalized Exposure", fontsize=12, color=self.text_color)
        ax3.tick_params(colors=self.text_color)
        ax3.grid(True, alpha=0.2, color=self.grid_color)
        
        # Add legend
        ax3.legend(facecolor=self.bg_color, edgecolor=self.grid_color, 
                  labelcolor=self.text_color, framealpha=0.8)

        # 4. GEX Heatmap 
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(self.bg_color)
        
        # Create a GEX heatmap based on strike and net gamma
        pivot_data = filtered_df[["Strike", "GEX"]].copy()
        
        # Normalize gamma for coloring
        max_abs_gamma = max(abs(pivot_data["GEX"].min()), abs(pivot_data["GEX"].max()))
        pivot_data["Normalized_Gamma"] = pivot_data["GEX"] / max_abs_gamma
        
        # Sort by strike
        pivot_data = pivot_data.sort_values("Strike")
        
        # Create colormap
        colors = []
        for gamma in pivot_data["Normalized_Gamma"]:
            if gamma > 0:
                # Green for positive gamma (support)
                intensity = min(1.0, abs(gamma) * 0.8)
                colors.append((0, intensity, 0))  # Green with varying intensity
            else:
                # Red for negative gamma (resistance)
                intensity = min(1.0, abs(gamma) * 0.8)
                colors.append((intensity, 0, 0))  # Red with varying intensity
        
        # Plot horizontal bars
        barh = ax4.barh(pivot_data["Strike"], pivot_data["GEX"], color=colors, height=0.6)
        ax4.axvline(x=0, color=self.zero_line_color, linestyle='dashed', linewidth=1, alpha=0.7)
        ax4.axhline(y=self.current_price, color=self.current_line_color, linestyle='dashed', 
                   linewidth=1.5, label=f"Current: ${self.current_price:.2f}")
        
        # Identify significant levels
        gamma_sum = filtered_df["GEX"].sum()
        top_support = filtered_df[filtered_df["GEX"] > 0].nlargest(3, "GEX")
        top_resistance = filtered_df[filtered_df["GEX"] < 0].nsmallest(3, "GEX")
        
        # Add annotations for significant levels
        for _, row in top_support.iterrows():
            ax4.annotate(f"S: ${row['Strike']:.0f}", 
                        xy=(row['GEX'], row['Strike']),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center',
                        color=self.support_color,
                        fontweight="bold")
        
        for _, row in top_resistance.iterrows():
            ax4.annotate(f"R: ${row['Strike']:.0f}", 
                        xy=(row['GEX'], row['Strike']),
                        xytext=(-5, 0),
                        textcoords="offset points",
                        ha='right', va='center',
                        color=self.resistance_color,
                        fontweight="bold")
        
        # Style the axes
        ax4.set_title(f"Gamma Exposure (GEX) - Net: {gamma_sum:.2f}", 
                     fontsize=14, color=self.text_color, pad=10)
        ax4.set_xlabel("Gamma Exposure", fontsize=12, color=self.text_color)
        ax4.set_ylabel("Strike Price", fontsize=12, color=self.text_color)
        ax4.tick_params(colors=self.text_color)
        ax4.grid(True, alpha=0.2, color=self.grid_color)
        
        # Add legend
        ax4.legend(facecolor=self.bg_color, edgecolor=self.grid_color, 
                  labelcolor=self.text_color, framealpha=0.8)

        # 5. Volume Profile (Calls vs Puts)
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_facecolor(self.bg_color)
        
        # Filter for strikes with significant volume
        volume_df = filtered_df[filtered_df["Volume_Calls"] + filtered_df["Volume_Puts"] > 0].copy()
        
        # Plot volume profile
        ax5.bar(volume_df["Strike"], volume_df["Volume_Calls"], 
               color=self.call_color, alpha=0.7, label="Calls Volume", width=1.0)
        ax5.bar(volume_df["Strike"], -volume_df["Volume_Puts"], 
               color=self.put_color, alpha=0.7, label="Puts Volume", width=1.0)
        
        ax5.axvline(self.current_price, color=self.current_line_color, linestyle="--", 
                   linewidth=1.5, label=f"Current: ${self.current_price:.2f}")
        
        # Style the axes
        ax5.set_title("Volume Profile", fontsize=14, color=self.text_color, pad=10)
        ax5.set_xlabel("Strike Price", fontsize=12, color=self.text_color)
        ax5.set_ylabel("Volume", fontsize=12, color=self.text_color)
        ax5.tick_params(colors=self.text_color)
        ax5.grid(True, alpha=0.2, color=self.grid_color)
        
        # Add legend
        ax5.legend(facecolor=self.bg_color, edgecolor=self.grid_color, 
                  labelcolor=self.text_color, framealpha=0.8)
        
        # 6. Put/Call Ratio by Strike
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_facecolor(self.bg_color)
        
        # Calculate PCR for strikes with non-zero call OI
        pcr_df = filtered_df[filtered_df["OI_Calls"] > 0].copy()
        pcr_df["PCR"] = pcr_df["OI_Puts"] / pcr_df["OI_Calls"]
        
        # Create color gradient based on PCR values
        pcr_colors = []
        for pcr in pcr_df["PCR"]:
            if pcr < 0.8:  # Bullish
                pcr_colors.append(self.call_color)
            elif pcr < 1.2:  # Neutral
                pcr_colors.append(self.zero_line_color)
            else:  # Bearish
                pcr_colors.append(self.put_color)
        
        # Plot PCR bars
        ax6.bar(pcr_df["Strike"], pcr_df["PCR"], color=pcr_colors, alpha=0.7, width=1.0)
        ax6.axvline(self.current_price, color=self.current_line_color, linestyle="--", 
                   linewidth=1.5, label=f"Current: ${self.current_price:.2f}")
        ax6.axhline(1.0, color=self.zero_line_color, linestyle="-", 
                   linewidth=1.5, label="PCR = 1.0")
        
        # Style the axes
        ax6.set_title("Put/Call Ratio by Strike", fontsize=14, color=self.text_color, pad=10)
        ax6.set_xlabel("Strike Price", fontsize=12, color=self.text_color)
        ax6.set_ylabel("Put/Call Ratio", fontsize=12, color=self.text_color)
        ax6.tick_params(colors=self.text_color)
        ax6.grid(True, alpha=0.2, color=self.grid_color)
        
        # Add legend
        ax6.legend(facecolor=self.bg_color, edgecolor=self.grid_color, 
                  labelcolor=self.text_color, framealpha=0.8)

        # Add text box with key metrics
        exp_date_obj = datetime.strptime(self.exp_date, '%Y-%m-%d').date()
        days_to_expiry = (exp_date_obj - datetime.today().date()).days
        
        summary_text = (
            f"Key Metrics for {self.symbol}:\n"
            f"Current Price: ${self.current_price:.2f}\n"
            f"Expiration: {self.exp_date} ({days_to_expiry} days)\n"
            f"Call OI: {filtered_df['OI_Calls'].sum():,.0f}\n"
            f"Put OI: {filtered_df['OI_Puts'].sum():,.0f}\n"
            f"P/C Ratio: {filtered_df['OI_Puts'].sum() / max(1, filtered_df['OI_Calls'].sum()):.2f}\n"
            f"Net GEX: {filtered_df['GEX'].sum():.2f}\n"
            f"Net DEX: {filtered_df['DEX'].sum():.2f}\n"
        )
        
        fig.text(0.02, 0.01, summary_text, fontsize=10, color=self.text_color,
                 bbox=dict(facecolor=self.bg_color, edgecolor=self.grid_color, alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Add settings for tight layout to prevent overlapping
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3, w_pad=3)
        
        return fig
    
    def visualize_advanced_flow(self, df):
        """Create advanced 3D visualizations of options flow"""
        # Apply dark mode style if selected
        if self.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
            
        # Filter strikes for visualization (wider range for heatmaps)
        min_strike = self.current_price * 0.80
        max_strike = self.current_price * 1.20
        filtered_df = df[(df["Strike"] >= min_strike) & (df["Strike"] <= max_strike)]
        
        # Create figure for advanced visualizations
        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor(self.bg_color)
        
        # Add title
        fig.suptitle(f"Advanced Options Flow Analysis: {self.symbol} - {self.exp_date}", 
                    fontsize=20, color=self.text_color, fontweight='bold', y=0.98)
        
        # 1. 3D Surface Plot for Gamma Landscape
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_facecolor(self.bg_color)
        
        # Create strike price mesh
        x = filtered_df["Strike"].values
        y = np.linspace(-10, 10, 20)  # % change in underlying
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Calculate gamma impact for each price point
        for i, price_pct in enumerate(y):
            estimated_price = self.current_price * (1 + price_pct/100)
            for j, strike in enumerate(x):
                # Simplified model: gamma impact decreases as distance from strike increases
                distance_factor = np.exp(-0.5 * ((strike - estimated_price) / (strike * 0.05))**2)
                gamma_at_strike = filtered_df[filtered_df["Strike"] == strike]["Net_Gamma"].values
                if len(gamma_at_strike) > 0:
                    Z[i, j] = gamma_at_strike[0] * distance_factor
                else:
                    Z[i, j] = 0
        
        # Plot the surface
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8,
                               linewidth=0, antialiased=True)
        
        # Mark current price
        ax1.plot([self.current_price] * len(y), y, [0] * len(y), 
                 color=self.current_line_color, linestyle='--', linewidth=2)
        
        # Style the axes
        ax1.set_title("Gamma Landscape", color=self.text_color, pad=20, fontsize=14)
        ax1.set_xlabel("Strike Price", color=self.text_color, labelpad=10)
        ax1.set_ylabel("Price Change (%)", color=self.text_color, labelpad=10)
        ax1.set_zlabel("Gamma Impact", color=self.text_color, labelpad=10)
        ax1.tick_params(colors=self.text_color)
        
        # Add a color bar
        cbar = fig.colorbar(surf, ax=ax1, shrink=0.7, pad=0.1)
        cbar.ax.tick_params(colors=self.text_color)
        cbar.set_label("Gamma Exposure", color=self.text_color)
        
        # 2. Options Chain Heatmap (IV Skew)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor(self.bg_color)
        
        # Create data for heatmap
        strikes = filtered_df["Strike"].unique()
        strikes.sort()
        
        # Prepare data for heatmap
        heatmap_data = []
        strike_labels = []
        
        for strike in strikes:
            strike_data = df[df["Strike"] == strike]
            if not strike_data.empty:
                call_iv = strike_data["IV_Calls"].values[0]
                put_iv = strike_data["IV_Puts"].values[0]
                iv_skew = call_iv - put_iv
                moneyness = (strike / self.current_price - 1) * 100  # % from current price
                
                heatmap_data.append([moneyness, iv_skew])
                strike_labels.append(f"${strike:.0f}")
        
        if heatmap_data:
            heatmap_data = np.array(heatmap_data)
            
            # Sort by moneyness
            idx = np.argsort(heatmap_data[:, 0])
            sorted_moneyness = heatmap_data[idx, 0]
            sorted_iv_skew = heatmap_data[idx, 1]
            sorted_labels = [strike_labels[i] for i in idx]
            
            # Create IV skew plot with color gradient
            scatter = ax2.scatter(sorted_moneyness, sorted_iv_skew, 
                                c=sorted_iv_skew, cmap='coolwarm', 
                                s=100, alpha=0.8, edgecolors='none')
            
            # Add strike labels
            for i, txt in enumerate(sorted_labels):
                if i % 5 == 0:  # Label every 5th point to avoid overcrowding
                    ax2.annotate(txt, (sorted_moneyness[i], sorted_iv_skew[i]),
                               xytext=(0, 5), textcoords="offset points",
                               ha='center', va='bottom', color=self.text_color, fontsize=8)
            
            # Add trend line
            z = np.polyfit(sorted_moneyness, sorted_iv_skew, 1)
            p = np.poly1d(z)
            ax2.plot(sorted_moneyness, p(sorted_moneyness), 
                    color=self.accent_color, linestyle='--', linewidth=2,
                    label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
            
            # Style the axes
            ax2.set_title("IV Skew (Call IV - Put IV)", color=self.text_color, pad=10, fontsize=14)
            ax2.set_xlabel("% From Current Price", color=self.text_color)
            ax2.set_ylabel("IV Skew", color=self.text_color)
            ax2.grid(True, alpha=0.2, color=self.grid_color)
            ax2.tick_params(colors=self.text_color)
            ax2.axvline(0, color=self.current_line_color, linestyle='--', linewidth=1)
            ax2.axhline(0, color=self.grid_color, linestyle='-', linewidth=1)
            ax2.legend(facecolor=self.bg_color, edgecolor=self.grid_color, labelcolor=self.text_color)
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax2)
            cbar.ax.tick_params(colors=self.text_color)
            cbar.set_label("IV Skew", color=self.text_color)
        
        # 3. Options Chain Visualization with Size = OI and Color = Put/Call
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor(self.bg_color)
        
        # Create custom colormap for P/C ratio
        colors = [(0, 0.7, 0), (0.7, 0.7, 0), (0.7, 0, 0)]  # Green -> Yellow -> Red
        pcr_cmap = LinearSegmentedColormap.from_list("pcr_colors", colors, N=100)
        
        # Prepare data
        strikes = []
        total_oi = []
        pcr_vals = []
        
        for idx, row in filtered_df.iterrows():
            if row["OI_Calls"] > 0 or row["OI_Puts"] > 0:
                strikes.append(row["Strike"])
                total_oi.append(row["OI_Calls"] + row["OI_Puts"])
                
                # Calculate P/C ratio (capped at 3 for visualization)
                if row["OI_Calls"] > 0:
                    pcr = min(3, row["OI_Puts"] / row["OI_Calls"])
                else:
                    pcr = 3  # Max bearish if no calls
                pcr_vals.append(pcr)
        
        # Normalize size for plotting
        max_oi = max(total_oi) if total_oi else 1
        norm_sizes = [100 + 2000 * (oi / max_oi) for oi in total_oi]
        
        # Plot bubbles
        scatter = ax3.scatter(strikes, [0] * len(strikes), s=norm_sizes, 
                             c=pcr_vals, cmap=pcr_cmap, alpha=0.7, 
                             edgecolors=self.text_color, linewidths=1)
        
        # Add strike price annotations
        for i, strike in enumerate(strikes):
            if i % 5 == 0 or total_oi[i] > 0.8 * max_oi:  # Label significant OI
                ax3.annotate(f"${strike:.0f}", (strike, 0), 
                           ha='center', va='center', color=self.text_color, 
                           fontweight='bold', fontsize=8)
        
        # Style the axes
        ax3.set_title("Options Chain Activity Map", color=self.text_color, pad=10, fontsize=14)
        ax3.set_xlabel("Strike Price", color=self.text_color)
        ax3.get_yaxis().set_visible(False)  # Hide y-axis
        ax3.axvline(self.current_price, color=self.current_line_color, 
                   linestyle='--', linewidth=2, 
                   label=f"Current: ${self.current_price:.2f}")
        ax3.grid(True, alpha=0.2, color=self.grid_color, axis='x')
        ax3.tick_params(colors=self.text_color)
        
        # Add legend for current price
        ax3.legend(facecolor=self.bg_color, edgecolor=self.grid_color, labelcolor=self.text_color)
        
        # Add colorbar for P/C ratio
        cbar = fig.colorbar(scatter, ax=ax3, orientation='horizontal', pad=0.2)
        cbar.ax.tick_params(colors=self.text_color)
        cbar.set_label("Put/Call Ratio", color=self.text_color)
        cbar.ax.set_xticks([0, 1, 2, 3])
        cbar.ax.set_xticklabels(['Bullish (0)', 'Neutral (1)', 'Bearish (2)', '≥3'])
        
        # 4. Theta Decay Heatmap
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor(self.bg_color)
        
        # Prepare theta data
        strikes = filtered_df["Strike"].values
        theta_values = filtered_df["Net_Theta"].values
        
        # Create colors based on theta values
        theta_colors = []
        for theta in theta_values:
            if theta > 0:  # Positive theta (usually market makers profit)
                intensity = min(1.0, abs(theta) / (max(abs(theta_values)) * 0.8))
                theta_colors.append((0, intensity, 0))  # Green
            else:  # Negative theta (usually option buyers lose)
                intensity = min(1.0, abs(theta) / (max(abs(theta_values)) * 0.8))
                theta_colors.append((intensity, 0, 0))  # Red
        
        # Create scatter plot with size proportional to absolute theta
        sizes = [100 + 1000 * (abs(t) / max(abs(theta_values))) for t in theta_values]
        scatter = ax4.scatter(strikes, [0] * len(strikes), s=sizes, 
                             c=theta_colors, alpha=0.7, 
                             edgecolors=self.text_color, linewidths=1)
        
        # Add vertical line for current price
        ax4.axvline(self.current_price, color=self.current_line_color, 
                   linestyle='--', linewidth=2, 
                   label=f"Current: ${self.current_price:.2f}")
        
        # Style the axes
        ax4.set_title("Theta Decay Profile", color=self.text_color, pad=10, fontsize=14)
        ax4.set_xlabel("Strike Price", color=self.text_color)
        ax4.get_yaxis().set_visible(False)  # Hide y-axis
        ax4.grid(True, alpha=0.2, color=self.grid_color, axis='x')
        ax4.tick_params(colors=self.text_color)
        
        # Add legend
        ax4.legend(facecolor=self.bg_color, edgecolor=self.grid_color, labelcolor=self.text_color)
        
        # Add text with theta summary
        total_theta = filtered_df["Net_Theta"].sum()
        theta_text = (
            f"Daily Theta Decay: ${total_theta:.2f}\n"
            f"Weekly Decay: ${total_theta * 5:.2f}\n"
            f"Monthly Decay: ${total_theta * 21:.2f}"
        )
        
        # Add theta summary text box
        ax4.text(0.05, 0.95, theta_text, transform=ax4.transAxes, 
                fontsize=10, color=self.text_color, 
                bbox=dict(facecolor=self.bg_color, edgecolor=self.grid_color, alpha=0.8, boxstyle='round,pad=0.5'),
                verticalalignment='top')
                
        # Add settings for tight layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3, w_pad=3)
        
        return fig
    
    def analyze(self, exp_date_index=0, advanced_visuals=False):
        """Run a complete analysis"""
        # Fetch data
        df = self.fetch_data(exp_date_index)
        if df is None:
            return {"error": "Failed to fetch options data"}
        
        # Generate visualizations
        standard_fig = self.visualize_standard_flow(df)
        
        # Generate advanced visualizations if requested
        advanced_fig = None
        if advanced_visuals:
            advanced_fig = self.visualize_advanced_flow(df)
        
        # Identify key levels
        levels = self.identify_key_levels(df)
        
        # Create result summary
        result = {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "expiration_date": self.exp_date,
            "call_put_ratio": df["OI_Calls"].sum() / max(df["OI_Puts"].sum(), 1),
            "put_call_ratio": df["OI_Puts"].sum() / max(df["OI_Calls"].sum(), 1),
            "call_premium_total": df["Call_Notional"].sum(),
            "put_premium_total": df["Put_Notional"].sum(),
            "premium_ratio": df["Call_Notional"].sum() / max(df["Put_Notional"].sum(), 1),
            "net_gamma": df["GEX"].sum(),
            "net_delta": df["DEX"].sum(),
            "net_vanna": df["VEX"].sum(),
            "key_levels": levels,
            "visualization": standard_fig,
            "advanced_visualization": advanced_fig,
            "data": df  # Keep data for reference
        }
        
        return result
    
    def display_results(self, analysis):
        """Display analysis results in a readable format"""
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        # Display summary information
        print(f"\n{'='*50}")
        print(f"OPTIONS FLOW ANALYSIS: {analysis['symbol']} - Exp: {analysis['expiration_date']}")
        print(f"{'='*50}")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        
        print(f"\n{'-'*25} MARKET SENTIMENT {'-'*25}")
        print(f"Call/Put Ratio: {analysis['call_put_ratio']:.2f} (>1 is bullish)")
        print(f"Put/Call Ratio: {analysis['put_call_ratio']:.2f} (<1 is bullish)")
        
        if analysis['premium_ratio'] > 0:
            print(f"Premium Ratio: {analysis['premium_ratio']:.2f} (Call $ / Put $)")
        
        print(f"Net Gamma: {analysis['net_gamma']:.2f}")
        print(f"Net Delta: {analysis['net_delta']:.2f}")
        print(f"Net Vanna: {analysis['net_vanna']:.2f}")
        print(f"Overall Sentiment: {analysis['key_levels']['current']['overall_sentiment']}")
        
        # Display zero gamma line if available
        if (analysis['key_levels']['gex_summary']['zero_gamma_line'] != analysis['current_price']):
            zero_line = analysis['key_levels']['gex_summary']['zero_gamma_line']
            distance = ((zero_line / analysis['current_price']) - 1) * 100
            direction = "above" if zero_line > analysis['current_price'] else "below"
            print(f"Zero Gamma Line: ${zero_line:.2f} ({abs(distance):.1f}% {direction} current price)")
        
        # Display resistance levels
        print(f"\n{'-'*25} KEY RESISTANCE LEVELS {'-'*25}")
        
        if 'gamma' in analysis['key_levels']['resistance'] and not analysis['key_levels']['resistance']['gamma'].empty:
            print("\n🔹 By Gamma Exposure:")
            for _, row in analysis['key_levels']['resistance']['gamma'].iterrows():
                print(f"  ${row['Strike']:.0f} (GEX: {row['GEX']:.2f})")
        
        print("\n🔹 By Open Interest:")
        for _, row in analysis['key_levels']['resistance']['oi'].iterrows():
            print(f"  ${row['Strike']:.0f} (OI Diff: {row['Net_OI']:.0f})")
        
        # Display support levels
        print(f"\n{'-'*25} KEY SUPPORT LEVELS {'-'*25}")
        
        if 'gamma' in analysis['key_levels']['support'] and not analysis['key_levels']['support']['gamma'].empty:
            print("\n🔹 By Gamma Exposure:")
            for _, row in analysis['key_levels']['support']['gamma'].iterrows():
                print(f"  ${row['Strike']:.0f} (GEX: {row['GEX']:.2f})")
        
        print("\n🔹 By Open Interest:")
        for _, row in analysis['key_levels']['support']['oi'].iterrows():
            print(f"  ${row['Strike']:.0f} (OI Diff: {row['Net_OI']:.0f})")
        
        # Display nearest strikes
        print(f"\n{'-'*25} NEAREST STRIKES {'-'*25}")
        for _, row in analysis['key_levels']['current']['nearest_strikes'].iterrows():
            pcr = row['OI_Puts'] / row['OI_Calls'] if row['OI_Calls'] > 0 else float('inf')
            print(f"  ${row['Strike']:.0f} (PCR: {pcr:.2f}, GEX: {row['GEX']:.2f})")
        
        # Show the visualization
        plt.show()


def main():
    """Main function for running the analysis"""
    # Get user input
    symbol = input("Enter ticker symbol (default: SPY): ") or "SPY"
    dark_mode_input = input("Use dark mode visualization? (y/n, default: y): ").lower() or "y"
    dark_mode = dark_mode_input == "y"
    
    # Initialize analyzer
    analyzer = OptionsFlowAnalyzer(symbol, dark_mode=dark_mode)
    
    try:
        # Fetch available expiration dates
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        if not expirations:
            print("No options data available for this symbol.")
            return
        
        # Display available expiration dates
        print("\nAvailable expiration dates:")
        for i, date in enumerate(expirations):
            exp_date_obj = datetime.strptime(date, '%Y-%m-%d').date()
            days_to_expiry = (exp_date_obj - datetime.today().date()).days
            print(f"{i}: {date} ({days_to_expiry} days)")
        
        # Get user selection
        exp_idx = int(input(f"\nSelect expiration date index (0-{len(expirations)-1}, default: 0): ") or "0")
        
        # Ask for advanced visualizations
        advanced_input = input("\nGenerate advanced visualizations? (y/n, default: n): ").lower() or "n"
        advanced_visuals = advanced_input == "y"
        
        # Run analysis and display results
        print("\nAnalyzing options data, please wait...")
        analysis = analyzer.analyze(exp_idx, advanced_visuals)
        analyzer.display_results(analysis)
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()