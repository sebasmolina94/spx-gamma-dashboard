import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mibian
from datetime import datetime, timedelta
import time
import scipy.stats as stats
from plyer import notification

# Adjustable settings
TICKER = "SPY"
INDEX_TICKERS = ["SPX"]
REFRESH_RATE = 15
CONFIDENCE_THRESHOLD = 75
MIN_LIQUIDITY = 10
STRIKES_PER_SIDE = 20  # 20 strikes above and below current price

# List of available tickers for the dropdown
AVAILABLE_TICKERS = ["SPY", "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA"]

# Cache for historical data
historical_cache = {}

# Function to fetch and process data
def fetch_and_process_data(stock, prev_sentiment=None, prev_trades=None, selected_date=None):
    current_price = stock.history(period="1d")['Close'].iloc[-1]
    expirations = stock.options

    # Select the first future expiration date
    current_date = datetime.now() if selected_date is None else selected_date
    future_expirations = []
    for exp in expirations:
        exp_date = datetime.strptime(exp, '%Y-%m-%d')
        if exp_date > current_date:
            future_expirations.append(exp)
    if not future_expirations:
        raise ValueError("No future expiration dates available. All options have expired.")
    SELECTED_EXPIRATION = future_expirations[0]

    opt_chain = stock.option_chain(SELECTED_EXPIRATION)
    calls = opt_chain.calls
    puts = opt_chain.puts

    is_index = stock.ticker in INDEX_TICKERS
    multiplier = 100 if not is_index else 1

    expiration_date = datetime.strptime(SELECTED_EXPIRATION, '%Y-%m-%d')
    time_to_expiration = (expiration_date - current_date).days / 365.0

    # Ensure time_to_expiration is positive
    if time_to_expiration <= 0:
        raise ValueError(f"Selected expiration date {SELECTED_EXPIRATION} is not in the future relative to {current_date.strftime('%Y-%m-%d')}. Cannot calculate probabilities.")

    if "historical_volumes" not in historical_cache:
        historical_volumes = []
        historical_oi = []
        for i in range(30):
            date = (current_date - timedelta(days=i)).strftime('%Y-%m-%d')
            try:
                hist_chain = stock.option_chain(date)
                hist_calls = hist_chain.calls
                hist_puts = hist_chain.puts
                hist_volumes = (hist_calls['volume'].sum() + hist_puts['volume'].sum()) / 2
                hist_oi = (hist_calls['openInterest'].sum() + hist_puts['openInterest'].sum()) / 2
                historical_volumes.append(hist_volumes)
                historical_oi.append(hist_oi)
            except:
                continue
        historical_cache["historical_volumes"] = historical_volumes
        historical_cache["historical_oi"] = historical_oi

    avg_historical_volume = np.nanmean(historical_cache["historical_volumes"]) if historical_cache["historical_volumes"] else 0
    avg_historical_oi = np.nanmean(historical_cache["historical_oi"]) if historical_cache["historical_oi"] else 0

    strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
    # Select 20 strikes below and 20 strikes above current price
    lower_strikes = [s for s in strikes if s < current_price]
    upper_strikes = [s for s in strikes if s > current_price]
    liquid_strikes = (lower_strikes[-STRIKES_PER_SIDE:] if len(lower_strikes) > STRIKES_PER_SIDE else lower_strikes) + \
                     (upper_strikes[:STRIKES_PER_SIDE] if len(upper_strikes) > STRIKES_PER_SIDE else upper_strikes)
    liquid_strikes = [s for s in liquid_strikes if (calls[calls['strike'] == s]['volume'].sum() >= MIN_LIQUIDITY or
                                                   puts[puts['strike'] == s]['volume'].sum() >= MIN_LIQUIDITY or
                                                   calls[calls['strike'] == s]['openInterest'].sum() >= MIN_LIQUIDITY or
                                                   puts[puts['strike'] == s]['openInterest'].sum() >= MIN_LIQUIDITY)]

    call_volumes = {strike: calls[calls['strike'] == strike]['volume'].sum() if not calls[calls['strike'] == strike].empty else 0 for strike in liquid_strikes}
    put_volumes = {strike: puts[puts['strike'] == strike]['volume'].sum() if not puts[puts['strike'] == strike].empty else 0 for strike in liquid_strikes}
    call_oi = {strike: calls[calls['strike'] == strike]['openInterest'].sum() if not calls[calls['strike'] == strike].empty else 0 for strike in liquid_strikes}
    put_oi = {strike: puts[puts['strike'] == strike]['openInterest'].sum() if not puts[puts['strike'] == strike].empty else 0 for strike in liquid_strikes}
    call_iv = {strike: calls[calls['strike'] == strike]['impliedVolatility'].iloc[0] if not calls[calls['strike'] == strike].empty and not pd.isna(calls[calls['strike'] == strike]['impliedVolatility'].iloc[0]) else np.nan for strike in liquid_strikes}
    put_iv = {strike: puts[puts['strike'] == strike]['impliedVolatility'].iloc[0] if not puts[puts['strike'] == strike].empty and not pd.isna(puts[puts['strike'] == strike]['impliedVolatility'].iloc[0]) else np.nan for strike in liquid_strikes}
    call_prices = {strike: calls[calls['strike'] == strike]['lastPrice'].iloc[0] if not calls[calls['strike'] == strike].empty and not pd.isna(calls[calls['strike'] == strike]['lastPrice'].iloc[0]) else np.nan for strike in liquid_strikes}
    put_prices = {strike: puts[puts['strike'] == strike]['lastPrice'].iloc[0] if not puts[puts['strike'] == strike].empty and not pd.isna(puts[puts['strike'] == strike]['lastPrice'].iloc[0]) else np.nan for strike in liquid_strikes}

    call_greeks = {}
    put_greeks = {}
    call_prob = {}
    put_prob = {}
    risk_free_rate = 0.05
    for strike in liquid_strikes:
        call_row = calls[calls['strike'] == strike]
        put_row = puts[puts['strike'] == strike]
        if not call_row.empty and not np.isnan(call_iv[strike]):
            bs = mibian.BS([current_price, strike, risk_free_rate * 100, time_to_expiration * 365], volatility=call_iv[strike] * 100)
            call_greeks[strike] = {
                'gamma': bs.gamma,
                'delta': bs.callDelta,
                'theta': bs.callTheta / 365,
                'vega': bs.vega / 100
            }
            d1 = (np.log(current_price / strike) + (risk_free_rate + (call_iv[strike] ** 2) / 2) * time_to_expiration) / (call_iv[strike] * np.sqrt(time_to_expiration))
            call_prob[strike] = stats.norm.cdf(d1) * 100
        if not put_row.empty and not np.isnan(put_iv[strike]):
            bs = mibian.BS([current_price, strike, risk_free_rate * 100, time_to_expiration * 365], volatility=put_iv[strike] * 100)
            put_greeks[strike] = {
                'gamma': bs.gamma,
                'delta': bs.putDelta,
                'theta': bs.putTheta / 365,
                'vega': bs.vega / 100
            }
            d1 = (np.log(current_price / strike) + (risk_free_rate + (put_iv[strike] ** 2) / 2) * time_to_expiration) / (put_iv[strike] * np.sqrt(time_to_expiration))
            put_prob[strike] = stats.norm.cdf(-d1) * 100

    call_itm = {s: call_volumes[s] for s in liquid_strikes if s < current_price}
    call_atm = {s: call_volumes[s] for s in liquid_strikes if abs(s - current_price) <= 5}
    call_otm = {s: call_volumes[s] for s in liquid_strikes if s > current_price}
    put_itm = {s: put_volumes[s] for s in liquid_strikes if s > current_price}
    put_atm = {s: put_volumes[s] for s in liquid_strikes if abs(s - current_price) <= 5}
    put_otm = {s: put_volumes[s] for s in liquid_strikes if s < current_price}

    total_puts_otm = sum(put_otm.values())
    total_puts_itm_atm = sum(put_itm.values()) + sum(put_atm.values())
    total_calls_itm = sum(call_itm.values())
    total_calls_otm_atm = sum(call_otm.values()) + sum(call_atm.values())

    sentiment = ""
    hedging_sentiment = ""
    confidence = 0.0

    total_put_volume = total_puts_otm + total_puts_itm_atm
    if total_put_volume > 0:
        puts_otm_ratio = total_puts_otm / total_put_volume
        puts_itm_atm_ratio = total_puts_itm_atm / total_put_volume
        diff_puts = abs(puts_otm_ratio - puts_itm_atm_ratio)
        confidence = diff_puts * 100

        threshold = 0.05
        if puts_otm_ratio > puts_itm_atm_ratio + threshold:
            sentiment = "Bullish"
        elif puts_itm_atm_ratio > puts_otm_ratio + threshold:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
    else:
        sentiment = "Neutral"
        confidence = 0.0

    total_call_volume_for_hedging = total_calls_itm + total_calls_otm_atm
    hedging_confidence = 0.0
    if total_call_volume_for_hedging > 0:
        calls_itm_ratio = total_calls_itm / total_call_volume_for_hedging
        calls_otm_atm_ratio = total_calls_otm_atm / total_call_volume_for_hedging
        diff_calls = abs(calls_itm_ratio - calls_otm_atm_ratio)
        hedging_confidence = diff_calls * 100

        if calls_itm_ratio > calls_otm_atm_ratio + threshold:
            hedging_sentiment = "Bullish Hedging"
        elif calls_otm_atm_ratio > calls_itm_ratio + threshold:
            hedging_sentiment = "Bearish Hedging"

    if hedging_sentiment:
        market_sentiment = hedging_sentiment
        confidence = max(confidence, hedging_confidence)
    else:
        market_sentiment = sentiment

    otm_call_iv = np.nanmean([call_iv[s] for s in call_otm.keys() if not np.isnan(call_iv[s])])
    otm_put_iv = np.nanmean([put_iv[s] for s in put_otm.keys() if not np.isnan(put_iv[s])])
    vol_skew = (otm_put_iv - otm_call_iv) * 100 if not (np.isnan(otm_put_iv) or np.isnan(otm_call_iv)) else 0

    call_gamma = {s: call_greeks[s]['gamma'] * call_oi[s] * multiplier if s in call_greeks else 0 for s in liquid_strikes}
    put_gamma = {s: put_greeks[s]['gamma'] * put_oi[s] * multiplier if s in put_greeks else 0 for s in liquid_strikes}
    net_gamma = {s: call_gamma[s] - put_gamma[s] for s in liquid_strikes}
    avg_net_gamma = np.mean([abs(g) for g in net_gamma.values()])
    gamma_walls = [s for s, g in net_gamma.items() if abs(g) > 2 * avg_net_gamma]

    avg_call_volume = np.mean([call_volumes[s] for s in liquid_strikes])
    avg_put_volume = np.mean([put_volumes[s] for s in liquid_strikes])
    avg_call_oi = np.mean([call_oi[s] for s in liquid_strikes])
    avg_put_oi = np.mean([put_oi[s] for s in liquid_strikes])

    high_confidence_strikes = []
    for s in liquid_strikes:
        call_vol_anomaly = call_volumes[s] > max(2 * avg_call_volume, avg_historical_volume)
        put_vol_anomaly = put_volumes[s] > max(2 * avg_put_volume, avg_historical_volume)
        call_oi_anomaly = call_oi[s] > max(2 * avg_call_oi, avg_historical_oi)
        put_oi_anomaly = put_oi[s] > max(2 * avg_put_oi, avg_historical_oi)
        if (call_vol_anomaly or put_vol_anomaly or call_oi_anomaly or put_oi_anomaly) and confidence > CONFIDENCE_THRESHOLD:
            high_confidence_strikes.append(s)

    trade_recommendations = []
    for s in liquid_strikes:
        if not np.isnan(call_prices[s]) and call_volumes[s] > 0:
            cost = call_prices[s] * multiplier
            potential_profit = max(0, s - current_price - call_prices[s]) * multiplier
            potential_loss = cost
            risk_reward = potential_profit / potential_loss if potential_loss > 0 else 0
            breakeven = current_price + call_prices[s]
            greeks = call_greeks.get(s, {'delta': 0, 'theta': 0, 'vega': 0})
            prob = call_prob.get(s, 0)
            if s in high_confidence_strikes and risk_reward > 1:
                trade_recommendations.append({
                    'Type': 'Long Call',
                    'Strike': s,
                    'Cost': cost,
                    'Potential Profit': potential_profit,
                    'Risk/Reward': risk_reward,
                    'Breakeven': breakeven,
                    'Confidence': confidence,
                    'Delta': greeks['delta'],
                    'Theta': greeks['theta'],
                    'Vega': greeks['vega'],
                    'Probability': prob
                })
        if not np.isnan(put_prices[s]) and put_volumes[s] > 0:
            cost = put_prices[s] * multiplier
            potential_profit = max(0, current_price - s - put_prices[s]) * multiplier
            potential_loss = cost
            risk_reward = potential_profit / potential_loss if potential_loss > 0 else 0
            breakeven = current_price - put_prices[s]
            greeks = put_greeks.get(s, {'delta': 0, 'theta': 0, 'vega': 0})
            prob = put_prob.get(s, 0)
            if s in high_confidence_strikes and risk_reward > 1:
                trade_recommendations.append({
                    'Type': 'Long Put',
                    'Strike': s,
                    'Cost': cost,
                    'Potential Profit': potential_profit,
                    'Risk/Reward': risk_reward,
                    'Breakeven': breakeven,
                    'Confidence': confidence,
                    'Delta': greeks['delta'],
                    'Theta': greeks['theta'],
                    'Vega': greeks['vega'],
                    'Probability': prob
                })

    call_weighted_iv = {s: call_iv[s] * call_volumes[s] for s in liquid_strikes}
    put_weighted_iv = {s: put_iv[s] * put_volumes[s] for s in liquid_strikes}
    total_call_weight = sum(call_volumes.values())
    total_put_weight = sum(put_volumes.values())
    call_weighted_iv = {s: val / total_call_weight if total_call_weight > 0 else 0 for s, val in call_weighted_iv.items()}
    put_weighted_iv = {s: val / total_put_weight if total_put_weight > 0 else 0 for s, val in put_weighted_iv.items()}

    total_call_volume = sum(call_volumes.values())
    total_put_volume = sum(put_volumes.values())
    put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
    max_pain = liquid_strikes[np.argmax([sum(abs(s - st) * (call_oi[st] if st > s else put_oi[st]) for st in liquid_strikes) for s in liquid_strikes])]

    max_oi = max(max(call_oi.values(), default=0), max(put_oi.values(), default=0)) * 1.1

    # Calculate expected move range (using implied volatility)
    avg_iv = np.nanmean([iv for iv in list(call_iv.values()) + list(put_iv.values()) if not np.isnan(iv)])
    expected_move = current_price * avg_iv * np.sqrt(time_to_expiration)
    expected_move_range = (current_price - expected_move, current_price + expected_move)

    if prev_sentiment and prev_sentiment != market_sentiment:
        notification.notify(
            title="Options Visualization Alert",
            message=f"Market sentiment changed to {market_sentiment} (Confidence: {confidence:.2f}%)",
            timeout=10
        )
    if prev_trades and trade_recommendations:
        new_trades = [t for t in trade_recommendations if t not in prev_trades]
        if new_trades:
            for trade in new_trades:
                notification.notify(
                    title="Options Visualization Alert",
                    message=f"New Trade: {trade['Type']} at ${trade['Strike']}, R/R={trade['Risk/Reward']:.2f}, Prob={trade['Probability']:.2f}%",
                    timeout=10
                )

    return (current_price, SELECTED_EXPIRATION, market_sentiment, confidence, vol_skew, liquid_strikes,
            call_itm, call_atm, call_otm, put_itm, put_atm, put_otm, call_volumes, put_volumes,
            call_oi, put_oi, call_iv, put_iv, call_weighted_iv, put_weighted_iv, high_confidence_strikes,
            gamma_walls, trade_recommendations, total_call_volume, total_put_volume, put_call_ratio, max_pain,
            max_oi, expected_move, expected_move_range)

# Initialize stock object
stock = yf.Ticker(TICKER)

# Create the initial Plotly figure
fig = make_subplots(
    rows=4, cols=2,
    row_heights=[0.1, 0.4, 0.3, 0.2],
    column_widths=[0.7, 0.3],
    subplot_titles=("Options Volume Heatmap", "Implied Volatility Smile", "Open Interest by Strike", "Volatility Skew"),
    vertical_spacing=0.05,
    specs=[
        [{"colspan": 2}, None],
        [{"type": "heatmap"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"colspan": 2}, None]
    ]
)

# Initial data fetch to get expiration date for the date dropdown
(current_price, SELECTED_EXPIRATION, market_sentiment, confidence, vol_skew, liquid_strikes,
 call_itm, call_atm, call_otm, put_itm, put_atm, put_otm, call_volumes, put_volumes,
 call_oi, put_oi, call_iv, put_iv, call_weighted_iv, put_weighted_iv, high_confidence_strikes,
 gamma_walls, trade_recommendations, total_call_volume, total_put_volume, put_call_ratio, max_pain,
 max_oi, expected_move, expected_move_range) = fetch_and_process_data(stock)

# Generate date options for the dropdown (current date to expiration, max 7 days)
current_date = datetime.now()
expiration_date = datetime.strptime(SELECTED_EXPIRATION, '%Y-%m-%d')
date_options = []
for i in range(min(7, (expiration_date - current_date).days + 1)):
    date = current_date + timedelta(days=i)
    date_str = date.strftime('%Y-%m-%d')
    date_options.append(date_str)

# Create dropdown menus
updatemenus = [
    # Dropdown for selecting date
    dict(
        buttons=[
            dict(
                label=date_str,
                method="update",
                args=[{"title": f"Options Visualization (Date: {date_str})"}]
            ) for date_str in date_options
        ],
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="left",
        y=1.15,
        yanchor="top",
        bgcolor="rgba(0,0,0,0.5)",
        font=dict(color="white")
    ),
    # Dropdown for selecting ticker
    dict(
        buttons=[
            dict(
                label=ticker,
                method="update",
                args=[{"title": f"Options Visualization (Ticker: {ticker})"}]
            ) for ticker in AVAILABLE_TICKERS
        ],
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.3,
        xanchor="left",
        y=1.15,
        yanchor="top",
        bgcolor="rgba(0,0,0,0.5)",
        font=dict(color="white")
    )
]

fig.update_layout(
    title=f"Options Visualization (Ticker: {TICKER}, Date: {current_date.strftime('%Y-%m-%d')})",
    title_x=0.5,
    title_font=dict(size=20, color="white"),
    plot_bgcolor="#1c2526",
    paper_bgcolor="#1c2526",
    font=dict(color="white"),
    showlegend=True,
    height=1000,
    updatemenus=updatemenus,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    xaxis2=dict(
        title="Strike",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        visible=True,
        zeroline=False,
        showgrid=False,
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="white"
    ),
    yaxis2=dict(
        title="Option Type",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        visible=True,
        zeroline=False,
        showgrid=False,
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="white"
    ),
    xaxis3=dict(
        title="Strike",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        visible=True,
        zeroline=False,
        showgrid=False,
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="white"
    ),
    yaxis3=dict(
        title="Open Interest",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        visible=True,
        zeroline=False,
        showgrid=False,
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="white"
    ),
    xaxis4=dict(
        title="Strike",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        visible=True,
        zeroline=False,
        showgrid=False,
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="white"
    ),
    yaxis4=dict(
        title="Implied Volatility",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        visible=True,
        zeroline=False,
        showgrid=False,
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="white"
    ),
)

# Update x-axis ranges for heatmap and open interest plots
fig.update_xaxes(range=[min(liquid_strikes) - 1, max(liquid_strikes) + 1], row=2, col=1)  # Heatmap
fig.update_xaxes(range=[min(liquid_strikes) - 1, max(liquid_strikes) + 1], row=3, col=1)  # Open Interest

# Info annotation
sentiment_color = 'green' if "Bullish" in market_sentiment else 'red' if "Bearish" in market_sentiment else 'gray'
fig.add_annotation(
    x=0.05, y=0.95, xref="paper", yref="paper",
    text=f"Symbol: {TICKER}<br>Expiration: {SELECTED_EXPIRATION}<br>Price: ${current_price:.2f}<br>"
         f"Market Sentiment: {market_sentiment} (Confidence: {confidence:.2f}%)<br>"
         f"Vol Skew (OTM Put - Call): {vol_skew:.2f}%",
    showarrow=False,
    font=dict(size=12, color="white"),
    align="left",
    bgcolor=sentiment_color
)

# Additional info annotation
fig.add_annotation(
    x=0.5, y=0.95, xref="paper", yref="paper",
    text=f"Expected Move: ±{expected_move:.2f}%  Range: ${expected_move_range[0]:.2f} - ${expected_move_range[1]:.2f}<br>"
         f"ITM Call/Put Ratio: {put_call_ratio:.2f}",
    showarrow=False,
    font=dict(size=12, color="white"),
    align="center",
    bgcolor="rgba(0,0,0,0.5)"
)

# Heatmap with dark background and updated text color
heatmap_data = np.array([call_itm.get(s, 0) for s in liquid_strikes] +
                        [call_atm.get(s, 0) for s in liquid_strikes] +
                        [call_otm.get(s, 0) for s in liquid_strikes] +
                        [put_itm.get(s, 0) for s in liquid_strikes] +
                        [put_atm.get(s, 0) for s in liquid_strikes] +
                        [put_otm.get(s, 0) for s in liquid_strikes]).reshape(6, len(liquid_strikes))
fig.add_trace(
    go.Heatmap(
        z=heatmap_data,
        x=liquid_strikes,
        y=['Call ITM', 'Call ATM', 'Call OTM', 'Put ITM', 'Put ATM', 'Put OTM'],
        colorscale=[[0, "#1c2526"], [0.5, "red"], [1, "white"]],
        hoverinfo="x+y+z",
        colorbar=dict(
            title="Normalized Volume",
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            outlinecolor="white"
        ),
        zmin=0,
        zmax=np.max(heatmap_data) if np.max(heatmap_data) > 0 else 1,
        text=heatmap_data,
        texttemplate="%{text:.0f}",
        textfont=dict(color="black")  # Changed to black for better contrast
    ),
    row=2, col=1
)
fig.add_vline(x=current_price, line=dict(color="cyan", dash="dash"), row=2, col=1)
fig.add_vline(x=max_pain, line=dict(color="yellow", dash="dot"), row=2, col=1, annotation_text="Max Pain", annotation_position="top")

# IV Smile
fig.add_trace(go.Scatter(x=liquid_strikes, y=[call_iv[s] for s in liquid_strikes], name="Call IV", line=dict(color="blue")), row=2, col=2)
fig.add_trace(go.Scatter(x=liquid_strikes, y=[put_iv[s] for s in liquid_strikes], name="Put IV", line=dict(color="red")), row=2, col=2)
fig.add_trace(go.Scatter(x=liquid_strikes, y=[call_weighted_iv[s] for s in liquid_strikes], name="Call Weighted IV", line=dict(color="blue", dash="dash")), row=2, col=2)
fig.add_trace(go.Scatter(x=liquid_strikes, y=[put_weighted_iv[s] for s in liquid_strikes], name="Put Weighted IV", line=dict(color="red", dash="dash")), row=2, col=2)
fig.add_vline(x=current_price, line=dict(color="cyan", dash="dash"), row=2, col=2)
fig.add_vline(x=expected_move_range[0], line=dict(color="yellow", dash="dot"), row=2, col=2)
fig.add_vline(x=expected_move_range[1], line=dict(color="yellow", dash="dot"), row=2, col=2)

# OI by Strike
fig.add_trace(go.Bar(x=liquid_strikes, y=[call_oi[s] for s in liquid_strikes], name="Call OI", marker_color="blue", opacity=0.6), row=3, col=1)
fig.add_trace(go.Bar(x=liquid_strikes, y=[put_oi[s] for s in liquid_strikes], name="Put OI", marker_color="red", opacity=0.6), row=3, col=1)
fig.add_vline(x=current_price, line=dict(color="cyan", dash="dash"), row=3, col=1)
fig.add_vline(x=max_pain, line=dict(color="yellow", dash="dot"), row=3, col=1, annotation_text="Max Pain", annotation_position="top")
fig.update_yaxes(range=[0, max_oi], row=3, col=1)

# Volatility Skew
otm_strikes = sorted(set(call_otm.keys()).union(set(put_otm.keys())))
otm_call_ivs = [call_iv[s] for s in otm_strikes if s in call_iv]
otm_put_ivs = [put_iv[s] for s in otm_strikes if s in put_iv]
fig.add_trace(go.Scatter(x=otm_strikes, y=otm_call_ivs, name="OTM Call IV", line=dict(color="blue")), row=3, col=2)
fig.add_trace(go.Scatter(x=otm_strikes, y=otm_put_ivs, name="OTM Put IV", line=dict(color="red")), row=3, col=2)

# Trade Recommendations
trade_text = "Potential Trades (High Confidence):<br>"
for trade in sorted(trade_recommendations, key=lambda x: x['Risk/Reward'], reverse=True)[:5]:
    trade_text += (f"{trade['Type']} at ${trade['Strike']}: Cost=${trade['Cost']:.2f}, "
                   f"Profit=${trade['Potential Profit']:.2f}, R/R={trade['Risk/Reward']:.2f}, "
                   f"Breakeven=${trade['Breakeven']:.2f}, Conf={trade['Confidence']:.2f}%, "
                   f"Delta={trade['Delta']:.2f}, Theta={trade['Theta']:.2f}, Vega={trade['Vega']:.2f}, "
                   f"Prob={trade['Probability']:.2f}%<br>")
fig.add_annotation(
    x=0.1, y=0.15, xref="paper", yref="paper",
    text=trade_text if trade_recommendations else "No high-confidence trades identified.",
    showarrow=False,
    font=dict(size=12, color="white"),
    align="left",
    bgcolor="rgba(0,0,0,0.5)"
)

# Add gamma walls to Open Interest plot
for strike in gamma_walls:
    fig.add_vline(x=strike, line=dict(color="purple", dash="dash", width=1), row=3, col=1)

# Show the initial plot
fig.show()

# Refresh loop
prev_sentiment = market_sentiment
prev_trades = trade_recommendations
selected_date = None
selected_ticker = TICKER

while True:
    time.sleep(REFRESH_RATE)
    print(f"Refreshing data at {datetime.now()}...")

    # Update ticker if changed
    if selected_ticker != TICKER:
        stock = yf.Ticker(selected_ticker)
        historical_cache.clear()  # Clear cache when ticker changes

    # Fetch data with selected date
    (current_price, SELECTED_EXPIRATION, market_sentiment, confidence, vol_skew, liquid_strikes,
     call_itm, call_atm, call_otm, put_itm, put_atm, put_otm, call_volumes, put_volumes,
     call_oi, put_oi, call_iv, put_iv, call_weighted_iv, put_weighted_iv, high_confidence_strikes,
     gamma_walls, trade_recommendations, total_call_volume, total_put_volume, put_call_ratio, max_pain,
     max_oi, expected_move, expected_move_range) = fetch_and_process_data(stock, prev_sentiment, prev_trades, selected_date)

    sentiment_color = 'green' if "Bullish" in market_sentiment else 'red' if "Bearish" in market_sentiment else 'gray'
    fig.layout.annotations[0].update(
        text=f"Symbol: {selected_ticker}<br>Expiration: {SELECTED_EXPIRATION}<br>Price: ${current_price:.2f}<br>"
             f"Market Sentiment: {market_sentiment} (Confidence: {confidence:.2f}%)<br>"
             f"Vol Skew (OTM Put - Call): {vol_skew:.2f}%",
        bgcolor=sentiment_color
    )

    fig.layout.annotations[1].update(
        text=f"Expected Move: ±{expected_move:.2f}%  Range: ${expected_move_range[0]:.2f} - ${expected_move_range[1]:.2f}<br>"
             f"ITM Call/Put Ratio: {put_call_ratio:.2f}"
    )

    heatmap_data = np.array([call_itm.get(s, 0) for s in liquid_strikes] +
                            [call_atm.get(s, 0) for s in liquid_strikes] +
                            [call_otm.get(s, 0) for s in liquid_strikes] +
                            [put_itm.get(s, 0) for s in liquid_strikes] +
                            [put_atm.get(s, 0) for s in liquid_strikes] +
                            [put_otm.get(s, 0) for s in liquid_strikes]).reshape(6, len(liquid_strikes))
    fig.data[0].update(z=heatmap_data, x=liquid_strikes, text=heatmap_data)

    fig.data[1].update(x=liquid_strikes, y=[call_iv[s] for s in liquid_strikes])
    fig.data[2].update(x=liquid_strikes, y=[put_iv[s] for s in liquid_strikes])
    fig.data[3].update(x=liquid_strikes, y=[call_weighted_iv[s] for s in liquid_strikes])
    fig.data[4].update(x=liquid_strikes, y=[put_weighted_iv[s] for s in liquid_strikes])

    fig.data[5].update(x=liquid_strikes, y=[call_oi[s] for s in liquid_strikes])
    fig.data[6].update(x=liquid_strikes, y=[put_oi[s] for s in liquid_strikes])
    fig.update_yaxes(range=[0, max_oi], row=3, col=1)

    otm_strikes = sorted(set(call_otm.keys()).union(set(put_otm.keys())))
    otm_call_ivs = [call_iv[s] for s in otm_strikes if s in call_iv]
    otm_put_ivs = [put_iv[s] for s in otm_strikes if s in put_iv]
    fig.data[7].update(x=otm_strikes, y=otm_call_ivs)
    fig.data[8].update(x=otm_strikes, y=otm_put_ivs)

    trade_text = "Potential Trades (High Confidence):<br>"
    for trade in sorted(trade_recommendations, key=lambda x: x['Risk/Reward'], reverse=True)[:5]:
        trade_text += (f"{trade['Type']} at ${trade['Strike']}: Cost=${trade['Cost']:.2f}, "
                       f"Profit=${trade['Potential Profit']:.2f}, R/R={trade['Risk/Reward']:.2f}, "
                       f"Breakeven=${trade['Breakeven']:.2f}, Conf={trade['Confidence']:.2f}%, "
                       f"Delta={trade['Delta']:.2f}, Theta={trade['Theta']:.2f}, Vega={trade['Vega']:.2f}, "
                       f"Prob={trade['Probability']:.2f}%<br>")
    fig.layout.annotations[2].update(text=trade_text if trade_recommendations else "No high-confidence trades identified.")

    fig.layout.shapes = []
    fig.add_vline(x=current_price, line=dict(color="cyan", dash="dash"), row=2, col=1)
    fig.add_vline(x=max_pain, line=dict(color="yellow", dash="dot"), row=2, col=1, annotation_text="Max Pain", annotation_position="top")
    fig.add_vline(x=current_price, line=dict(color="cyan", dash="dash"), row=3, col=1)
    fig.add_vline(x=max_pain, line=dict(color="yellow", dash="dot"), row=3, col=1, annotation_text="Max Pain", annotation_position="top")
    fig.add_vline(x=current_price, line=dict(color="cyan", dash="dash"), row=2, col=2)
    fig.add_vline(x=expected_move_range[0], line=dict(color="yellow", dash="dot"), row=2, col=2)
    fig.add_vline(x=expected_move_range[1], line=dict(color="yellow", dash="dot"), row=2, col=2)
    for strike in high_confidence_strikes:
        fig.add_vline(x=strike, line=dict(color="yellow", dash="dot", width=1), row=2, col=1)
    for strike in gamma_walls:
        fig.add_vline(x=strike, line=dict(color="purple", dash="dash", width=1), row=3, col=1)

    # Update x-axis ranges during refresh
    fig.update_xaxes(range=[min(liquid_strikes) - 1, max(liquid_strikes) + 1], row=2, col=1)
    fig.update_xaxes(range=[min(liquid_strikes) - 1, max(liquid_strikes) + 1], row=3, col=1)

    fig.update_layout(overwrite=True)

    prev_sentiment = market_sentiment
    prev_trades = trade_recommendations