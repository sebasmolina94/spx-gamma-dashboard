def setup_options_activity_tab(self):
    """Set up the Options Activity tab with heatmap and charts."""
    # Create a layout for the tab
    layout = QVBoxLayout()

    # Create a canvas for the heatmap
    self.heatmap_canvas = MplCanvas(self, width=8, height=4, dpi=100)
    layout.addWidget(self.heatmap_canvas)

    # Create a canvas for the Open Interest by Strike chart
    self.oi_canvas = MplCanvas(self, width=8, height=2, dpi=100)
    layout.addWidget(self.oi_canvas)

    # Create a canvas for the Implied Volatility Smile chart
    self.iv_canvas = MplCanvas(self, width=8, height=2, dpi=100)
    layout.addWidget(self.iv_canvas)

    # Set the layout for the tab
    self.exposure_tabs.widget(self.exposure_tabs.indexOf(self.options_activity_tab)).setLayout(layout)
import sys
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from math import log, sqrt
from scipy.stats import norm

# Helper function to safely convert numpy types to Python native types
def safe_convert(value):
    """Convert numpy types to Python native types to avoid numpy.int64 issues."""
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, (np.ndarray, list, tuple)):
        return [safe_convert(x) for x in value]
    elif isinstance(value, dict):
        return {safe_convert(k): safe_convert(v) for k, v in value.items()}
    else:
        return value
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QSplitter, QFrame, QSizePolicy, QMessageBox,
    QTabWidget, QTabBar, QSlider, QTableView, QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PyQt6.QtWidgets import QStyledItemDelegate
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QPicture
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.text import Annotation

# Try to import PyQtGraph, but don't fail if it's not available
try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    print("PyQtGraph not available. Candlestick chart will use matplotlib instead.")
    PYQTGRAPH_AVAILABLE = False
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta
from py_vollib.black_scholes.greeks.analytical import gamma as bs_gamma
from py_vollib.black_scholes.greeks.analytical import vega as bs_vega
from py_vollib.black_scholes.greeks.analytical import theta as bs_theta

# Custom CandlestickItem for PyQtGraph - only define if PyQtGraph is available
if PYQTGRAPH_AVAILABLE:
    class CandlestickItem(pg.GraphicsObject):
        def __init__(self, data, width=0.2, bull_color=(0, 255, 0), bear_color=(255, 0, 0)):
            pg.GraphicsObject.__init__(self)
            self.data = data
            self.width = width
            self.bull_color = bull_color
            self.bear_color = bear_color
            self.picture = None
            self.generatePicture()

        def generatePicture(self):
            self.picture = QPicture()

            painter = QPainter(self.picture)

            w = self.width / 2
            for (t, open, high, low, close) in self.data:
                if close >= open:
                    # Bull candle
                    # Create bull color pen with consistent width for both body and wicks
                    bull_pen = QPen(pg.mkPen(self.bull_color, width=1.5))
                    painter.setPen(bull_pen)
                    # Set bull color brush
                    painter.setBrush(pg.mkBrush(self.bull_color))
                    # Draw the candle body without visible border
                    painter.drawRect(QRectF(t-w, open, w*2, close-open))

                    # Draw the wick with same color as candle
                    # Draw upper wick if needed
                    if high > close:
                        painter.drawLine(QPointF(t, close), QPointF(t, high))
                    # Draw lower wick if needed
                    if low < open:
                        painter.drawLine(QPointF(t, open), QPointF(t, low))
                else:
                    # Bear candle
                    # Create bear color pen with consistent width for both body and wicks
                    bear_pen = QPen(pg.mkPen(self.bear_color, width=1.5))
                    painter.setPen(bear_pen)
                    # Set bear color brush
                    painter.setBrush(pg.mkBrush(self.bear_color))
                    # Draw the candle body without visible border
                    painter.drawRect(QRectF(t-w, open, w*2, close-open))

                    # Draw the wick with same color as candle
                    # Draw upper wick if needed
                    if high > open:
                        painter.drawLine(QPointF(t, open), QPointF(t, high))
                    # Draw lower wick if needed
                    if low < close:
                        painter.drawLine(QPointF(t, close), QPointF(t, low))
            painter.end()

        def paint(self, painter, option, widget):
            # option and widget are unused but required by the interface
            painter.drawPicture(0, 0, self.picture)

        def boundingRect(self):
            return QRectF(self.picture.boundingRect())

class OptionsChainTChartModel(QAbstractTableModel):
    """Model for displaying options chain data in a T-chart format with puts on left, strike in middle, calls on right"""
    def __init__(self):
        super().__init__()
        # TOS-like colors
        self.call_color = QColor('#00CC00')  # Softer green for calls
        self.put_color = QColor('#FF3333')   # Softer red for puts
        self.atm_color = QColor('#FFA500')   # Orange for ATM (more like TOS)
        self.header_color = QColor('#333333')  # Dark gray for header
        self.bg_color_dark = QColor('#1C1C1C')  # Very dark gray (almost black) for background
        self.bg_color_alt = QColor('#252525')  # Slightly lighter dark gray for alternating rows
        self.strike_bg_color = QColor('#333333')  # Darker gray for strike column

        # Initialize empty data
        self._strikes = []  # Unique strike prices
        self._put_data = {}  # Dictionary of put data by strike
        self._call_data = {}  # Dictionary of call data by strike
        self._columns = []  # Column names (excluding strike and option_type)
        self._all_columns = []  # All column names including computed ones

    def rowCount(self, parent=QModelIndex()):
        return len(self._strikes)

    def columnCount(self, parent=QModelIndex()):
        # Format: [Put columns] + [Strike] + [Call columns]
        if not self._columns:
            return 1  # Just the strike column if no data
        return len(self._columns) * 2 + 1  # Put columns + Strike + Call columns

    def formatValue(self, value, column_name):
        """Format a value based on its column name in a TOS-like style"""
        if pd.isna(value):
            return "--"  # TOS uses dashes for N/A values

        # Format contractSymbol specially
        if column_name == 'contractSymbol':
            # Shorten the symbol to make it more readable (TOS-like)
            symbol_str = str(value)
            # If it's a long symbol, try to make it more compact
            if len(symbol_str) > 12:
                parts = symbol_str.split('_')
                if len(parts) > 1:
                    # Keep just the essential parts
                    return parts[-1]
            return symbol_str

        # Format numbers appropriately in TOS-like style
        if isinstance(value, (int, float)):
            if column_name == 'strike':
                # Bold formatting for strike prices in TOS
                return f"{value:.2f}"
            elif column_name == 'impliedVolatility':
                # TOS shows IV as percentage with 1 decimal place
                return f"{value*100:.1f}%"
            elif column_name == 'calc_delta':
                # TOS shows delta as a decimal with 2 places
                return f"{value:.2f}"
            elif column_name in ['calc_gamma', 'calc_vega']:
                # TOS shows gamma and vega with 4 decimal places
                return f"{value:.4f}"
            elif column_name == 'calc_theta':
                # TOS shows theta with 2 decimal places and a negative sign
                return f"{value:.2f}"
            elif column_name in ['calc_vanna', 'calc_vomma']:
                # Other greeks with 4 decimal places
                return f"{value:.4f}"
            elif column_name in ['volume', 'openInterest']:
                # TOS shows volume and OI as integers with commas
                return f"{int(value):,}"
            elif column_name in ['lastPrice', 'bid', 'ask']:
                # TOS shows prices with dollar sign and 2 decimal places
                return f"${value:.2f}"
            else:
                return f"{value}"
        return str(value)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._strikes):
            return None

        row = index.row()
        col = index.column()
        strike = self._strikes[row]

        # Calculate which section we're in (put, strike, or call)
        col_count = len(self._columns)

        # Strike column (middle)
        if col == col_count:
            if role == Qt.ItemDataRole.DisplayRole:
                return f"{strike:.2f}"
            elif role == Qt.ItemDataRole.BackgroundRole:
                # Check if this strike is ATM in either puts or calls
                put_data = self._put_data.get(strike, {})
                call_data = self._call_data.get(strike, {})

                if ('moneyness' in put_data and put_data['moneyness'] == 'ATM') or \
                   ('moneyness' in call_data and call_data['moneyness'] == 'ATM'):
                    return self.atm_color
                return self.strike_bg_color  # Use darker background for strike column
            elif role == Qt.ItemDataRole.ForegroundRole:
                # White text for strike column for better readability
                return QColor('white')
            return None

        # Put columns (left side)
        elif col < col_count:
            column_name = self._columns[col]
            put_data = self._put_data.get(strike, {})

            if role == Qt.ItemDataRole.DisplayRole:
                if column_name in put_data:
                    return self.formatValue(put_data[column_name], column_name)
                return ""
            elif role == Qt.ItemDataRole.BackgroundRole:
                # Alternate row colors for better readability
                return self.bg_color_alt if row % 2 else self.bg_color_dark
            elif role == Qt.ItemDataRole.ForegroundRole:
                # Check if this strike is ATM
                if 'moneyness' in put_data and put_data['moneyness'] == 'ATM':
                    return QColor('white')  # White text for ATM options for emphasis
                return self.put_color
            return None

        # Call columns (right side)
        else:
            # Adjust column index for call side
            call_col = col - col_count - 1
            column_name = self._columns[call_col]
            call_data = self._call_data.get(strike, {})

            if role == Qt.ItemDataRole.DisplayRole:
                if column_name in call_data:
                    return self.formatValue(call_data[column_name], column_name)
                return ""
            elif role == Qt.ItemDataRole.BackgroundRole:
                # Alternate row colors for better readability
                return self.bg_color_alt if row % 2 else self.bg_color_dark
            elif role == Qt.ItemDataRole.ForegroundRole:
                # Check if this strike is ATM
                if 'moneyness' in call_data and call_data['moneyness'] == 'ATM':
                    return QColor('white')  # White text for ATM options for emphasis
                return self.call_color
            return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.DisplayRole:
                return section + 1
            elif role == Qt.ItemDataRole.BackgroundRole:
                # Dark background for row headers
                return self.header_color
            elif role == Qt.ItemDataRole.ForegroundRole:
                # White text for row headers
                return QColor('white')
            return None

        if orientation == Qt.Orientation.Horizontal:
            col_count = len(self._columns)

            # Common styling for all headers
            if role == Qt.ItemDataRole.BackgroundRole:
                return self.header_color
            elif role == Qt.ItemDataRole.ForegroundRole:
                # Strike column (middle)
                if section == col_count:
                    return QColor('white')  # White text for strike header
                # Put columns (left side)
                elif section < col_count:
                    return self.put_color  # Red text for put headers
                # Call columns (right side)
                else:
                    return self.call_color  # Green text for call headers
            elif role == Qt.ItemDataRole.DisplayRole:
                # Strike column (middle)
                if section == col_count:
                    return "Strike"

                # Put columns (left side)
                elif section < col_count:
                    header = self._columns[section]

                    # Special case for contractSymbol
                    if header == 'contractSymbol':
                        return 'Put Symbol'

                    # Replace underscores with spaces and capitalize each word
                    return 'Put ' + ' '.join(word.capitalize() for word in header.split('_'))

                # Call columns (right side)
                else:
                    # Adjust column index for call side
                    call_col = section - col_count - 1
                    header = self._columns[call_col]

                    # Special case for contractSymbol
                    if header == 'contractSymbol':
                        return 'Call Symbol'

                    # Replace underscores with spaces and capitalize each word
                    return 'Call ' + ' '.join(word.capitalize() for word in header.split('_'))

        return None

    def setOptionsData(self, puts_df, calls_df, columns):
        """Set the data for the T-chart model"""
        self.beginResetModel()

        # Store the column names (excluding strike and option_type which are handled specially)
        self._columns = [col for col in columns if col not in ['strike', 'option_type']]

        # Get all unique strikes from both puts and calls
        all_strikes = set(puts_df['strike'].unique()) | set(calls_df['strike'].unique())
        self._strikes = sorted(all_strikes)

        # Create dictionaries for quick lookup by strike
        self._put_data = {}
        for _, row in puts_df.iterrows():
            strike = row['strike']
            self._put_data[strike] = {col: row[col] for col in self._columns if col in row}

        self._call_data = {}
        for _, row in calls_df.iterrows():
            strike = row['strike']
            self._call_data[strike] = {col: row[col] for col in self._columns if col in row}

        self.endResetModel()

class OptionsChainModel(QAbstractTableModel):
    """Model for displaying options chain data in a table"""
    def __init__(self, data=None, headers=None):
        super().__init__()
        self._data = data if data is not None else pd.DataFrame()
        self._headers = headers if headers is not None else []
        # TOS-like colors
        self.call_color = QColor('#00CC00')  # Softer green for calls
        self.put_color = QColor('#FF3333')   # Softer red for puts
        self.atm_color = QColor('#FFA500')   # Orange for ATM (more like TOS)
        self.header_color = QColor('#333333')  # Dark gray for header
        self.bg_color_dark = QColor('#1C1C1C')  # Very dark gray (almost black) for background
        self.bg_color_alt = QColor('#252525')  # Slightly lighter dark gray for alternating rows

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._data)):
            return None

        row = index.row()
        col = index.column()
        column_name = self._headers[col]
        value = self._data.iloc[row][column_name]

        if role == Qt.ItemDataRole.DisplayRole:
            # Check for NaN values
            if pd.isna(value):
                return "N/A"

            # Format contractSymbol specially
            if column_name == 'contractSymbol':
                return str(value)

            # Format numbers appropriately
            if isinstance(value, (int, float)):
                if column_name == 'strike':
                    return f"{value:.2f}"
                elif column_name in ['impliedVolatility']:
                    return f"{value:.2%}"
                elif column_name in ['calc_delta', 'calc_gamma', 'calc_vega', 'calc_theta', 'calc_vanna', 'calc_vomma']:
                    return f"{value:.4f}"
                elif column_name in ['volume', 'openInterest']:
                    return f"{int(value):,}"
                elif column_name in ['lastPrice', 'bid', 'ask']:
                    return f"${value:.2f}"
                else:
                    return f"{value}"
            return str(value)

        elif role == Qt.ItemDataRole.BackgroundRole:
            # Highlight ATM options with orange background
            if 'moneyness' in self._data.columns and self._data.iloc[row]['moneyness'] == 'ATM':
                return self.atm_color

            # Alternate row colors for better readability
            return self.bg_color_alt if row % 2 else self.bg_color_dark

        elif role == Qt.ItemDataRole.ForegroundRole:
            # Color text based on option type and moneyness
            if 'option_type' in self._data.columns:
                option_type = self._data.iloc[row]['option_type']

                # Check if this is an ATM option
                is_atm = False
                if 'moneyness' in self._data.columns:
                    is_atm = self._data.iloc[row]['moneyness'] == 'ATM'

                if not pd.isna(option_type):
                    # For ATM options, use white text for better visibility
                    if is_atm:
                        return QColor('white')
                    # Otherwise use the appropriate color based on option type
                    elif option_type == 'call':
                        return self.call_color
                    else:  # put
                        return self.put_color

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.DisplayRole:
                return section + 1
            elif role == Qt.ItemDataRole.BackgroundRole:
                # Dark background for row headers
                return self.header_color
            elif role == Qt.ItemDataRole.ForegroundRole:
                # White text for row headers
                return QColor('white')
            return None

        if orientation == Qt.Orientation.Horizontal and section < len(self._headers):
            if role == Qt.ItemDataRole.BackgroundRole:
                # Dark background for column headers
                return self.header_color
            elif role == Qt.ItemDataRole.ForegroundRole:
                # Color header text based on option type column
                header = self._headers[section]
                if header == 'option_type':
                    return QColor('white')  # White for option type header
                elif 'moneyness' in self._headers and header == 'moneyness':
                    return QColor('white')  # White for moneyness header
                elif header == 'strike':
                    return QColor('white')  # White for strike header
                else:
                    # Check if we have option_type in the data
                    if 'option_type' in self._data.columns and not self._data.empty:
                        # Get the first row's option type (assuming all rows have the same type)
                        option_type = self._data.iloc[0]['option_type']
                        if option_type == 'call':
                            return self.call_color
                        else:  # put
                            return self.put_color
                    return QColor('white')  # Default to white
            elif role == Qt.ItemDataRole.DisplayRole:
                # Format header names for better display
                header = self._headers[section]

                # Special case for contractSymbol
                if header == 'contractSymbol':
                    return 'Symbol'

                # Replace underscores with spaces and capitalize each word
                return ' '.join(word.capitalize() for word in header.split('_'))
        return None

    def setData(self, data, headers):
        self.beginResetModel()
        self._data = data
        self._headers = headers
        self.endResetModel()

class MplCanvas(FigureCanvas):
    """Matplotlib canvas with crosshair functionality"""
    def __init__(self, parent=None, width=5, height=4, dpi=100, is_3d=False):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if is_3d:
            self.axes = self.fig.add_subplot(111, projection='3d')
            self.fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)
        else:
            self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        self.is_3d = is_3d
        self.exposure_type = "VEX"
        self.colorbar = None

        # Store default crosshair style
        self.default_crosshair_style = {'color': 'gray', 'lw': 0.5, 'ls': '--'}
        # Store TRACE crosshair style (more visible)
        self.trace_crosshair_style = {'color': 'white', 'lw': 1.0, 'ls': ':'}

        # Initialize crosshair elements with default style
        try:
            # Get the middle of the current axes limits
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()
            mid_x = (xlim[0] + xlim[1]) / 2
            mid_y = (ylim[0] + ylim[1]) / 2

            # Create crosshair at the middle position
            self.cursor_hline = self.axes.axhline(y=mid_y,
                                                color=self.default_crosshair_style['color'],
                                                lw=self.default_crosshair_style['lw'],
                                                ls=self.default_crosshair_style['ls'])
            self.cursor_vline = self.axes.axvline(x=mid_x,
                                                color=self.default_crosshair_style['color'],
                                                lw=self.default_crosshair_style['lw'],
                                                ls=self.default_crosshair_style['ls'])
        except Exception as e:
            # If we can't create the crosshair, create it at position 0,0
            print(f"Could not create crosshair at middle position: {e}")
            self.cursor_hline = self.axes.axhline(y=0,
                                                color=self.default_crosshair_style['color'],
                                                lw=self.default_crosshair_style['lw'],
                                                ls=self.default_crosshair_style['ls'])
            self.cursor_vline = self.axes.axvline(x=0,
                                                color=self.default_crosshair_style['color'],
                                                lw=self.default_crosshair_style['lw'],
                                                ls=self.default_crosshair_style['ls'])

        # Handle text differently for 2D vs 3D axes
        if is_3d:
            # For 3D axes, we need to provide x, y, z coordinates
            self.text_box = self.axes.text(0.7, 0.95, 0, "", transform=self.axes.transAxes,
                                          bbox=dict(facecolor='black', alpha=0.5),
                                          color='white', fontsize=8)
        else:
            # For 2D axes, we only need x, y coordinates
            self.text_box = self.axes.text(0.7, 0.95, "", transform=self.axes.transAxes,
                                          bbox=dict(facecolor='black', alpha=0.5),
                                          color='white', fontsize=8)

        # Connect motion event
        self.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_mouse_move(self, event):
        if not event.inaxes or self.is_3d:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # For axhline and axvline, we need to use different methods
        try:
            # Apply appropriate style based on chart type
            if hasattr(self, 'exposure_type') and self.exposure_type == "TRACE":
                # Use TRACE style for crosshair
                style = self.trace_crosshair_style
            else:
                # Use default style for crosshair
                style = self.default_crosshair_style

            # Safely remove old lines if they exist
            try:
                if self.cursor_hline in self.axes.lines:
                    self.cursor_hline.remove()
                if self.cursor_vline in self.axes.lines:
                    self.cursor_vline.remove()
            except Exception as e:
                # If we can't remove the lines, just continue
                print(f"Could not remove crosshair lines: {e}")
                pass

            # Create new lines with the appropriate style
            self.cursor_hline = self.axes.axhline(y=y, color=style['color'],
                                                lw=style['lw'],
                                                ls=style['ls'])

            self.cursor_vline = self.axes.axvline(x=x, color=style['color'],
                                                lw=style['lw'],
                                                ls=style['ls'])

            # Update text box based on chart type
            if hasattr(self, 'exposure_type') and self.exposure_type == "TRACE":
                # For TRACE chart, show more detailed information
                # For TRACE, x is typically time and y is price
                self.text_box.set_text(f"Time: {x:.2f}\nPrice: ${y:.2f}")
                # Move the text box to a better position for TRACE
                self.text_box.set_position((0.02, 0.02))
                self.text_box.set_transform(self.axes.transAxes)
                # Make the text box more visible with a darker background
                self.text_box.set_bbox(dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round'))
            else:
                # Default text for other charts
                self.text_box.set_text(f"x={x:.2f}, y={y:.2f}")

            self.draw_idle()
        except Exception as e:
            # Silently handle any errors with the crosshair
            print(f"Crosshair error: {e}")
            pass


class DataFetchThread(QThread):
    """Thread for fetching data to keep UI responsive"""
    data_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ticker, expiry_date):
        super().__init__()
        self.ticker = ticker
        self.expiry_date = expiry_date

    def run(self):
        try:
            # Format ticker for indices
            self.ticker = self.format_ticker(self.ticker)

            # Get current price
            S = self.get_current_price(self.ticker)
            if S is None or S <= 0:
                self.error.emit("Could not fetch current price or price is invalid.")
                return

            # Fetch options data
            calls, puts = self.fetch_options_for_date(self.ticker, self.expiry_date, S)

            if calls.empty and puts.empty:
                self.error.emit("No options data available for the selected date.")
                return

            try:
                # Calculate Greeks
                calls, puts = self.compute_greeks(calls, puts, S)

                # Emit the results (include max_pain_strike if it exists)
                max_pain_strike = getattr(self, 'max_pain_strike', S)
                self.data_ready.emit((calls, puts, S, max_pain_strike))
            except ZeroDivisionError as e:
                self.error.emit(f"Division by zero error while calculating Greeks: {str(e)}")
            except Exception as e:
                self.error.emit(f"Error calculating Greeks: {str(e)}")

        except ZeroDivisionError as e:
            self.error.emit(f"Division by zero error: {str(e)}")
        except Exception as e:
            self.error.emit(f"Error fetching data: {str(e)}")

    def format_ticker(self, ticker):
        """Helper function to format tickers for indices"""
        ticker = ticker.upper()
        if ticker == "SPX":
            return "^SPX"
        elif ticker == "NDX":
            return "^NDX"
        elif ticker == "VIX":
            return "^VIX"
        elif ticker == "DJI":
            return "^DJI"
        elif ticker == "RUT":
            return "^RUT"
        return ticker

    def get_current_price(self, ticker):
        """Get current price with fallback logic"""
        formatted_ticker = ticker.replace('%5E', '^')

        if formatted_ticker in ['^SPX'] or ticker in ['%5ESPX', 'SPX']:
            try:
                gspc = yf.Ticker('^GSPC')
                price = gspc.info.get("regularMarketPrice")
                if price is None:
                    price = gspc.fast_info.get("lastPrice")
                if price is not None:
                    return round(float(price), 2)
            except Exception as e:
                print(f"Error fetching SPX price: {str(e)}")

        try:
            stock = yf.Ticker(ticker)
            price = stock.info.get("regularMarketPrice")
            if price is None:
                price = stock.fast_info.get("lastPrice")
            if price is not None:
                return round(float(price), 2)
        except Exception as e:
            print(f"Yahoo Finance error: {str(e)}")

        return None

    def fetch_options_for_date(self, ticker, date, S=None):
        """Fetches option chains for the given ticker and date."""
        print(f"Fetching option chain for {ticker} EXP {date}")
        stock = yf.Ticker(ticker)
        try:
            if S is None:
                S = self.get_current_price(ticker)
            chain = stock.option_chain(date)
            calls = chain.calls
            puts = chain.puts

            # Add additional data for better analysis
            # Calculate volume/OI ratio (a measure of activity)
            if 'volume' in calls.columns and 'openInterest' in calls.columns:
                calls['volume_oi_ratio'] = calls['volume'] / calls['openInterest'].replace(0, 1)  # Avoid division by zero
            if 'volume' in puts.columns and 'openInterest' in puts.columns:
                puts['volume_oi_ratio'] = puts['volume'] / puts['openInterest'].replace(0, 1)  # Avoid division by zero

            # Add moneyness column (how far from current price)
            if 'strike' in calls.columns and S is not None:
                calls['moneyness'] = (calls['strike'] - S) / S
            if 'strike' in puts.columns and S is not None:
                puts['moneyness'] = (puts['strike'] - S) / S

            # Add GEX, DEX, and CEX columns if they don't exist yet
            # These will be calculated in compute_greeks
            if 'GEX' not in calls.columns:
                calls['GEX'] = 0.0
            if 'DEX' not in calls.columns:
                calls['DEX'] = 0.0
            if 'CEX' not in calls.columns:
                calls['CEX'] = 0.0
            if 'GEX' not in puts.columns:
                puts['GEX'] = 0.0
            if 'DEX' not in puts.columns:
                puts['DEX'] = 0.0
            if 'CEX' not in puts.columns:
                puts['CEX'] = 0.0

            return calls, puts
        except Exception as e:
            print(f"Error fetching options chain for date {date}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def compute_greeks(self, calls, puts, S):
        """Compute Greeks for the options data"""
        # Get current date and calculate time to expiration
        today = datetime.today().date()
        selected_expiry = datetime.strptime(self.expiry_date, '%Y-%m-%d').date()
        t_days = max((selected_expiry - today).days, 1)  # Ensure at least 1 day
        t = t_days / 365.0

        # Risk-free rate (could be fetched from a source like ^IRX)
        r = 0.05  # Using 5% as a default

        # Create deep copies to avoid modifying original dataframes
        calls = calls.copy()
        puts = puts.copy()

        # Initialize columns for exposure calculations
        calls['GEX'] = 0.0  # Gamma Exposure
        calls['DEX'] = 0.0  # Delta Exposure
        calls['CEX'] = 0.0  # Charm Exposure (delta decay)
        puts['GEX'] = 0.0  # Gamma Exposure
        puts['DEX'] = 0.0  # Delta Exposure
        puts['CEX'] = 0.0  # Charm Exposure (delta decay)

        # Define function to compute vanna
        def compute_vanna(row, _):
            sigma = row.get("impliedVolatility", None)
            if sigma is None or sigma <= 0:
                return None
            try:
                # Calculate d1 and d2
                K = row["strike"]
                # Avoid division by zero in log(S/K)
                if S <= 0 or K <= 0:
                    return None

                # Avoid division by zero in sqrt(t)
                if t <= 0:
                    return None

                d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
                d2 = d1 - sigma * sqrt(t)

                # Calculate vanna
                # Avoid division by zero
                if abs(sigma) < 1e-10:
                    return None

                try:
                    vanna_val = -norm.pdf(d1) * d2 / sigma
                    return vanna_val
                except ZeroDivisionError:
                    print("Zero division in vanna calculation")
                    return None
            except Exception as e:
                print(f"Vanna calculation error: {e}")
                return None

        # Define function to compute gamma
        def compute_gamma(row, flag):
            sigma = row.get("impliedVolatility", None)
            if sigma is None or sigma <= 0:
                return None
            try:
                # Calculate gamma using py_vollib
                K = row["strike"]
                gamma_val = bs_gamma(flag, S, K, t, r, sigma)
                return gamma_val
            except Exception:
                return None

        # Define function to compute delta
        def compute_delta(row, flag):
            sigma = row.get("impliedVolatility", None)
            if sigma is None or sigma <= 0:
                return None
            try:
                # Calculate delta using py_vollib
                K = row["strike"]
                delta_val = bs_delta(flag, S, K, t, r, sigma)
                return delta_val
            except Exception:
                return None

        # Define function to compute charm (delta decay)
        def compute_charm(row, flag):
            sigma = row.get("impliedVolatility", None)
            if sigma is None or sigma <= 0:
                return None
            try:
                # Calculate charm (delta decay) using first principles
                K = row["strike"]
                # Avoid division by zero in log(S/K)
                if S <= 0 or K <= 0:
                    return None

                # Avoid division by zero in sqrt(t)
                if t <= 0:
                    return None

                # Calculate d1 and d2 from Black-Scholes
                d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
                d2 = d1 - sigma * sqrt(t)

                # Avoid division by zero
                if abs(sigma) < 1e-10 or abs(t) < 1e-10:
                    return None

                try:
                    # Charm formula depends on option type
                    if flag == 'c':
                        # Call option charm
                        charm_val = -norm.pdf(d1) * (2 * (r - 0) / (sigma * sqrt(t)) - d2 / (2 * t))
                    else:
                        # Put option charm
                        charm_val = -norm.pdf(d1) * (2 * (r - 0) / (sigma * sqrt(t)) - d2 / (2 * t))
                        # For puts, we negate the charm to match the convention
                        charm_val = -charm_val

                    # Scale charm to daily basis (from annual)
                    charm_val = charm_val / 365.0
                    return charm_val
                except ZeroDivisionError:
                    print("Zero division in charm calculation")
                    return None
            except Exception as e:
                print(f"Charm calculation error: {e}")
                return None

        # Define function to compute theta
        def compute_theta(row, flag):
            sigma = row.get("impliedVolatility", None)
            if sigma is None or sigma <= 0:
                return None
            try:
                # Calculate theta using py_vollib
                K = row["strike"]
                theta_val = bs_theta(flag, S, K, t, r, sigma)
                # Convert to daily theta (from annual)
                theta_val = theta_val / 365.0
                return theta_val
            except Exception:
                return None

        # Define function to compute vega
        def compute_vega(row, flag):
            sigma = row.get("impliedVolatility", None)
            if sigma is None or sigma <= 0:
                return None
            try:
                # Calculate vega using py_vollib
                K = row["strike"]
                vega_val = bs_vega(flag, S, K, t, r, sigma)
                return vega_val
            except Exception:
                return None

        # Define function to compute vomma (second derivative of option price with respect to volatility)
        def compute_vomma(row, _):  # Using _ for unused parameter
            sigma = row.get("impliedVolatility", None)
            if sigma is None or sigma <= 0:
                return None
            try:
                # Calculate d1 and d2
                K = row["strike"]
                # Avoid division by zero in log(S/K)
                if S <= 0 or K <= 0:
                    return None

                # Avoid division by zero in sqrt(t)
                if t <= 0:
                    return None

                d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
                d2 = d1 - sigma * sqrt(t)

                # Calculate vomma (derivative of vega with respect to volatility)
                # Vomma = vega * (d1*d2 - 1) / sigma
                # Note: flag parameter is not used as vomma calculation is the same for calls and puts
                vega_val = S * sqrt(t) * norm.pdf(d1)

                # Avoid division by zero
                if abs(sigma) < 1e-10:
                    return None

                try:
                    vomma_val = vega_val * (d1*d2 - 1) / sigma
                    return vomma_val
                except ZeroDivisionError:
                    print("Zero division in vomma calculation")
                    return None
            except Exception as e:
                print(f"Vomma calculation error: {e}")
                return None

        # Calculate vanna, gamma, delta, charm, theta, vega, and vomma for calls and puts
        calls["calc_vanna"] = calls.apply(lambda row: compute_vanna(row, "c"), axis=1)
        puts["calc_vanna"] = puts.apply(lambda row: compute_vanna(row, "p"), axis=1)

        calls["calc_gamma"] = calls.apply(lambda row: compute_gamma(row, "c"), axis=1)
        puts["calc_gamma"] = puts.apply(lambda row: compute_gamma(row, "p"), axis=1)

        calls["calc_delta"] = calls.apply(lambda row: compute_delta(row, "c"), axis=1)
        puts["calc_delta"] = puts.apply(lambda row: compute_delta(row, "p"), axis=1)

        calls["calc_charm"] = calls.apply(lambda row: compute_charm(row, "c"), axis=1)
        puts["calc_charm"] = puts.apply(lambda row: compute_charm(row, "p"), axis=1)

        calls["calc_theta"] = calls.apply(lambda row: compute_theta(row, "c"), axis=1)
        puts["calc_theta"] = puts.apply(lambda row: compute_theta(row, "p"), axis=1)

        calls["calc_vega"] = calls.apply(lambda row: compute_vega(row, "c"), axis=1)
        puts["calc_vega"] = puts.apply(lambda row: compute_vega(row, "p"), axis=1)

        calls["calc_vomma"] = calls.apply(lambda row: compute_vomma(row, "c"), axis=1)
        puts["calc_vomma"] = puts.apply(lambda row: compute_vomma(row, "p"), axis=1)

        # Drop rows with NaN values
        calls = calls.dropna(subset=["calc_vanna", "calc_gamma", "calc_delta", "calc_charm", "calc_theta", "calc_vega", "calc_vomma"])
        puts = puts.dropna(subset=["calc_vanna", "calc_gamma", "calc_delta", "calc_charm", "calc_theta", "calc_vega", "calc_vomma"])

        # Calculate Vanna Exposure (VEX), Gamma Exposure (GEX), Delta Exposure (DEX), Charm Exposure (CEX), Theta Exposure (TEX), Vega Exposure (VEGX), Vomma Exposure (VOMX), and Max Pain (MPX)
        calls["VEX"] = calls["calc_vanna"] * calls["openInterest"] * 100
        puts["VEX"] = puts["calc_vanna"] * puts["openInterest"] * 100

        # For GEX, make put values negative to display them below the x-axis like VEX
        calls["GEX"] = calls["calc_gamma"] * calls["openInterest"] * 100
        puts["GEX"] = -1 * puts["calc_gamma"] * puts["openInterest"] * 100

        # Also calculate GEX using volume for more accurate flow visualization
        calls["GEX_vol"] = calls["calc_gamma"] * calls["volume"] * 100
        puts["GEX_vol"] = -1 * puts["calc_gamma"] * puts["volume"] * 100

        # Volatility trigger will be calculated in the main class

        # For VEGX, make put values negative to display them below the x-axis
        calls["VEGX"] = calls["calc_vega"] * calls["openInterest"] * 100
        puts["VEGX"] = -1 * puts["calc_vega"] * puts["openInterest"] * 100

        # For VOMX, make put values negative to display them below the x-axis
        calls["VOMX"] = calls["calc_vomma"] * calls["openInterest"] * 100
        puts["VOMX"] = -1 * puts["calc_vomma"] * puts["openInterest"] * 100

        # Calculate IV Skew (Call IV - Put IV) by strike
        # First, create dictionaries to store IV by strike for calls and puts
        call_iv_by_strike = {}
        put_iv_by_strike = {}

        # Aggregate IV by strike
        for _, row in calls.iterrows():
            strike = row['strike']
            iv = row.get('impliedVolatility', 0)
            if strike not in call_iv_by_strike:
                call_iv_by_strike[strike] = []
            call_iv_by_strike[strike].append(iv)

        for _, row in puts.iterrows():
            strike = row['strike']
            iv = row.get('impliedVolatility', 0)
            if strike not in put_iv_by_strike:
                put_iv_by_strike[strike] = []
            put_iv_by_strike[strike].append(iv)

        # For DEX, make put values negative to display them below the x-axis
        # Note: Put delta is already negative, but we need to ensure it's displayed below the x-axis
        calls["DEX"] = calls["calc_delta"] * calls["openInterest"] * 100
        puts["DEX"] = puts["calc_delta"] * puts["openInterest"] * 100  # Put delta is already negative

        # Also calculate DEX using volume for more accurate flow visualization
        calls["DEX_vol"] = calls["calc_delta"] * calls["volume"] * 100
        puts["DEX_vol"] = puts["calc_delta"] * puts["volume"] * 100

        # For CEX (Charm Exposure), put values should be displayed below the x-axis
        # Note: Charm for puts is already adjusted in the compute_charm function
        calls["CEX"] = calls["calc_charm"] * calls["openInterest"] * 100
        puts["CEX"] = puts["calc_charm"] * puts["openInterest"] * 100

        # Also calculate CEX using volume for more accurate flow visualization
        calls["CEX_vol"] = calls["calc_charm"] * calls["volume"] * 100
        puts["CEX_vol"] = puts["calc_charm"] * puts["volume"] * 100

        # For TEX (Theta Exposure), put values should be displayed below the x-axis
        # Note: Theta is already negative for both calls and puts, so we need to adjust the sign for visualization
        # We'll make calls positive and puts negative for consistent visualization
        calls["TEX"] = -1 * calls["calc_theta"] * calls["openInterest"] * 100  # Invert sign to make calls positive
        puts["TEX"] = puts["calc_theta"] * puts["openInterest"] * 100  # Keep puts negative

        # Calculate IV Skew (Call IV - Put IV) for each strike
        all_strikes = sorted(set(call_iv_by_strike.keys()) | set(put_iv_by_strike.keys()))

        for strike in all_strikes:
            # Calculate average IV for calls and puts at this strike
            call_iv_avg = np.mean(call_iv_by_strike.get(strike, [0])) if strike in call_iv_by_strike else 0
            put_iv_avg = np.mean(put_iv_by_strike.get(strike, [0])) if strike in put_iv_by_strike else 0

            # Calculate IV skew (call IV - put IV)
            iv_skew = call_iv_avg - put_iv_avg

            # Add IV skew to calls dataframe for this strike
            calls.loc[calls['strike'] == strike, 'IVSKEW'] = iv_skew

            # Add IV skew to puts dataframe for this strike
            puts.loc[puts['strike'] == strike, 'IVSKEW'] = iv_skew

        # Calculate Put/Call Ratio (PCR) by strike
        # First, create dictionaries to store OI by strike for calls and puts
        call_oi_by_strike = {}
        put_oi_by_strike = {}

        # Aggregate OI by strike
        for _, row in calls.iterrows():
            strike = row['strike']
            if strike not in call_oi_by_strike:
                call_oi_by_strike[strike] = 0
            call_oi_by_strike[strike] += row['openInterest']

        for _, row in puts.iterrows():
            strike = row['strike']
            if strike not in put_oi_by_strike:
                put_oi_by_strike[strike] = 0
            put_oi_by_strike[strike] += row['openInterest']

        # Calculate PCR for each strike and add to dataframes
        all_strikes = sorted(set(call_oi_by_strike.keys()) | set(put_oi_by_strike.keys()))

        for strike in all_strikes:
            call_oi = call_oi_by_strike.get(strike, 0.001)  # Avoid division by zero
            put_oi = put_oi_by_strike.get(strike, 0)

            # Extra safety check for division by zero
            if call_oi < 0.001:
                call_oi = 0.001

            try:
                pcr = put_oi / call_oi
            except ZeroDivisionError:
                pcr = 0 if put_oi == 0 else 999  # Use 0 if no puts, otherwise use a high value

            # Add PCR to calls dataframe for this strike
            calls.loc[calls['strike'] == strike, 'PCR'] = pcr

            # Add PCR to puts dataframe for this strike
            puts.loc[puts['strike'] == strike, 'PCR'] = pcr

        # Calculate Max Pain (MPX)
        # For each strike, calculate the total pain for option writers
        unique_strikes = sorted(set(calls['strike'].unique()) | set(puts['strike'].unique()))

        # Initialize MPX columns
        calls["MPX"] = 0
        puts["MPX"] = 0

        # Dictionary to store total pain at each strike
        total_pain_by_strike = {}

        # Calculate pain at each strike
        for strike in unique_strikes:
            # Call pain at this strike (loss to option writers if price ends at this strike)
            call_pain = calls[calls['strike'] <= strike]['openInterest'] * (strike - calls[calls['strike'] <= strike]['strike'])
            call_pain_sum = call_pain.sum()

            # Update the MPX value for calls at this strike
            for idx in calls[calls['strike'] == strike].index:
                calls.at[idx, "MPX"] = call_pain_sum

            # Put pain at this strike (loss to option writers if price ends at this strike)
            put_pain = puts[puts['strike'] >= strike]['openInterest'] * (puts[puts['strike'] >= strike]['strike'] - strike)
            put_pain_sum = put_pain.sum()

            # Update the MPX value for puts at this strike
            for idx in puts[puts['strike'] == strike].index:
                puts.at[idx, "MPX"] = put_pain_sum

            # Store total pain at this strike
            total_pain_by_strike[strike] = call_pain_sum + put_pain_sum

        # Find the strike with minimum total pain (max pain point)
        if total_pain_by_strike:
            self.max_pain_strike = min(total_pain_by_strike.items(), key=lambda x: x[1])[0]
        else:
            self.max_pain_strike = S  # Default to current price if no data

        return calls, puts

class OptionsExposureDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Options Exposure Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # Default colors for calls and puts
        self.call_color = '#00FF00'  # Green
        self.put_color = '#FF0000'   # Red

        # Default strike range
        self.strike_range = 20

        # Default exposure type
        self.exposure_type = "VEX"  # VEX, GEX, DEX, CEX, TEX, VEGX, MPX, PCR, DELTA_PROFILE, VEGA_PROFILE, ACTIVITY_MAP, or TRACE

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create compact top controls with improved styling
        top_controls = QHBoxLayout()
        top_controls.setContentsMargins(0, 0, 0, 0)  # Remove margins
        top_controls.setSpacing(4)  # Reduce spacing further

        # Create a frame for top controls to give it a consistent look
        top_frame = QFrame()
        top_frame.setFrameShape(QFrame.Shape.StyledPanel)
        top_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 3px;
            }
            QLabel {
                color: #ddd;
                font-size: 10px;
                font-weight: bold;
            }
            QLineEdit, QComboBox {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 2px;
                padding: 2px;
                font-size: 10px;
            }
            QPushButton {
                background-color: #2d4263;
                color: white;
                border: 1px solid #555;
                border-radius: 2px;
                padding: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a5380;
            }
            QPushButton:pressed {
                background-color: #1e2d40;
            }
        """)

        top_frame_layout = QHBoxLayout(top_frame)
        top_frame_layout.setContentsMargins(5, 3, 5, 3)  # Very small margins
        top_frame_layout.setSpacing(4)  # Small spacing

        # Ticker input - more compact
        ticker_label = QLabel("Ticker:")
        ticker_label.setMaximumWidth(38)  # Reduce width
        self.ticker_input = QLineEdit("SPY")
        self.ticker_input.setMaximumWidth(55)  # Reduce width

        # Expiry date selector - more compact
        expiry_label = QLabel("Exp:")
        expiry_label.setMaximumWidth(22)  # Reduce width
        self.expiry_selector = QComboBox()
        self.expiry_selector.setMaximumWidth(110)  # Reduce width

        # Refresh button - more compact
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setMaximumWidth(55)  # Reduce width
        self.refresh_button.clicked.connect(self.refresh_data)

        # Add widgets to top frame layout
        top_frame_layout.addWidget(ticker_label)
        top_frame_layout.addWidget(self.ticker_input)
        top_frame_layout.addWidget(expiry_label)
        top_frame_layout.addWidget(self.expiry_selector)
        top_frame_layout.addWidget(self.refresh_button)
        top_frame_layout.addStretch()

        # Add the frame to top controls
        top_controls.addWidget(top_frame)
        top_controls.addStretch()

        # Exposure type tabs - more compact and organized
        self.exposure_tabs = QTabWidget()
        self.exposure_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.exposure_tabs.setMaximumHeight(50)  # Increased height to accommodate group labels
        self.exposure_tabs.setDocumentMode(True)  # More modern look
        self.exposure_tabs.setElideMode(Qt.TextElideMode.ElideNone)  # Don't cut off tab text
        self.exposure_tabs.setUsesScrollButtons(True)  # Add scroll buttons if needed
        self.exposure_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                background: #1e1e1e;
                border-radius: 3px;
                top: -1px; /* Overlap with tabs */
            }
            QTabBar::tab {
                background: #333;
                color: white;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 90px;
                max-width: 110px;
                text-align: center;
                margin-right: 1px; /* Space between tabs */
            }
            QTabBar::tab:selected {
                background: #1e1e1e;
                border-bottom: 1px solid #1e1e1e;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #444;
            }
            /* Group styling */
            QTabBar::tab:first {
                margin-left: 5px; /* Extra space before first tab */
            }
            /* Group 1: Basic exposure metrics */
            QTabBar::tab:nth-child(1), QTabBar::tab:nth-child(2), QTabBar::tab:nth-child(3), QTabBar::tab:nth-child(4) {
                background-color: #2d4263;
                border-top: 3px solid #4a69bd;
            }
            QTabBar::tab:nth-child(4) {
                margin-right: 12px; /* Space after group */
                border-right: 2px solid #555;
            }
            /* Group 2: Advanced exposure metrics */
            QTabBar::tab:nth-child(5), QTabBar::tab:nth-child(6), QTabBar::tab:nth-child(7), QTabBar::tab:nth-child(8) {
                background-color: #264653;
                border-top: 3px solid #2a9d8f;
            }
            QTabBar::tab:nth-child(8) {
                margin-right: 12px; /* Space after group */
                border-right: 2px solid #555;
            }
            /* Group 3: Profile visualizations */
            QTabBar::tab:nth-child(9), QTabBar::tab:nth-child(10), QTabBar::tab:nth-child(11) {
                background-color: #2a9d8f;
                border-top: 3px solid #e9c46a;
            }
            QTabBar::tab:nth-child(11) {
                margin-right: 12px; /* Space after group */
                border-right: 2px solid #555;
            }
            /* Group 4: 3D visualizations */
            QTabBar::tab:nth-child(12), QTabBar::tab:nth-child(13), QTabBar::tab:nth-child(14) {
                background-color: #e76f51;
                border-top: 3px solid #f4a261;
            }
            QTabBar::tab:nth-child(14) {
                margin-right: 12px; /* Space after group */
                border-right: 2px solid #555;
            }
            /* Group 5: Market data visualizations */
            QTabBar::tab:nth-child(15), QTabBar::tab:nth-child(16), QTabBar::tab:nth-child(17), QTabBar::tab:nth-child(18), QTabBar::tab:nth-child(19) {
                background-color: #6a994e;
                border-top: 3px solid #a7c957;
            }
            QTabBar::tab:nth-child(19) {
                margin-right: 12px; /* Space after group */
                border-right: 2px solid #555;
            }
            /* Group 6: TRACE visualization */
            QTabBar::tab:nth-child(20) {
                background-color: #7209b7;
                border-top: 3px solid #f72585;
            }
            /* Selected tab styling for all groups */
            QTabBar::tab:selected {
                background: #1e1e1e;
                border-bottom: 1px solid #1e1e1e;
            }
        """)

        # Create empty tabs for each exposure type (removing descriptions to save space)
        vanna_tab = QWidget()
        gamma_tab = QWidget()
        delta_tab = QWidget()
        delta_profile_tab = QWidget()  # New tab for Delta Profile
        charm_tab = QWidget()
        theta_tab = QWidget()  # New tab for Theta Decay
        vega_tab = QWidget()  # New tab for Vega Exposure
        vega_profile_tab = QWidget()  # New tab for Vega Profile
        vomma_tab = QWidget()  # New tab for Vomma Exposure
        vomma_profile_tab = QWidget()  # New tab for Vomma Profile
        max_pain_tab = QWidget()
        oi_tab = QWidget()  # Tab for Open Interest
        pcr_tab = QWidget()  # New tab for Put/Call Ratio
        gamma_landscape_tab = QWidget()  # New tab for Gamma Landscape
        greek_landscape_tab = QWidget()  # New tab for Greek Landscape
        iv_skew_tab = QWidget()  # New tab for IV Skew
        iv_surface_tab = QWidget()  # New tab for IV Surface
        activity_map_tab = QWidget()  # New tab for Options Chain Activity Map
        candlestick_tab = QWidget()  # New tab for Candlestick Chart
        options_chain_tab = QWidget()  # New tab for Options Chain

        # Add tabs to the tab widget in a logical order

        # Group 1: Basic exposure metrics (most commonly used)
        delta_idx = self.exposure_tabs.addTab(delta_tab, "Delta (DEX)")
        self.exposure_tabs.setTabToolTip(delta_idx, "Delta Exposure: Shows price sensitivity of options")

        gamma_idx = self.exposure_tabs.addTab(gamma_tab, "Gamma (GEX)")
        self.exposure_tabs.setTabToolTip(gamma_idx, "Gamma Exposure: Shows rate of change of delta with respect to price")

        theta_idx = self.exposure_tabs.addTab(theta_tab, "Theta (TEX)")
        self.exposure_tabs.setTabToolTip(theta_idx, "Theta Exposure: Shows time decay of options value")

        vega_idx = self.exposure_tabs.addTab(vega_tab, "Vega (VEGX)")
        self.exposure_tabs.setTabToolTip(vega_idx, "Vega Exposure: Shows sensitivity to volatility changes")

        # Group 2: Advanced exposure metrics
        vanna_idx = self.exposure_tabs.addTab(vanna_tab, "Vanna (VEX)")
        self.exposure_tabs.setTabToolTip(vanna_idx, "Vanna Exposure: Shows sensitivity of delta to volatility changes")

        charm_idx = self.exposure_tabs.addTab(charm_tab, "Charm (CEX)")
        self.exposure_tabs.setTabToolTip(charm_idx, "Charm Exposure: Shows rate of change of delta over time")

        vomma_idx = self.exposure_tabs.addTab(vomma_tab, "Vomma (VOMX)")
        self.exposure_tabs.setTabToolTip(vomma_idx, "Vomma Exposure: Shows sensitivity of vega to volatility changes")

        max_pain_idx = self.exposure_tabs.addTab(max_pain_tab, "Max Pain (MPX)")
        self.exposure_tabs.setTabToolTip(max_pain_idx, "Max Pain: Shows price level where option holders experience maximum loss")

        # Group 3: Profile visualizations
        delta_profile_idx = self.exposure_tabs.addTab(delta_profile_tab, "Delta Profile")
        self.exposure_tabs.setTabToolTip(delta_profile_idx, "Delta Profile: Shows distribution of delta across strikes")

        vega_profile_idx = self.exposure_tabs.addTab(vega_profile_tab, "Vega Profile")
        self.exposure_tabs.setTabToolTip(vega_profile_idx, "Vega Profile: Shows distribution of vega across strikes")

        vomma_profile_idx = self.exposure_tabs.addTab(vomma_profile_tab, "Vomma Profile")
        self.exposure_tabs.setTabToolTip(vomma_profile_idx, "Vomma Profile: Shows distribution of vomma across strikes")

        # Group 4: 3D visualizations
        gamma_landscape_idx = self.exposure_tabs.addTab(gamma_landscape_tab, "Gamma Landscape")
        self.exposure_tabs.setTabToolTip(gamma_landscape_idx, "Gamma Landscape: 3D visualization of gamma across strikes and volatility")

        greek_landscape_idx = self.exposure_tabs.addTab(greek_landscape_tab, "Greek Landscape")
        self.exposure_tabs.setTabToolTip(greek_landscape_idx, "Greek Landscape: 3D visualization of selected greek across strikes and time")

        iv_surface_idx = self.exposure_tabs.addTab(iv_surface_tab, "IV Surface")
        self.exposure_tabs.setTabToolTip(iv_surface_idx, "IV Surface: 3D visualization of implied volatility across strikes and time")

        # Group 5: Market data visualizations
        oi_idx = self.exposure_tabs.addTab(oi_tab, "Open Interest (OI)")
        self.exposure_tabs.setTabToolTip(oi_idx, "Open Interest: Shows number of outstanding option contracts")

        pcr_idx = self.exposure_tabs.addTab(pcr_tab, "Put/Call Ratio (PCR)")
        self.exposure_tabs.setTabToolTip(pcr_idx, "Put/Call Ratio: Shows ratio of put to call options as a sentiment indicator")

        iv_skew_idx = self.exposure_tabs.addTab(iv_skew_tab, "IV Skew")
        self.exposure_tabs.setTabToolTip(iv_skew_idx, "IV Skew: Shows difference in implied volatility across strikes")

        activity_map_idx = self.exposure_tabs.addTab(activity_map_tab, "Activity Map")
        self.exposure_tabs.setTabToolTip(activity_map_idx, "Activity Map: Heatmap of options chain activity")

        candlestick_idx = self.exposure_tabs.addTab(candlestick_tab, "Candlestick Chart")
        self.exposure_tabs.setTabToolTip(candlestick_idx, "Candlestick Chart: Price chart with optional overlays")

        # Group 6: TRACE visualization
        trace_tab = QWidget()
        trace_idx = self.exposure_tabs.addTab(trace_tab, "TRACE")
        self.exposure_tabs.setTabToolTip(trace_idx, "TRACE: Time-series visualization of options exposure")

        # Group 7: Options Chain
        options_chain_idx = self.exposure_tabs.addTab(options_chain_tab, "Options Chain")
        self.exposure_tabs.setTabToolTip(options_chain_idx, "Options Chain: Full options chain data in tabular format")

        # Reorder tabs for better organization
        self.reorder_tabs()

        # Add group labels
        self.add_tab_group_labels()

        # Connect tab change signal
        self.exposure_tabs.currentChanged.connect(self.update_exposure_type_from_tab)

        # We'll set up the options chain tab later after creating the container

        # Chart type selector
        chart_type_group = QGroupBox("Chart Type")
        chart_type_layout = QHBoxLayout()
        self.chart_type_selector = QComboBox()
        self.chart_type_selector.addItems(["Bar", "Line", "Scatter", "Area"])
        self.chart_type_selector.currentTextChanged.connect(self.update_chart)
        chart_type_layout.addWidget(self.chart_type_selector)
        chart_type_group.setLayout(chart_type_layout)

        # Display options
        display_options_group = QGroupBox("Display Options")
        display_options_layout = QHBoxLayout()

        self.show_calls_checkbox = QCheckBox("Show Calls")
        self.show_calls_checkbox.setChecked(True)
        self.show_calls_checkbox.stateChanged.connect(self.update_chart)

        self.show_puts_checkbox = QCheckBox("Show Puts")
        self.show_puts_checkbox.setChecked(True)
        self.show_puts_checkbox.stateChanged.connect(self.update_chart)

        self.show_net_checkbox = QCheckBox("Show Net")
        self.show_net_checkbox.setChecked(True)  # Always checked by default
        self.show_net_checkbox.stateChanged.connect(self.update_chart)

        self.show_vol_trigger_checkbox = QCheckBox("Vol Trigger")
        self.show_vol_trigger_checkbox.setChecked(True)  # Checked by default
        self.show_vol_trigger_checkbox.setToolTip("Show SpotGamma's Volatility Trigger level")
        self.show_vol_trigger_checkbox.stateChanged.connect(self.update_chart)

        display_options_layout.addWidget(self.show_calls_checkbox)
        display_options_layout.addWidget(self.show_puts_checkbox)
        display_options_layout.addWidget(self.show_net_checkbox)
        display_options_layout.addWidget(self.show_vol_trigger_checkbox)
        display_options_group.setLayout(display_options_layout)

        # Strike range control
        strike_range_group = QGroupBox("Strike Range")
        strike_range_layout = QHBoxLayout()
        strike_range_label = QLabel("Range:")
        self.strike_range_input = QLineEdit(str(self.strike_range))
        self.strike_range_input.setMaximumWidth(50)
        self.strike_range_input.textChanged.connect(self.update_strike_range)
        strike_range_layout.addWidget(strike_range_label)
        strike_range_layout.addWidget(self.strike_range_input)
        strike_range_group.setLayout(strike_range_layout)

        # Gamma landscape controls
        self.gamma_landscape_group = QGroupBox("Gamma Landscape Settings")
        gamma_landscape_layout = QHBoxLayout()

        # Min volatility control
        min_vol_label = QLabel("Min Vol:")
        self.min_vol_input = QLineEdit("0.1")
        self.min_vol_input.setMaximumWidth(50)
        self.min_vol_input.textChanged.connect(self.update_gamma_landscape_settings)

        # Max volatility control
        max_vol_label = QLabel("Max Vol:")
        self.max_vol_input = QLineEdit("1.0")
        self.max_vol_input.setMaximumWidth(50)
        self.max_vol_input.textChanged.connect(self.update_gamma_landscape_settings)

        # Grid points control
        grid_points_label = QLabel("Grid:")
        self.grid_points_input = QLineEdit("50")
        self.grid_points_input.setMaximumWidth(50)
        self.grid_points_input.textChanged.connect(self.update_gamma_landscape_settings)

        # Option type selector
        option_type_label = QLabel("Type:")
        self.option_type_selector = QComboBox()
        self.option_type_selector.addItems(["Call", "Put"])
        self.option_type_selector.setMaximumWidth(70)
        self.option_type_selector.currentTextChanged.connect(self.update_gamma_landscape_settings)

        # View angle controls
        view_angle_label = QLabel("View:")
        self.view_angle_selector = QComboBox()
        self.view_angle_selector.addItems(["Default", "Top", "Side", "Front"])
        self.view_angle_selector.setMaximumWidth(70)
        self.view_angle_selector.currentTextChanged.connect(self.update_gamma_landscape_settings)

        # Add widgets to layout
        gamma_landscape_layout.addWidget(min_vol_label)
        gamma_landscape_layout.addWidget(self.min_vol_input)
        gamma_landscape_layout.addWidget(max_vol_label)
        gamma_landscape_layout.addWidget(self.max_vol_input)
        gamma_landscape_layout.addWidget(grid_points_label)
        gamma_landscape_layout.addWidget(self.grid_points_input)
        gamma_landscape_layout.addWidget(option_type_label)
        gamma_landscape_layout.addWidget(self.option_type_selector)
        gamma_landscape_layout.addWidget(view_angle_label)
        gamma_landscape_layout.addWidget(self.view_angle_selector)

        self.gamma_landscape_group.setLayout(gamma_landscape_layout)
        self.gamma_landscape_group.hide()  # Hide initially

        # Greek landscape controls
        self.greek_landscape_group = QGroupBox("Greek Landscape Settings")
        greek_landscape_layout = QHBoxLayout()

        # Greek type selector
        greek_type_label = QLabel("Greek:")
        self.greek_type_selector = QComboBox()
        self.greek_type_selector.addItems(["Delta", "Gamma", "Vega", "Theta", "Vomma"])
        self.greek_type_selector.setMaximumWidth(70)
        self.greek_type_selector.currentTextChanged.connect(self.update_greek_landscape_settings)

        # Time range controls
        min_time_label = QLabel("Min Days:")
        self.min_time_input = QLineEdit("1")
        self.min_time_input.setMaximumWidth(50)
        self.min_time_input.textChanged.connect(self.update_greek_landscape_settings)

        max_time_label = QLabel("Max Days:")
        self.max_time_input = QLineEdit("30")
        self.max_time_input.setMaximumWidth(50)
        self.max_time_input.textChanged.connect(self.update_greek_landscape_settings)

        # Grid points control
        grid_points_label_greek = QLabel("Grid:")
        self.grid_points_input_greek = QLineEdit("50")
        self.grid_points_input_greek.setMaximumWidth(50)
        self.grid_points_input_greek.textChanged.connect(self.update_greek_landscape_settings)

        # Option type selector
        option_type_label_greek = QLabel("Type:")
        self.option_type_selector_greek = QComboBox()
        self.option_type_selector_greek.addItems(["Call", "Put"])
        self.option_type_selector_greek.setMaximumWidth(70)
        self.option_type_selector_greek.currentTextChanged.connect(self.update_greek_landscape_settings)

        # View angle controls
        view_angle_label_greek = QLabel("View:")
        self.view_angle_selector_greek = QComboBox()
        self.view_angle_selector_greek.addItems(["Default", "Top", "Side", "Front"])
        self.view_angle_selector_greek.setMaximumWidth(70)
        self.view_angle_selector_greek.currentTextChanged.connect(self.update_greek_landscape_settings)

        # Add widgets to layout
        greek_landscape_layout.addWidget(greek_type_label)
        greek_landscape_layout.addWidget(self.greek_type_selector)
        greek_landscape_layout.addWidget(min_time_label)
        greek_landscape_layout.addWidget(self.min_time_input)
        greek_landscape_layout.addWidget(max_time_label)
        greek_landscape_layout.addWidget(self.max_time_input)
        greek_landscape_layout.addWidget(grid_points_label_greek)
        greek_landscape_layout.addWidget(self.grid_points_input_greek)
        greek_landscape_layout.addWidget(option_type_label_greek)
        greek_landscape_layout.addWidget(self.option_type_selector_greek)
        greek_landscape_layout.addWidget(view_angle_label_greek)
        greek_landscape_layout.addWidget(self.view_angle_selector_greek)

        self.greek_landscape_group.setLayout(greek_landscape_layout)
        self.greek_landscape_group.hide()  # Hide initially

        # IV Surface controls
        self.iv_surface_group = QGroupBox("IV Surface Settings")
        iv_surface_layout = QHBoxLayout()

        # Time range controls for IV Surface
        min_time_label_iv = QLabel("Min Days:")
        self.min_time_input_iv = QLineEdit("1")
        self.min_time_input_iv.setMaximumWidth(50)
        self.min_time_input_iv.textChanged.connect(self.update_iv_surface_settings)

        max_time_label_iv = QLabel("Max Days:")
        self.max_time_input_iv = QLineEdit("30")
        self.max_time_input_iv.setMaximumWidth(50)
        self.max_time_input_iv.textChanged.connect(self.update_iv_surface_settings)

        # Grid points control for IV Surface
        grid_points_label_iv = QLabel("Grid:")
        self.grid_points_input_iv = QLineEdit("50")
        self.grid_points_input_iv.setMaximumWidth(50)
        self.grid_points_input_iv.textChanged.connect(self.update_iv_surface_settings)

        # Option type selector for IV Surface
        option_type_label_iv = QLabel("Type:")
        self.option_type_selector_iv = QComboBox()
        self.option_type_selector_iv.addItems(["Call", "Put", "Average"])
        self.option_type_selector_iv.setMaximumWidth(70)
        self.option_type_selector_iv.currentTextChanged.connect(self.update_iv_surface_settings)

        # View angle controls for IV Surface
        view_angle_label_iv = QLabel("View:")
        self.view_angle_selector_iv = QComboBox()
        self.view_angle_selector_iv.addItems(["Default", "Top", "Side", "Front"])
        self.view_angle_selector_iv.setMaximumWidth(70)
        self.view_angle_selector_iv.currentTextChanged.connect(self.update_iv_surface_settings)

        # Colormap selector for IV Surface
        colormap_label_iv = QLabel("Color:")
        self.colormap_selector_iv = QComboBox()
        self.colormap_selector_iv.addItems(["coolwarm", "viridis", "plasma", "inferno", "magma", "jet"])
        self.colormap_selector_iv.setMaximumWidth(70)
        self.colormap_selector_iv.currentTextChanged.connect(self.update_iv_surface_settings)

        # Add widgets to layout
        iv_surface_layout.addWidget(min_time_label_iv)
        iv_surface_layout.addWidget(self.min_time_input_iv)
        iv_surface_layout.addWidget(max_time_label_iv)
        iv_surface_layout.addWidget(self.max_time_input_iv)
        iv_surface_layout.addWidget(grid_points_label_iv)
        iv_surface_layout.addWidget(self.grid_points_input_iv)
        iv_surface_layout.addWidget(option_type_label_iv)
        iv_surface_layout.addWidget(self.option_type_selector_iv)
        iv_surface_layout.addWidget(view_angle_label_iv)
        iv_surface_layout.addWidget(self.view_angle_selector_iv)
        iv_surface_layout.addWidget(colormap_label_iv)
        iv_surface_layout.addWidget(self.colormap_selector_iv)

        self.iv_surface_group.setLayout(iv_surface_layout)
        self.iv_surface_group.hide()  # Hide initially

        # Activity map controls
        self.activity_map_group = QGroupBox("Activity Map Settings")
        activity_map_layout = QHBoxLayout()

        # Activity metric selector
        activity_metric_label = QLabel("Metric:")
        self.activity_metric_selector = QComboBox()
        self.activity_metric_selector.addItems(["Volume", "Open Interest"])
        self.activity_metric_selector.setMaximumWidth(100)
        self.activity_metric_selector.currentTextChanged.connect(self.update_chart)

        # Color map selector
        colormap_label = QLabel("Color:")
        self.colormap_selector = QComboBox()
        self.colormap_selector.addItems(["hot", "viridis", "plasma", "inferno", "magma", "cividis"])
        self.colormap_selector.setMaximumWidth(100)
        self.colormap_selector.currentTextChanged.connect(self.update_chart)

        # Threshold for annotations
        threshold_label = QLabel("Threshold:")
        self.threshold_input = QLineEdit("0.5")
        self.threshold_input.setMaximumWidth(50)
        self.threshold_input.textChanged.connect(self.update_chart)

        # Add widgets to layout
        activity_map_layout.addWidget(activity_metric_label)
        activity_map_layout.addWidget(self.activity_metric_selector)
        activity_map_layout.addWidget(colormap_label)
        activity_map_layout.addWidget(self.colormap_selector)
        activity_map_layout.addWidget(threshold_label)
        activity_map_layout.addWidget(self.threshold_input)

        self.activity_map_group.setLayout(activity_map_layout)
        self.activity_map_group.hide()  # Hide initially

        # Candlestick chart controls
        self.candlestick_group = QGroupBox("Candlestick Settings")
        candlestick_layout = QHBoxLayout()

        # TRACE controls
        self.trace_group = QGroupBox("TRACE Settings")
        trace_layout = QHBoxLayout()

        # Timeframe selector
        timeframe_label = QLabel("Timeframe:")
        self.timeframe_selector = QComboBox()
        self.timeframe_selector.addItems(["1d", "1h", "30m", "15m", "5m", "1m"])
        self.timeframe_selector.setMaximumWidth(70)
        self.timeframe_selector.currentTextChanged.connect(self.update_candlestick_chart)

        # Days to Load input field
        period_label = QLabel("DTL:")
        self.days_to_load_input = QLineEdit("30")  # Default to 30 days
        self.days_to_load_input.setMaximumWidth(70)
        self.days_to_load_input.setToolTip("Number of days to load")
        # Connect to editingFinished to update only when user presses Enter or focus changes
        self.days_to_load_input.editingFinished.connect(self.update_candlestick_chart)

        # Candlestick type selector
        candlestick_type_label = QLabel("Type:")
        self.candlestick_type_selector = QComboBox()
        self.candlestick_type_selector.addItems(["Regular", "Hollow", "Heikin-Ashi"])
        self.candlestick_type_selector.setMaximumWidth(100)
        self.candlestick_type_selector.currentTextChanged.connect(self.update_candlestick_chart)

        # Volume display has been removed
        # Create hidden volume checkbox for compatibility with existing code
        self.show_volume_checkbox = QCheckBox("Show Volume")
        self.show_volume_checkbox.setChecked(False)
        self.show_volume_checkbox.hide()

        # MA controls removed
        # Create hidden MA controls for compatibility with existing code
        self.show_ma_checkbox = QCheckBox("Show MA")
        self.show_ma_checkbox.setChecked(False)
        self.show_ma_checkbox.hide()
        self.ma_period_input = QLineEdit("20")
        self.ma_period_input.hide()

        # Add widgets to layout
        candlestick_layout.addWidget(timeframe_label)
        candlestick_layout.addWidget(self.timeframe_selector)
        candlestick_layout.addWidget(period_label)
        candlestick_layout.addWidget(self.days_to_load_input)
        candlestick_layout.addWidget(candlestick_type_label)
        candlestick_layout.addWidget(self.candlestick_type_selector)

        self.candlestick_group.setLayout(candlestick_layout)
        self.candlestick_group.hide()  # Hide initially

        # TRACE heatmap type selector
        trace_type_label = QLabel("Heatmap:")
        self.trace_type_selector = QComboBox()
        self.trace_type_selector.addItems(["Gamma", "Delta Pressure", "Charm Pressure"])
        self.trace_type_selector.setMaximumWidth(120)
        self.trace_type_selector.currentTextChanged.connect(self.update_trace)

        # Time slider for TRACE with timestamp display
        time_slider_layout = QHBoxLayout()
        time_slider_label = QLabel("Time:")
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.setValue(100)  # Default to current time (100%)
        self.time_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.time_slider.setTickInterval(10)
        self.time_slider.valueChanged.connect(self.update_trace)
        self.time_display_label = QLabel("Current")
        self.time_display_label.setMaximumWidth(80)
        time_slider_layout.addWidget(time_slider_label)
        time_slider_layout.addWidget(self.time_slider)
        time_slider_layout.addWidget(self.time_display_label)
        time_slider_widget = QWidget()
        time_slider_widget.setLayout(time_slider_layout)

        # 0DTE toggle for TRACE
        self.zero_dte_checkbox = QCheckBox("0DTE Only")
        self.zero_dte_checkbox.setChecked(False)
        self.zero_dte_checkbox.stateChanged.connect(self.update_trace)

        # Strike plot type selector
        strike_plot_label = QLabel("Strike Plot:")
        self.strike_plot_selector = QComboBox()
        self.strike_plot_selector.addItems(["GEX", "OI", "Net OI", "Combined"])
        self.strike_plot_selector.setMaximumWidth(80)
        self.strike_plot_selector.currentTextChanged.connect(self.update_trace)

        # Bucket size selector for TRACE
        bucket_size_label = QLabel("Bucket:")
        self.bucket_size_selector = QComboBox()
        self.bucket_size_selector.addItems(["1m", "5m", "10m", "15m", "30m", "1h"])
        self.bucket_size_selector.setCurrentText("10m")  # Default to 10-minute buckets
        self.bucket_size_selector.setMaximumWidth(70)
        self.bucket_size_selector.currentTextChanged.connect(self.update_trace)

        # Timeframe selector for TRACE candlesticks
        trace_timeframe_label = QLabel("Timeframe:")
        self.trace_timeframe_selector = QComboBox()
        self.trace_timeframe_selector.addItems(["1d", "1h", "30m", "15m", "5m", "1m"])
        self.trace_timeframe_selector.setCurrentText("15m")  # Default to 15-minute candles
        self.trace_timeframe_selector.setMaximumWidth(70)
        self.trace_timeframe_selector.currentTextChanged.connect(self.update_trace)

        # Zoom checkbox for TRACE
        self.trace_zoom_checkbox = QCheckBox("Enable Zoom")
        self.trace_zoom_checkbox.setChecked(False)
        self.trace_zoom_checkbox.stateChanged.connect(self.toggle_trace_zoom)

        # Focus strike input for TRACE with tooltip
        focus_strike_label = QLabel("Focus Strike:")
        self.focus_strike_input = QLineEdit()
        self.focus_strike_input.setPlaceholderText("Enter strike...")
        self.focus_strike_input.setMaximumWidth(80)
        self.focus_strike_input.returnPressed.connect(self.update_trace)
        self.focus_strike_input.setToolTip("Enter a strike price to visualize options flow activity at that strike.\nFlow dots will appear on the chart showing buying/selling activity.")

        # Add a small help button next to the focus strike input
        self.focus_strike_help = QPushButton("?")
        self.focus_strike_help.setFixedSize(20, 20)
        self.focus_strike_help.setStyleSheet("background-color: #555555; color: white; border-radius: 10px;")
        self.focus_strike_help.clicked.connect(self.show_focus_strike_help)

        # Candlestick type selector for TRACE
        candlestick_type_label = QLabel("Candle Type:")
        self.trace_candlestick_type_selector = QComboBox()
        self.trace_candlestick_type_selector.addItems(["Regular", "Hollow"])
        self.trace_candlestick_type_selector.setMaximumWidth(80)
        self.trace_candlestick_type_selector.currentTextChanged.connect(self.update_trace)

        # Create a horizontal layout for focus strike input and help button
        focus_strike_layout = QHBoxLayout()
        focus_strike_layout.setSpacing(2)
        focus_strike_layout.addWidget(focus_strike_label)
        focus_strike_layout.addWidget(self.focus_strike_input)
        focus_strike_layout.addWidget(self.focus_strike_help)
        focus_strike_widget = QWidget()
        focus_strike_widget.setLayout(focus_strike_layout)

        # Add widgets to layout
        trace_layout.addWidget(trace_type_label)
        trace_layout.addWidget(self.trace_type_selector)
        trace_layout.addWidget(time_slider_widget)
        trace_layout.addWidget(self.zero_dte_checkbox)
        trace_layout.addWidget(strike_plot_label)
        trace_layout.addWidget(self.strike_plot_selector)
        trace_layout.addWidget(bucket_size_label)
        trace_layout.addWidget(self.bucket_size_selector)
        trace_layout.addWidget(trace_timeframe_label)
        trace_layout.addWidget(self.trace_timeframe_selector)
        trace_layout.addWidget(self.trace_zoom_checkbox)
        trace_layout.addWidget(focus_strike_widget)  # Add the combined widget instead
        trace_layout.addWidget(candlestick_type_label)
        trace_layout.addWidget(self.trace_candlestick_type_selector)

        self.trace_group.setLayout(trace_layout)
        self.trace_group.hide()  # Hide initially

        # HIRO tab removed

        # Note: We're now using separate layouts for exposure tabs and other controls

        # Create the matplotlib canvas for the chart
        self.canvas = MplCanvas(self, width=12, height=10)

        # Create a separate 3D canvas for the gamma landscape
        self.canvas_3d = MplCanvas(self, width=12, height=10, is_3d=True)
        self.canvas_3d.hide()  # Hide initially

        # Mouse tracking removed

        # Crosshair functionality removed

        # Create a more compact layout for exposure tabs
        exposure_layout = QVBoxLayout()
        exposure_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        exposure_layout.setSpacing(0)  # Remove spacing
        exposure_layout.addWidget(self.exposure_tabs)

        # Create a more compact layout for other chart controls
        other_controls = QHBoxLayout()
        other_controls.setContentsMargins(0, 0, 0, 0)  # Remove margins
        other_controls.setSpacing(3)  # Reduce spacing further

        # Create backtesting controls
        self.backtesting_group = QGroupBox("Backtesting")
        backtesting_layout = QHBoxLayout()

        # Backtesting slider label
        backtesting_label = QLabel("Price Adjustment:")

        # Backtesting slider
        self.backtesting_slider = QSlider(Qt.Orientation.Horizontal)
        self.backtesting_slider.setMinimum(-50)  # -50% price
        self.backtesting_slider.setMaximum(50)   # +50% price
        self.backtesting_slider.setValue(0)      # Default to current price (0% change)
        self.backtesting_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.backtesting_slider.setTickInterval(10)
        self.backtesting_slider.valueChanged.connect(self.update_backtesting_value)

        # Value display label
        self.backtesting_value_label = QLabel("0%")

        # Add widgets to layout
        backtesting_layout.addWidget(backtesting_label)
        backtesting_layout.addWidget(self.backtesting_slider)
        backtesting_layout.addWidget(self.backtesting_value_label)

        self.backtesting_group.setLayout(backtesting_layout)

        # Create options chain controls group
        self.options_chain_group = QGroupBox("Options Chain Controls")
        options_chain_layout = QHBoxLayout()
        self.options_chain_group.setLayout(options_chain_layout)
        self.options_chain_group.hide()  # Hide initially

        # Make all control groups more compact and visually consistent
        control_groups = [chart_type_group, display_options_group, strike_range_group,
                         self.backtesting_group, self.gamma_landscape_group, self.greek_landscape_group,
                         self.iv_surface_group, self.activity_map_group, self.candlestick_group,
                         self.trace_group, self.options_chain_group]

        for group in control_groups:
            # Apply consistent styling to all control groups
            group.setStyleSheet("""
                QGroupBox {
                    font-size: 9px;
                    padding-top: 8px;
                    margin-top: 0px;
                    border: 1px solid #444;
                    border-radius: 3px;
                    background-color: #2a2a2a;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 3px;
                    background-color: #2a2a2a;
                }
            """)

            group_layout = group.layout()
            if group_layout:
                group_layout.setContentsMargins(4, 12, 4, 4)  # Reduce margins further
                group_layout.setSpacing(2)  # Minimal spacing

        # Add the control groups to the layout
        other_controls.addWidget(chart_type_group)
        other_controls.addWidget(display_options_group)
        other_controls.addWidget(strike_range_group)
        other_controls.addWidget(self.backtesting_group)
        other_controls.addWidget(self.gamma_landscape_group)
        other_controls.addWidget(self.greek_landscape_group)
        other_controls.addWidget(self.iv_surface_group)
        other_controls.addWidget(self.activity_map_group)
        other_controls.addWidget(self.candlestick_group)
        other_controls.addWidget(self.trace_group)
        other_controls.addWidget(self.options_chain_group)
        other_controls.addStretch()

        # Add all components to main layout with minimal spacing
        main_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        main_layout.setSpacing(2)  # Minimal spacing

        # Add controls with minimal vertical space
        main_layout.addLayout(top_controls)
        main_layout.addLayout(exposure_layout)
        main_layout.addLayout(other_controls)

        # Create a container widget for the canvases
        self.canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addWidget(self.canvas_3d)

        # Create a container for the options chain table
        self.options_chain_container = QWidget()
        options_chain_layout = QVBoxLayout(self.options_chain_container)
        options_chain_layout.setContentsMargins(0, 0, 0, 0)
        # The options_chain_table will be added in setup_options_chain_tab

        # Hide options chain container initially
        self.options_chain_container.hide()

        # Give the chart more space by setting a stretch factor
        main_layout.addWidget(self.canvas_container, 1)  # Stretch factor of 1 makes it take available space
        main_layout.addWidget(self.options_chain_container, 1)  # Add options chain container

        # Now that containers are created, set up the options chain tab
        self.setup_options_chain_tab(options_chain_tab)

        # Initialize data
        self.calls_df = pd.DataFrame()
        self.puts_df = pd.DataFrame()
        self.current_price = None
        self.historical_data = pd.DataFrame()
        self.volatility_trigger = None

        # Fetch available expiry dates for the default ticker
        self.fetch_expiry_dates()

    def fetch_expiry_dates(self):
        """Fetch available expiration dates for the current ticker"""
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            return

        try:
            # Format ticker for indices
            ticker = self.format_ticker(ticker)

            # Get options data
            stock = yf.Ticker(ticker)
            available_dates = stock.options

            if not available_dates:
                QMessageBox.warning(self, "Warning", "No options data available for this ticker.")
                return

            # Update expiry selector
            self.expiry_selector.clear()
            self.expiry_selector.addItems(available_dates)

            # Select the first date and fetch data
            if available_dates:
                self.refresh_data()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error fetching expiration dates: {str(e)}")

    def format_ticker(self, ticker):
        """Helper function to format tickers for indices"""
        ticker = ticker.upper()
        if ticker == "SPX":
            return "^SPX"
        elif ticker == "NDX":
            return "^NDX"
        elif ticker == "VIX":
            return "^VIX"
        elif ticker == "DJI":
            return "^DJI"
        elif ticker == "RUT":
            return "^RUT"
        return ticker

    def calculate_volatility_trigger(self, calls, puts, S):
        """Calculate the SpotGamma Volatility Trigger level

        The Volatility Trigger is the level below which market makers' gamma position
        flips from positive to negative, potentially leading to increased volatility.
        """
        try:
            # Create a copy of the dataframes to avoid modifying the originals
            calls_filtered = calls[['strike', 'GEX']].copy()
            puts_filtered = puts[['strike', 'GEX']].copy()

            # Calculate net gamma exposure by strike
            if not calls_filtered.empty:
                call_exposure = calls_filtered.groupby('strike')['GEX'].sum()
            else:
                call_exposure = pd.Series(dtype='float64')

            if not puts_filtered.empty:
                put_exposure = puts_filtered.groupby('strike')['GEX'].sum()
            else:
                put_exposure = pd.Series(dtype='float64')

            # Combine call and put exposures
            net_exposure = pd.Series(0, index=sorted(set(call_exposure.index) | set(put_exposure.index)))
            net_exposure = net_exposure.add(call_exposure, fill_value=0)
            net_exposure = net_exposure.add(put_exposure, fill_value=0)

            # Sort strikes in ascending order
            strikes = sorted(net_exposure.index)

            # Find where net gamma exposure crosses from positive to negative below current price
            # This is the volatility trigger level
            volatility_trigger = None

            # First, find all gamma flip points (where gamma crosses zero)
            gamma_flip_points = []
            for i in range(len(strikes) - 1):
                current_strike = strikes[i]
                next_strike = strikes[i + 1]
                current_value = net_exposure[current_strike]
                next_value = net_exposure[next_strike]

                # Check if there's a sign change (crossing zero)
                if (current_value * next_value <= 0) and (current_value != 0 or next_value != 0):
                    # Linear interpolation to find the exact zero-crossing point
                    if current_value == next_value or abs(next_value - current_value) < 1e-10:  # Avoid division by zero
                        flip_point = (current_strike + next_strike) / 2
                    else:
                        try:
                            # Calculate the zero-crossing point using linear interpolation
                            t = -current_value / (next_value - current_value)
                            flip_point = current_strike + t * (next_strike - current_strike)
                        except ZeroDivisionError:
                            # Fallback if division by zero occurs
                            flip_point = (current_strike + next_strike) / 2

                    gamma_flip_points.append(flip_point)

            # Find the volatility trigger - the highest gamma flip point below current price
            # where gamma transitions from positive to negative
            below_price_flips = [point for point in gamma_flip_points if point < S]

            if below_price_flips:
                # The volatility trigger is the highest flip point below current price
                volatility_trigger = max(below_price_flips)

                # Store the volatility trigger level as an instance variable
                self.volatility_trigger = volatility_trigger
                print(f"Volatility Trigger calculated: {volatility_trigger:.2f}")
            else:
                # If no flip points below current price, set to None
                self.volatility_trigger = None
                print("No Volatility Trigger found below current price")

        except Exception as e:
            print(f"Error calculating volatility trigger: {e}")
            self.volatility_trigger = None

    def update_strike_range(self):
        """Update strike range from input field"""
        try:
            new_range = float(self.strike_range_input.text())
            if new_range > 0:
                self.strike_range = new_range
                self.update_chart()
        except ValueError:
            pass  # Ignore invalid input

    def update_backtesting_value(self):
        """Update backtesting value label and trigger chart update"""
        value = self.backtesting_slider.value()
        self.backtesting_value_label.setText(f"{value}%")
        self.update_chart()

    def get_backtested_price(self):
        """Calculate the backtested price based on slider value"""
        if self.current_price is None:
            return None

        adjustment = self.backtesting_slider.value() / 100.0  # Convert percentage to decimal
        backtested_price = self.current_price * (1 + adjustment)
        return backtested_price

    def reorder_tabs(self):
        """Reorder tabs for better organization and workflow"""
        # This method doesn't actually reorder the tabs since they're already in a logical order
        # But we could implement tab reordering here if needed in the future
        pass

    def add_tab_group_labels(self):
        """Add labels above tab groups to better organize the interface"""
        # Create a custom QLabel for each tab group that will be displayed in the tab bar
        # We'll use the tab bar's style to add visual indicators for each group

        # Add group tooltips to the tabs
        # Group 1: Basic Metrics (tabs 0-3)
        for i in range(4):
            current_tooltip = self.exposure_tabs.tabToolTip(i)
            self.exposure_tabs.setTabToolTip(i, f"[Basic Metrics] {current_tooltip}")

        # Group 2: Advanced Metrics (tabs 4-7)
        for i in range(4, 8):
            current_tooltip = self.exposure_tabs.tabToolTip(i)
            self.exposure_tabs.setTabToolTip(i, f"[Advanced Metrics] {current_tooltip}")

        # Group 3: Profiles (tabs 8-10)
        for i in range(8, 11):
            current_tooltip = self.exposure_tabs.tabToolTip(i)
            self.exposure_tabs.setTabToolTip(i, f"[Profiles] {current_tooltip}")

        # Group 4: 3D Visualizations (tabs 11-13)
        for i in range(11, 14):
            current_tooltip = self.exposure_tabs.tabToolTip(i)
            self.exposure_tabs.setTabToolTip(i, f"[3D Visualizations] {current_tooltip}")

        # Group 5: Market Data (tabs 14-18)
        for i in range(14, 19):
            current_tooltip = self.exposure_tabs.tabToolTip(i)
            self.exposure_tabs.setTabToolTip(i, f"[Market Data] {current_tooltip}")

        # Group 6: TRACE (tab 19)
        current_tooltip = self.exposure_tabs.tabToolTip(19)
        self.exposure_tabs.setTabToolTip(19, f"[Time Series] {current_tooltip}")

        # Group 7: Options Chain (tab 20)
        current_tooltip = self.exposure_tabs.tabToolTip(20)
        self.exposure_tabs.setTabToolTip(20, f"[Data] {current_tooltip}")

    def setup_options_chain_tab(self, tab):
        """Setup the options chain tab with table view and controls"""
        # Create a simple layout for the tab just to indicate it's selected
        tab_layout = QVBoxLayout(tab)
        tab_label = QLabel("Options Chain View")
        tab_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tab_layout.addWidget(tab_label)

        # Get the layout of the options chain container
        options_chain_layout = self.options_chain_container.layout()

        # Create controls for the options chain
        controls_layout = QVBoxLayout()  # Changed to vertical layout for better organization

        # Top row of controls
        top_controls_layout = QHBoxLayout()

        # Moneyness filter
        moneyness_group = QGroupBox("Moneyness Filter")
        moneyness_layout = QHBoxLayout()
        self.show_itm_checkbox = QCheckBox("ITM")
        self.show_itm_checkbox.setChecked(True)
        self.show_itm_checkbox.stateChanged.connect(self.update_options_chain)

        self.show_atm_checkbox = QCheckBox("ATM")
        self.show_atm_checkbox.setChecked(True)
        self.show_atm_checkbox.stateChanged.connect(self.update_options_chain)

        self.show_otm_checkbox = QCheckBox("OTM")
        self.show_otm_checkbox.setChecked(True)
        self.show_otm_checkbox.stateChanged.connect(self.update_options_chain)

        moneyness_layout.addWidget(self.show_itm_checkbox)
        moneyness_layout.addWidget(self.show_atm_checkbox)
        moneyness_layout.addWidget(self.show_otm_checkbox)
        moneyness_group.setLayout(moneyness_layout)

        # Column selector
        columns_group = QGroupBox("Columns")
        columns_layout = QHBoxLayout()

        self.show_greeks_checkbox = QCheckBox("Greeks")
        self.show_greeks_checkbox.setChecked(True)
        self.show_greeks_checkbox.stateChanged.connect(self.update_options_chain)

        self.show_prices_checkbox = QCheckBox("Prices")
        self.show_prices_checkbox.setChecked(True)
        self.show_prices_checkbox.stateChanged.connect(self.update_options_chain)

        self.show_volume_oi_checkbox = QCheckBox("Volume/OI")
        self.show_volume_oi_checkbox.setChecked(True)
        self.show_volume_oi_checkbox.stateChanged.connect(self.update_options_chain)

        self.show_symbols_checkbox = QCheckBox("Symbols")
        self.show_symbols_checkbox.setChecked(True)
        self.show_symbols_checkbox.stateChanged.connect(self.update_options_chain)

        columns_layout.addWidget(self.show_greeks_checkbox)
        columns_layout.addWidget(self.show_prices_checkbox)
        columns_layout.addWidget(self.show_volume_oi_checkbox)
        columns_layout.addWidget(self.show_symbols_checkbox)
        columns_group.setLayout(columns_layout)

        # Add to top controls layout
        top_controls_layout.addWidget(moneyness_group)
        top_controls_layout.addWidget(columns_group)
        top_controls_layout.addStretch()

        # Bottom row of controls for sorting and filtering
        bottom_controls_layout = QHBoxLayout()

        # Strike filter
        strike_filter_group = QGroupBox("Strike Filter")
        strike_filter_layout = QHBoxLayout()

        strike_filter_label = QLabel("Strike:")
        self.strike_filter_input = QLineEdit()
        self.strike_filter_input.setPlaceholderText("Filter by strike...")
        self.strike_filter_input.setMaximumWidth(120)
        self.strike_filter_input.textChanged.connect(self.filter_options_chain)

        strike_filter_layout.addWidget(strike_filter_label)
        strike_filter_layout.addWidget(self.strike_filter_input)
        strike_filter_group.setLayout(strike_filter_layout)

        # Sort controls
        sort_group = QGroupBox("Sort Options")
        sort_layout = QHBoxLayout()

        sort_column_label = QLabel("Sort by:")
        self.sort_column_selector = QComboBox()
        self.sort_column_selector.addItems(["Symbol", "Strike", "Option Type", "Last Price", "Volume", "Open Interest", "Implied Volatility", "Delta", "Gamma", "Theta", "Vega"])
        self.sort_column_selector.setCurrentText("Strike")
        self.sort_column_selector.currentTextChanged.connect(self.sort_options_chain)

        self.sort_direction_button = QPushButton("↑")
        self.sort_direction_button.setMaximumWidth(30)
        self.sort_direction_button.setCheckable(True)
        self.sort_direction_button.setChecked(False)  # Default to ascending
        self.sort_direction_button.clicked.connect(self.toggle_sort_direction)

        sort_layout.addWidget(sort_column_label)
        sort_layout.addWidget(self.sort_column_selector)
        sort_layout.addWidget(self.sort_direction_button)
        sort_group.setLayout(sort_layout)

        # Add to bottom controls layout
        bottom_controls_layout.addWidget(strike_filter_group)
        bottom_controls_layout.addWidget(sort_group)
        bottom_controls_layout.addStretch()

        # Add both rows to main controls layout
        controls_layout.addLayout(top_controls_layout)
        controls_layout.addLayout(bottom_controls_layout)

        # T-chart view has been removed - using regular view only

        # Create table view for options chain with TOS-like styling
        self.options_chain_table = QTableView()
        self.options_chain_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.options_chain_table.setAlternatingRowColors(True)
        self.options_chain_table.setSortingEnabled(True)
        self.options_chain_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.options_chain_table.horizontalHeader().setStretchLastSection(True)

        # TOS-like styling for the table
        self.options_chain_table.setStyleSheet("""
            QTableView {
                background-color: #1C1C1C;
                alternate-background-color: #252525;
                color: white;
                gridline-color: #444444;
                border: 1px solid #444444;
                selection-background-color: #3A3A3A;
                selection-color: white;
            }
            QHeaderView::section {
                background-color: #333333;
                color: white;
                padding: 4px;
                border: 1px solid #444444;
                font-weight: bold;
            }
            QHeaderView::section:checked {
                background-color: #444444;
            }
        """)

        # Set alignment for the table cells (center alignment for strike column)
        self.options_chain_table.setItemDelegate(self.create_center_delegate())

        # Create model for options chain (T-chart model removed)
        self.options_chain_model = OptionsChainModel()  # Regular model only

        # Create proxy model for filtering/sorting
        self.options_chain_proxy_model = QSortFilterProxyModel()

        # Set the model
        self.options_chain_proxy_model.setSourceModel(self.options_chain_model)
        self.options_chain_table.setModel(self.options_chain_proxy_model)

        # Create a horizontal layout to center the table
        center_layout = QHBoxLayout()
        center_layout.addStretch(1)  # Add stretch on the left
        center_layout.addWidget(self.options_chain_table, 4)  # Table takes 4 parts of space
        center_layout.addStretch(1)  # Add stretch on the right

        # Add widgets to the options chain container layout
        options_chain_layout.addLayout(controls_layout)
        options_chain_layout.addLayout(center_layout)

    def update_options_chain(self):
        """Update the options chain table with current data"""
        if self.calls_df.empty and self.puts_df.empty:
            return

        # Get current price for determining moneyness
        current_price = self.current_price

        # Create copies of the dataframes
        calls = self.calls_df.copy()
        puts = self.puts_df.copy()

        # Add option type column
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'

        # Determine moneyness (ITM, ATM, OTM)
        # For calls: ITM if strike < price, OTM if strike > price
        # For puts: ITM if strike > price, OTM if strike < price
        # ATM if within 1% of current price
        atm_threshold = current_price * 0.01

        # Define a function to handle NaN values in strike
        def determine_call_moneyness(strike):
            if pd.isna(strike):
                return 'Unknown'
            return 'ATM' if abs(strike - current_price) <= atm_threshold else \
                   ('ITM' if strike < current_price else 'OTM')

        def determine_put_moneyness(strike):
            if pd.isna(strike):
                return 'Unknown'
            return 'ATM' if abs(strike - current_price) <= atm_threshold else \
                   ('ITM' if strike > current_price else 'OTM')

        calls['moneyness'] = calls['strike'].apply(determine_call_moneyness)
        puts['moneyness'] = puts['strike'].apply(determine_put_moneyness)

        # Filter based on moneyness checkboxes
        moneyness_filters = ['Unknown']  # Always include Unknown category
        if self.show_itm_checkbox.isChecked():
            moneyness_filters.append('ITM')
        if self.show_atm_checkbox.isChecked():
            moneyness_filters.append('ATM')
        if self.show_otm_checkbox.isChecked():
            moneyness_filters.append('OTM')

        calls = calls[calls['moneyness'].isin(moneyness_filters)]
        puts = puts[puts['moneyness'].isin(moneyness_filters)]

        # Combine the dataframes
        combined_df = pd.concat([calls, puts])

        # Select columns based on checkboxes in TOS-like order
        columns = ['option_type', 'moneyness', 'strike']

        # TOS-like column ordering
        price_columns = []
        greek_columns = []
        volume_columns = []
        symbol_column = []

        # Check if contractSymbol is available in the dataframe
        if 'contractSymbol' in combined_df.columns and self.show_symbols_checkbox.isChecked():
            symbol_column = ['contractSymbol']

        if self.show_prices_checkbox.isChecked():
            # TOS typically shows: Last, Bid, Ask, IV
            price_columns = ['lastPrice', 'bid', 'ask', 'impliedVolatility']

        if self.show_greeks_checkbox.isChecked():
            # TOS typically shows: Delta, Gamma, Theta, Vega in that order
            greek_columns = ['calc_delta', 'calc_gamma', 'calc_theta', 'calc_vega']
            # Add advanced Greeks if available
            if 'calc_vanna' in combined_df.columns:
                greek_columns.append('calc_vanna')
            if 'calc_vomma' in combined_df.columns:
                greek_columns.append('calc_vomma')

        if self.show_volume_oi_checkbox.isChecked():
            # TOS typically shows volume before open interest
            volume_columns = ['volume', 'openInterest']

        # TOS-like column order: Symbol, Strike, Last, Bid, Ask, IV, Delta, Gamma, Theta, Vega, Volume, OI
        columns = symbol_column + ['option_type', 'moneyness', 'strike'] + price_columns + greek_columns + volume_columns

        # Filter columns that exist in the dataframe
        columns = [col for col in columns if col in combined_df.columns]

        # Store the processed dataframe and columns for filtering/sorting
        self.processed_options_df = combined_df
        self.processed_options_columns = columns

        # Apply any existing filters and sorting
        self.apply_options_chain_filters()

    # toggle_options_chain_view method removed (T-chart view removed)

    def apply_options_chain_filters(self):
        """Apply filters and sorting to the options chain"""
        if not hasattr(self, 'processed_options_df') or self.processed_options_df.empty:
            return

        filtered_df = self.processed_options_df.copy()
        columns = self.processed_options_columns

        # Apply strike filter if it exists
        if hasattr(self, 'strike_filter_input') and self.strike_filter_input.text().strip():
            strike_filter = self.strike_filter_input.text().strip()
            try:
                # Try to interpret as a number for exact match
                strike_value = float(strike_filter)
                # Filter strikes that are close to the entered value (within 1%)
                tolerance = strike_value * 0.01
                filtered_df = filtered_df[(filtered_df['strike'] >= strike_value - tolerance) &
                                         (filtered_df['strike'] <= strike_value + tolerance)]
            except ValueError:
                # If not a number, use it as a string filter
                filtered_df = filtered_df[filtered_df['strike'].astype(str).str.contains(strike_filter, case=False)]

        # Apply sorting if sort column is selected
        if hasattr(self, 'sort_column_selector'):
            sort_column = self.sort_column_selector.currentText()
            sort_ascending = not (hasattr(self, 'sort_direction_button') and self.sort_direction_button.isChecked())

            # Map display names to actual column names
            column_map = {
                'Strike': 'strike',
                'Option Type': 'option_type',
                'Last Price': 'lastPrice',
                'Volume': 'volume',
                'Open Interest': 'openInterest',
                'Implied Volatility': 'impliedVolatility',
                'Delta': 'calc_delta',
                'Gamma': 'calc_gamma',
                'Theta': 'calc_theta',
                'Vega': 'calc_vega',
                'Symbol': 'contractSymbol'
            }

            # Get the actual column name
            sort_col = column_map.get(sort_column, 'strike')

            # Check if the column exists in the dataframe
            if sort_col in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_ascending)
            else:
                # Default to sorting by strike and option type
                filtered_df = filtered_df.sort_values(by=['strike', 'option_type'], ascending=[sort_ascending, True])
        else:
            # Default sorting by strike and option type
            filtered_df = filtered_df.sort_values(by=['strike', 'option_type'])

        # Update the regular model (T-chart model removed)
        self.options_chain_model.setData(filtered_df, columns)

    def filter_options_chain(self):
        """Filter the options chain based on user input"""
        self.apply_options_chain_filters()

    def sort_options_chain(self):
        """Sort the options chain based on selected column"""
        self.apply_options_chain_filters()

    def toggle_sort_direction(self):
        """Toggle the sort direction between ascending and descending"""
        if self.sort_direction_button.isChecked():
            self.sort_direction_button.setText("↓")  # Down arrow
        else:
            self.sort_direction_button.setText("↑")  # Up arrow
        self.apply_options_chain_filters()

    def create_center_delegate(self):
        """Create a delegate that centers specific columns in the options chain"""
        class CenterAlignDelegate(QStyledItemDelegate):
            def initStyleOption(self, option, index):
                super().initStyleOption(option, index)
                # Get the column name from the model
                if hasattr(self.parent(), 'options_chain_model') and hasattr(self.parent().options_chain_model, '_headers'):
                    model = self.parent().options_chain_model
                    if 0 <= index.column() < len(model._headers):
                        column_name = model._headers[index.column()]
                        # Center align strike, moneyness, and option_type columns
                        if column_name in ['strike', 'moneyness', 'option_type']:
                            option.displayAlignment = Qt.AlignmentFlag.AlignCenter
                        # Right align numeric columns
                        elif column_name in ['lastPrice', 'bid', 'ask', 'impliedVolatility',
                                           'calc_delta', 'calc_gamma', 'calc_theta', 'calc_vega',
                                           'volume', 'openInterest']:
                            option.displayAlignment = Qt.AlignmentFlag.AlignRight

        return CenterAlignDelegate(self.options_chain_table)

    def update_exposure_type_from_tab(self, index):
        """Update the exposure type based on tab selection"""
        # Show canvas container and hide options chain container if coming from options chain tab
        if self.exposure_type == "OPTIONS_CHAIN":
            self.canvas_container.show()
            self.options_chain_container.hide()

        tab_text = self.exposure_tabs.tabText(index)
        self.update_exposure_type(tab_text)

    def update_exposure_type(self, text):
        """Update the exposure type based on selection"""
        # First hide all canvases
        self.canvas.hide()
        self.canvas_3d.hide()

        # Hide PyQtGraph widget if it exists
        if hasattr(self, 'pg_widget') and self.pg_widget is not None:
            self.pg_widget.hide()

        # Reset both canvases to avoid visual artifacts
        # Reset 3D canvas
        try:
            # Clear the figure
            self.canvas_3d.fig.clear()

            # Reset the colorbar reference
            if hasattr(self.canvas_3d, 'colorbar'):
                self.canvas_3d.colorbar = None

            # Create a new 3D axes
            self.canvas_3d.axes = self.canvas_3d.fig.add_subplot(111, projection='3d')
        except Exception as e:
            print(f"Error resetting 3D canvas during tab switch: {e}")

        # Reset 2D canvas
        try:
            # Clear the figure
            self.canvas.fig.clear()

            # Reset the colorbar reference
            if hasattr(self.canvas, 'colorbar'):
                self.canvas.colorbar = None

            # Create a new axes
            self.canvas.axes = self.canvas.fig.add_subplot(111)
        except Exception as e:
            print(f"Error resetting 2D canvas during tab switch: {e}")

        # Hide/show landscape controls
        if "Gamma Landscape" in text:
            self.gamma_landscape_group.show()
            self.greek_landscape_group.hide()
            self.iv_surface_group.hide()
            self.activity_map_group.hide()
            self.exposure_type = "GAMMA_LANDSCAPE"
            self.canvas_3d.exposure_type = "GAMMA_LANDSCAPE"

            # Set the 3D figure layout
            self.canvas_3d.fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)

            self.canvas_3d.show()  # Show 3D canvas
            self.setWindowTitle("Options Exposure Dashboard - Gamma Landscape")
            self.update_gamma_landscape()
            return  # Skip regular chart update
        elif "Greek Landscape" in text:
            self.greek_landscape_group.show()
            self.gamma_landscape_group.hide()
            self.iv_surface_group.hide()
            self.activity_map_group.hide()
            self.exposure_type = "GREEK_LANDSCAPE"
            self.canvas_3d.exposure_type = "GREEK_LANDSCAPE"

            # Set the 3D figure layout
            self.canvas_3d.fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)

            self.canvas_3d.show()  # Show 3D canvas
            self.setWindowTitle("Options Exposure Dashboard - Greek Landscape")
            self.update_greek_landscape()
            return  # Skip regular chart update
        elif "IV Surface" in text:
            self.iv_surface_group.show()
            self.gamma_landscape_group.hide()
            self.greek_landscape_group.hide()
            self.activity_map_group.hide()
            self.exposure_type = "IV_SURFACE"
            self.canvas_3d.exposure_type = "IV_SURFACE"

            # Set the 3D figure layout
            self.canvas_3d.fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)

            self.canvas_3d.show()  # Show 3D canvas
            self.setWindowTitle("Options Exposure Dashboard - IV Surface")
            self.update_iv_surface()
            return  # Skip regular chart update
        elif "Activity Map" in text:
            self.activity_map_group.show()
            self.gamma_landscape_group.hide()
            self.greek_landscape_group.hide()
            self.iv_surface_group.hide()
            self.candlestick_group.hide()
            # Show regular canvas
            self.canvas.show()
            self.canvas_3d.hide()
        elif "Candlestick Chart" in text:
            self.candlestick_group.show()
            self.gamma_landscape_group.hide()
            self.greek_landscape_group.hide()
            self.iv_surface_group.hide()
            self.activity_map_group.hide()
            # Hide both matplotlib canvases
            self.canvas.hide()
            self.canvas_3d.hide()
            self.exposure_type = "CANDLESTICK"
            self.canvas.exposure_type = "CANDLESTICK"
            self.setWindowTitle("Options Exposure Dashboard - Candlestick Chart")
            self.update_candlestick_chart()
            return  # Skip regular chart update
        else:
            self.gamma_landscape_group.hide()
            self.greek_landscape_group.hide()
            self.iv_surface_group.hide()
            self.activity_map_group.hide()
            self.candlestick_group.hide()
            # Show regular canvas for all other chart types
            self.canvas.show()
            self.canvas_3d.hide()

        if "Vanna" in text:
            self.exposure_type = "VEX"
            self.canvas.exposure_type = "VEX"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Vanna (VEX)")
        elif "Gamma" in text and "Landscape" not in text:
            self.exposure_type = "GEX"
            self.canvas.exposure_type = "GEX"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Gamma (GEX)")
        elif "Delta Profile" in text:
            self.exposure_type = "DELTA_PROFILE"
            self.canvas.exposure_type = "DELTA_PROFILE"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Delta Profile")
        elif "Delta" in text:
            self.exposure_type = "DEX"
            self.canvas.exposure_type = "DEX"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Delta (DEX)")
        elif "Charm" in text:
            self.exposure_type = "CEX"
            self.canvas.exposure_type = "CEX"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Charm (CEX)")
        elif "Theta" in text:
            self.exposure_type = "TEX"
            self.canvas.exposure_type = "TEX"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Theta (TEX)")
        elif "Vega Profile" in text:
            self.exposure_type = "VEGA_PROFILE"
            self.canvas.exposure_type = "VEGA_PROFILE"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Vega Profile")
        elif "Vomma Profile" in text:
            self.exposure_type = "VOMMA_PROFILE"
            self.canvas.exposure_type = "VOMMA_PROFILE"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Vomma Profile")
        elif "Vomma" in text:
            self.exposure_type = "VOMX"
            self.canvas.exposure_type = "VOMX"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Vomma (VOMX)")
        elif "Vega" in text and "Profile" not in text:
            self.exposure_type = "VEGX"
            self.canvas.exposure_type = "VEGX"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Vega (VEGX)")
        elif "Max Pain" in text:
            self.exposure_type = "MPX"
            self.canvas.exposure_type = "MPX"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Max Pain (MPX)")
        elif "Open Interest" in text:
            self.exposure_type = "OI"
            self.canvas.exposure_type = "OI"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Open Interest (OI)")
        elif "Put/Call Ratio" in text:
            self.exposure_type = "PCR"
            self.canvas.exposure_type = "PCR"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Put/Call Ratio (PCR)")
        elif "IV Skew" in text:
            self.exposure_type = "IVSKEW"
            self.canvas.exposure_type = "IVSKEW"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - IV Skew")
        elif "Activity Map" in text:
            self.exposure_type = "ACTIVITY_MAP"
            self.canvas.exposure_type = "ACTIVITY_MAP"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Options Chain Activity Map")
        elif "Candlestick Chart" in text:
            self.exposure_type = "CANDLESTICK"
            self.canvas.exposure_type = "CANDLESTICK"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - Candlestick Chart")
            return  # Skip regular chart update as we handle it in the condition above
        elif "TRACE" in text:
            self.trace_group.show()
            self.gamma_landscape_group.hide()
            self.greek_landscape_group.hide()
            self.iv_surface_group.hide()
            self.activity_map_group.hide()
            self.candlestick_group.hide()
            # Show regular canvas
            self.canvas.show()
            self.canvas_3d.hide()
            self.exposure_type = "TRACE"
            self.canvas.exposure_type = "TRACE"  # Update canvas exposure type
            self.setWindowTitle("Options Exposure Dashboard - TRACE")
            # Ensure 15-minute timeframe is selected
            self.trace_timeframe_selector.setCurrentText("15m")
            # Call update_trace directly instead of going through update_chart
            self.update_trace()
            return  # Skip regular chart update
        # HIRO tab removed
        elif "Options Chain" in text:
            # Hide all control groups
            self.gamma_landscape_group.hide()
            self.greek_landscape_group.hide()
            self.iv_surface_group.hide()
            self.activity_map_group.hide()
            self.candlestick_group.hide()
            self.trace_group.hide()
            # Hide both matplotlib canvases
            self.canvas.hide()
            self.canvas_3d.hide()
            # Hide the canvas container and show the options chain container
            self.canvas_container.hide()
            self.options_chain_container.show()
            self.exposure_type = "OPTIONS_CHAIN"
            self.setWindowTitle("Options Exposure Dashboard - Options Chain")
            # Update the options chain
            self.update_options_chain()
            return  # Skip regular chart update
        self.update_chart()

    def refresh_data(self):
        """Fetch new data for the selected ticker and expiry date"""
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Warning", "Please enter a ticker symbol.")
            return

        expiry_date = self.expiry_selector.currentText()
        if not expiry_date:
            # If no expiry date is selected, fetch available dates first
            self.fetch_expiry_dates()
            return

        # For 3D visualizations, reset the figure when refreshing data
        if self.exposure_type in ["GAMMA_LANDSCAPE", "GREEK_LANDSCAPE", "IV_SURFACE"]:
            try:
                # Clear the figure
                self.canvas_3d.fig.clear()

                # Reset the colorbar reference
                if hasattr(self.canvas_3d, 'colorbar'):
                    self.canvas_3d.colorbar = None

                # Create a new 3D axes
                self.canvas_3d.axes = self.canvas_3d.fig.add_subplot(111, projection='3d')
            except Exception as e:
                print(f"Error resetting 3D canvas during refresh: {e}")

        # Show loading indicator
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText("Loading...")

        # Create and start the data fetch thread
        self.fetch_thread = DataFetchThread(ticker, expiry_date)
        self.fetch_thread.data_ready.connect(self.on_data_ready)
        self.fetch_thread.error.connect(self.on_fetch_error)
        self.fetch_thread.finished.connect(self.on_fetch_finished)
        self.fetch_thread.start()

    def on_data_ready(self, data):
        """Handle the fetched data"""
        try:
            self.calls_df, self.puts_df, self.current_price, self.max_pain_strike = data

            # Check for invalid data that might cause division by zero
            if self.current_price is None or self.current_price <= 0:
                QMessageBox.warning(self, "Warning", "Invalid or missing current price data.")
                return

            # Calculate volatility trigger level
            self.calculate_volatility_trigger(self.calls_df, self.puts_df, self.current_price)

            if self.exposure_type == "GAMMA_LANDSCAPE":
                # We've already reset the figure in refresh_data, just update the landscape
                self.update_gamma_landscape()
            elif self.exposure_type == "GREEK_LANDSCAPE":
                # Update the Greek landscape
                self.update_greek_landscape()
            elif self.exposure_type == "IV_SURFACE":
                # Update the IV surface
                self.update_iv_surface()
            elif self.exposure_type == "CANDLESTICK":
                # Update the candlestick chart
                self.update_candlestick_chart()
            elif self.exposure_type == "TRACE":
                # Update the TRACE visualization
                self.update_trace()
            # HIRO tab removed
            elif self.exposure_type == "OPTIONS_CHAIN":
                # Update the options chain
                self.update_options_chain()
            else:
                self.update_chart()
        except ZeroDivisionError as e:
            print(f"Division by zero error in on_data_ready: {e}")
            QMessageBox.critical(self, "Error", "Error fetching data: division by zero. This usually happens with invalid or missing data.")
        except Exception as e:
            print(f"Error in on_data_ready: {e}")
            QMessageBox.critical(self, "Error", f"Error processing data: {str(e)}")

    def on_fetch_error(self, error_msg):
        """Handle data fetch errors"""
        QMessageBox.critical(self, "Error", error_msg)

    def on_fetch_finished(self):
        """Reset the UI after fetch is complete"""
        self.refresh_button.setEnabled(True)
        self.refresh_button.setText("Refresh")

    # Crosshair functionality removed

    def update_gamma_landscape_settings(self):
        """Update gamma landscape when settings change"""
        if self.exposure_type == "GAMMA_LANDSCAPE":
            self.update_gamma_landscape()

    def update_greek_landscape_settings(self):
        """Update Greek landscape when settings change"""
        if self.exposure_type == "GREEK_LANDSCAPE":
            self.update_greek_landscape()

    def update_iv_surface_settings(self):
        """Update IV surface when settings change"""
        if self.exposure_type == "IV_SURFACE":
            self.update_iv_surface()

    # HIRO tab and related methods removed

    def update_trace(self):
        """Create a visualization of options market pressure (TRACE)"""
        if self.calls_df.empty and self.puts_df.empty:
            return

        # Fetch historical data for candlesticks
        # Get the selected timeframe from the selector
        selected_timeframe = self.trace_timeframe_selector.currentText()

        # Try to fetch data with the selected timeframe
        if not self.fetch_historical_data(interval=selected_timeframe):
            # If that fails, try 15-minute data first, then other timeframes
            if not self.fetch_historical_data(interval='15m'):
                if not self.fetch_historical_data(interval='5m'):
                    if not self.fetch_historical_data(interval='1m'):
                        if not self.fetch_historical_data(interval='1h'):
                            # If all else fails, try daily data
                            if not self.fetch_historical_data(interval='1d'):
                                # Continue even if we can't get historical data
                                # We'll just show a flat line at the current price
                                # Show a popup warning when historical data isn't available
                                QMessageBox.warning(self, "Warning", "No historical data available for this timeframe. Using current price only.")
                                print("No historical data available, using current price only")

        # Update time display label based on slider value
        time_percent = self.time_slider.value() / 100.0
        if time_percent == 1.0:
            self.time_display_label.setText("Current")
        else:
            # If we have historical data, use it to determine the time point
            time_label_set = False
            if not self.historical_data.empty:
                # Get the timestamp at the position indicated by the slider
                timestamps = self.historical_data.index
                if len(timestamps) > 0:
                    # Calculate index based on slider position
                    # Use safe_convert to avoid numpy.int64 issues
                    idx_value = min(safe_convert(int(len(timestamps) * time_percent)), safe_convert(len(timestamps) - 1))
                    # Convert to int to ensure it's a valid index
                    idx_value = int(idx_value)
                    time_point = timestamps[idx_value]
                    self.time_display_label.setText(time_point.strftime('%H:%M'))
                    time_label_set = True

            # Fallback to calculating time based on market hours if no historical data
            if not time_label_set:
                market_open = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

                # Calculate the time point based on the slider percentage
                if datetime.now() < market_open or datetime.now() > market_close:
                    # Outside market hours, use full range
                    time_point = market_open + (market_close - market_open) * time_percent
                else:
                    # During market hours, use range from market open to current time
                    time_point = market_open + (datetime.now() - market_open) * time_percent

                self.time_display_label.setText(time_point.strftime('%H:%M'))

        try:
            # Clear the current chart
            self.canvas.axes.clear()

            # Get the selected heatmap type
            heatmap_type = self.trace_type_selector.currentText()

            # Get the selected strike plot type
            strike_plot_type = self.strike_plot_selector.currentText()

            # Get time slider value (0-100%)
            time_percent = self.time_slider.value() / 100.0

            # Check if 0DTE only is selected
            zero_dte_only = self.zero_dte_checkbox.isChecked()

            # Get the price to center around (current or backtested)
            center_price = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

            # Calculate strike range around center price
            min_strike = center_price - self.strike_range
            max_strike = center_price + self.strike_range

            # Filter data based on strike range
            calls_filtered = self.calls_df[self.calls_df['strike'].between(min_strike, max_strike)].copy()
            puts_filtered = self.puts_df[self.puts_df['strike'].between(min_strike, max_strike)].copy()

            # If 0DTE only is selected, filter for options expiring today
            if zero_dte_only:
                today = datetime.today().date()
                selected_expiry = datetime.strptime(self.expiry_selector.currentText(), '%Y-%m-%d').date()
                days_to_expiry = (selected_expiry - today).days

                if days_to_expiry > 0:
                    # Show message that there are no 0DTE options for this date
                    self.canvas.axes.text(0.5, 0.5, "No 0DTE options for selected expiry date",
                                         ha='center', va='center', fontsize=14, color='white',
                                         transform=self.canvas.axes.transAxes)

                    # Set dark background
                    self.canvas.fig.patch.set_facecolor('#121212')
                    self.canvas.axes.set_facecolor('#1e1e1e')
                    self.canvas.draw()
                    return

            # Create a figure with 2 subplots - heatmap on top, strike plot on bottom
            self.canvas.fig.clear()
            # Create gridspec layout with more space for the heatmap
            gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.25, figure=self.canvas.fig)
            heatmap_ax = self.canvas.fig.add_subplot(gs[0])
            strike_ax = self.canvas.fig.add_subplot(gs[1])

            # Create time range for x-axis (market hours: 9:30 AM to 4:00 PM ET)
            market_open = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

            # Adjust the time range based on the time slider position
            current_time = datetime.now()

            if time_percent < 1.0:
                # When time slider is not at 100%, only include times up to the slider position
                if current_time < market_open or current_time > market_close:
                    # Outside market hours, use adjusted range based on slider
                    adjusted_close = market_open + (market_close - market_open) * time_percent
                    time_range = pd.date_range(market_open, adjusted_close, periods=100)
                else:
                    # During market hours, use range from market open to adjusted current time
                    adjusted_time = market_open + (current_time - market_open) * time_percent
                    time_range = pd.date_range(market_open, adjusted_time, periods=100)
            else:
                # When time slider is at 100%, use the full time range
                if current_time < market_open or current_time > market_close:
                    time_range = pd.date_range(market_open, market_close, periods=100)
                else:
                    # Use range from market open to current time
                    time_range = pd.date_range(market_open, current_time, periods=100)

            # Time range is already adjusted based on the slider in the code above

            # Create strike range for y-axis
            strike_range = np.linspace(min_strike, max_strike, 50)

            # Initialize ohlc_data list and valid_data_points (will be populated later)
            ohlc_data = []
            valid_data_points = []

            # We'll generate the heatmap after we've collected the valid data points
            # This ensures the heatmap is properly aligned with the candlesticks

            # Format heatmap axes with larger font sizes
            heatmap_ax.set_title(f"{heatmap_type} Heatmap", color='white', fontsize=12)

            # Set x-axis to show time labels
            # For candlesticks, we want to show the actual time values at regular intervals
            if len(ohlc_data) > 0:
                # Get the number of candlesticks
                num_candles = len(ohlc_data)
                # Create evenly spaced ticks
                time_ticks = np.linspace(0, num_candles-1, min(5, num_candles), dtype=int)
                heatmap_ax.set_xticks(time_ticks)

                # Create time labels from the actual data points
                if len(valid_data_points) > 0:
                    time_labels = [valid_data_points[i][0].strftime('%H:%M') for i in time_ticks if i < len(valid_data_points)]
                    heatmap_ax.set_xticklabels(time_labels)
            else:
                # Fallback to the original time range if no candlesticks
                time_ticks = np.linspace(0, len(time_range)-1, 5, dtype=int)
                heatmap_ax.set_xticks(time_ticks)
                time_labels = [t.strftime('%H:%M') for t in time_range[time_ticks]]
                heatmap_ax.set_xticklabels(time_labels)

            # Set y-axis to show strike labels directly using price values
            # Create evenly spaced strike ticks within the visible range
            strike_ticks = np.linspace(min_strike, max_strike, 10)
            heatmap_ax.set_yticks(strike_ticks)
            strike_labels = [f"{tick:.0f}" for tick in strike_ticks]
            heatmap_ax.set_yticklabels(strike_labels)

            # Set the axis limits to ensure the entire chart is visible
            heatmap_ax.set_ylim(min_strike, max_strike)

            # Set x-axis limits to show the appropriate number of candlesticks based on time slider
            if len(ohlc_data) > 0:
                # When time slider is not at 100%, adjust the x-axis limits to match the filtered data
                if time_percent < 1.0:
                    num_visible_points = max(2, int(len(valid_data_points) * time_percent))
                    heatmap_ax.set_xlim(-1, num_visible_points)
                else:
                    # When time slider is at 100%, show all candlesticks
                    heatmap_ax.set_xlim(-1, len(ohlc_data))

            # Add current price line to heatmap using actual price value
            heatmap_ax.axhline(self.current_price, color='white', linestyle='--', alpha=0.7,
                              label=f'Current Price: {self.current_price:.2f}')

            # Add backtested price line if different from current price
            backtested_price = self.get_backtested_price()
            if backtested_price is not None and self.backtesting_slider.value() != 0:
                heatmap_ax.axhline(backtested_price, color='yellow', linestyle='-.', alpha=0.9, linewidth=1.5,
                                  label=f'Backtested Price: {backtested_price:.2f} ({self.backtesting_slider.value():+d}%)')

            # Add volatility trigger line if enabled
            if self.show_vol_trigger_checkbox.isChecked() and hasattr(self, 'volatility_trigger') and self.volatility_trigger is not None:
                # Use a distinctive purple color for the volatility trigger
                heatmap_ax.axhline(self.volatility_trigger, color='#9370DB', linestyle='-',
                                alpha=0.9, linewidth=2.0, label=f'Vol Trigger: {self.volatility_trigger:.2f}')

                # Add text annotation at the right edge of the heatmap
                x_min, x_max = heatmap_ax.get_xlim()
                heatmap_ax.text(x_max * 0.95, self.volatility_trigger, f'Vol Trigger: {self.volatility_trigger:.2f}',
                            color='#9370DB', ha='right', va='center', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

            # Add current time line to heatmap if using current time
            # For consecutive candlesticks, we'll mark the last candlestick as current time
            if time_percent == 1.0 and current_time >= market_open and current_time <= market_close:
                if len(ohlc_data) > 0:
                    current_time_idx = len(ohlc_data) - 1
                else:
                    current_time_idx = len(time_range) - 1
                heatmap_ax.axvline(current_time_idx, color='white', linestyle=':', alpha=0.7)

            # Add candlestick chart over time
            try:
                # Get historical price data
                ticker = self.ticker_input.text().strip().upper()
                candlestick_added = False  # Initialize flag

                # Get candlestick type
                candlestick_type = self.trace_candlestick_type_selector.currentText()

                # Check if we have any historical data
                if not self.historical_data.empty and all(col in self.historical_data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    # Get the most recent data available (up to 5 days back)
                    recent_days = 5
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=recent_days)

                    # Filter for recent data using the calculated start_date
                    # This ensures we only use recent data for visualization
                    recent_data = self.historical_data[
                        (self.historical_data.index.date >= start_date) &
                        (self.historical_data.index.date <= end_date)
                    ]

                    if not recent_data.empty:
                        # Resample data to match our time range
                        price_times = recent_data.index

                        # Map price times to our time range indices
                        time_indices = []
                        ohlc_data = []

                        # Create a mapping between real times and our time range
                        # This handles cases where market hours don't match exactly
                        real_market_start = price_times.min()
                        real_market_end = price_times.max()
                        real_market_duration = (real_market_end - real_market_start).total_seconds()

                        if real_market_duration > 0:  # Ensure we have a valid duration
                            # Create a list to store valid data points
                            valid_data_points = []

                            # First collect all valid data points
                            for i, price_time in enumerate(price_times):
                                try:
                                    # Get the actual OHLC values
                                    open_price = recent_data.iloc[i]['Open']
                                    high_price = recent_data.iloc[i]['High']
                                    low_price = recent_data.iloc[i]['Low']
                                    close_price = recent_data.iloc[i]['Close']

                                    # Ensure we have valid price data
                                    if not (pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price)):
                                        valid_data_points.append((price_time, open_price, high_price, low_price, close_price))
                                except Exception as e:
                                    print(f"Error collecting OHLC data: {e}")
                                    continue

                            # Filter data points based on time slider
                            # Calculate how many data points to include based on time_percent
                            if time_percent < 1.0:
                                # Calculate and use the number of points to include
                                filtered_data_points = valid_data_points[:max(2, int(len(valid_data_points) * time_percent))]
                            else:
                                # Include all data points if slider is at 100%
                                filtered_data_points = valid_data_points

                            # Now plot the filtered data points consecutively
                            for i, (price_time, open_price, high_price, low_price, close_price) in enumerate(filtered_data_points):
                                # Use consecutive indices for time
                                # Use safe_convert to avoid numpy.int64 issues
                                time_idx = safe_convert(i)
                                try:
                                    # Ensure min_strike and max_strike are defined and valid
                                    if min_strike >= max_strike or min_strike <= 0:
                                        # Use a reasonable range around current price if strike range is invalid
                                        min_strike = self.current_price * 0.9
                                        max_strike = self.current_price * 1.1

                                    # Map OHLC prices directly to y-coordinates in the heatmap
                                    # Instead of using indices, we'll use the actual price values
                                    # This will make the candlesticks appear at the correct price levels
                                    # Use safe_convert to avoid numpy.int64 issues
                                    open_idx = safe_convert(open_price)
                                    high_idx = safe_convert(high_price)
                                    low_idx = safe_convert(low_price)
                                    close_idx = safe_convert(close_price)

                                    # Ensure prices are within the visible range
                                    # Use safe_convert to avoid numpy.int64 issues
                                    open_idx = safe_convert(max(min_strike, min(open_idx, max_strike)))
                                    high_idx = safe_convert(max(min_strike, min(high_idx, max_strike)))
                                    low_idx = safe_convert(max(min_strike, min(low_idx, max_strike)))
                                    close_idx = safe_convert(max(min_strike, min(close_idx, max_strike)))

                                    # Add to our data collections
                                    time_indices.append(time_idx)
                                    ohlc_data.append((time_idx, open_idx, high_idx, low_idx, close_idx))
                                except Exception as e:
                                    print(f"Error mapping OHLC data: {e}")
                                    continue  # Skip this data point if there's an error

                        # Plot candlesticks if we have enough points
                        if len(ohlc_data) > 1:
                            # Sort by time index to ensure candlesticks are drawn correctly
                            ohlc_data.sort(key=lambda x: x[0])

                            # Define ThinkorSwim-like colors for up and down candles
                            up_color = '#00CC00'  # Softer green like TOS
                            down_color = '#FF3333'  # Softer red like TOS

                            # Add a label for the legend
                            heatmap_ax.plot([], [], color='white', linewidth=2, label='Price (Candlesticks)')

                            # Draw candlesticks
                            width = 0.3  # Width of candlestick body (reduced for better appearance, more like TOS)
                            for t, open_idx, high_idx, low_idx, close_idx in ohlc_data:
                                try:
                                    # Determine if it's an up or down candle
                                    is_up = close_idx >= open_idx
                                    color = up_color if is_up else down_color

                                    # Make sure there's at least 1 pixel height for the body
                                    # If open and close are the same, use a small height
                                    # Use safe_convert to avoid numpy.int64 issues
                                    if open_idx == close_idx:
                                        body_height = 1.0
                                    else:
                                        body_height = safe_convert(abs(close_idx - open_idx))

                                    # Get the top and bottom of the body
                                    # Use safe_convert to avoid numpy.int64 issues
                                    body_top = safe_convert(max(open_idx, close_idx))
                                    body_bottom = safe_convert(min(open_idx, close_idx))

                                    # Draw the candle body
                                    # Use safe_convert to avoid numpy.int64 issues
                                    rect_x = safe_convert(t) - safe_convert(width)/2.0
                                    rect_width = safe_convert(width)

                                    if candlestick_type == "Hollow" and is_up:
                                        # Hollow candle for up moves (TOS style)
                                        heatmap_ax.add_patch(plt.Rectangle((rect_x, body_bottom),
                                                                        rect_width, body_height,
                                                                        fill=False, edgecolor=color, linewidth=1.0,
                                                                        zorder=20))
                                    else:
                                        # Filled candle
                                        heatmap_ax.add_patch(plt.Rectangle((rect_x, body_bottom),
                                                                        rect_width, body_height,
                                                                        facecolor=color, edgecolor=color, linewidth=1.0,
                                                                        zorder=20))

                                    # Draw the wicks
                                    # Only draw wicks if they extend beyond the body
                                    # Use thinner lines for wicks (0.7 instead of 1.5) to match ThinkorSwim style
                                    # Use safe_convert to avoid numpy.int64 issues
                                    t_float = safe_convert(t)

                                    if low_idx < body_bottom:  # Lower wick
                                        heatmap_ax.plot([t_float, t_float], [safe_convert(low_idx), safe_convert(body_bottom)],
                                                      color=color, linewidth=0.8, zorder=20)
                                    if high_idx > body_top:  # Upper wick
                                        heatmap_ax.plot([t_float, t_float], [safe_convert(body_top), safe_convert(high_idx)],
                                                      color=color, linewidth=0.8, zorder=20)
                                except Exception as e:
                                    print(f"Error drawing candlestick at time {t}: {e}")
                                    continue  # Skip this candlestick if there's an error

                            # Successfully added candlesticks
                            candlestick_added = True

                            # Now that we have the candlesticks, generate the heatmap
                            # This ensures the heatmap is properly aligned with the candlesticks

                            # Calculate the number of visible candlesticks based on time slider
                            num_visible_points = max(2, int(len(valid_data_points) * time_percent)) if time_percent < 1.0 else len(valid_data_points)

                            # Set the heatmap extent to match the visible candlesticks
                            # The heatmap should start at 0 (first candlestick) and end at the last visible candlestick
                            # Use safe_convert to avoid numpy.int64 issues
                            x_min, x_max = 0.0, safe_convert(num_visible_points - 1)

                            # Add padding to ensure the heatmap covers the entire visible area
                            x_min -= 0.5  # Start half a candlestick before the first one
                            x_max += 0.5  # End half a candlestick after the last one

                            # Set the y-axis limits to the strike range
                            # Use safe_convert to avoid numpy.int64 issues
                            y_min, y_max = safe_convert(min_strike), safe_convert(max_strike)

                            # Initialize heatmap data with the correct dimensions
                            Z = np.zeros((len(strike_range), num_visible_points))

                            # Use real options data for the heatmap instead of synthetic data
                            # Get the options data for the selected expiry date
                            if not self.calls_df.empty and not self.puts_df.empty:
                                # Filter options data to the strike range
                                calls_filtered = self.calls_df[(self.calls_df['strike'] >= min_strike) &
                                                            (self.calls_df['strike'] <= max_strike)].copy()
                                puts_filtered = self.puts_df[(self.puts_df['strike'] >= min_strike) &
                                                           (self.puts_df['strike'] <= max_strike)].copy()

                                # Create a dictionary to store the values for each strike
                                strike_values = {}

                                # Process based on heatmap type
                                if heatmap_type == "Gamma":
                                    # Use GEX (Gamma Exposure) from options data
                                    for _, row in calls_filtered.iterrows():
                                        strike = row['strike']
                                        if strike not in strike_values:
                                            strike_values[strike] = 0
                                        strike_values[strike] += row['GEX']

                                    for _, row in puts_filtered.iterrows():
                                        strike = row['strike']
                                        if strike not in strike_values:
                                            strike_values[strike] = 0
                                        strike_values[strike] += row['GEX']

                                elif heatmap_type == "Delta Pressure":
                                    # Use DEX (Delta Exposure) from options data
                                    for _, row in calls_filtered.iterrows():
                                        strike = row['strike']
                                        if strike not in strike_values:
                                            strike_values[strike] = 0
                                        strike_values[strike] += row['DEX']

                                    for _, row in puts_filtered.iterrows():
                                        strike = row['strike']
                                        if strike not in strike_values:
                                            strike_values[strike] = 0
                                        strike_values[strike] += row['DEX']

                                elif heatmap_type == "Charm Pressure":
                                    # Use CEX (Charm Exposure) from options data
                                    for _, row in calls_filtered.iterrows():
                                        strike = row['strike']
                                        if strike not in strike_values:
                                            strike_values[strike] = 0
                                        strike_values[strike] += row['CEX']

                                    for _, row in puts_filtered.iterrows():
                                        strike = row['strike']
                                        if strike not in strike_values:
                                            strike_values[strike] = 0
                                        strike_values[strike] += row['CEX']

                                # Normalize the values to a reasonable range for visualization
                                if strike_values:
                                    max_abs_value = max(abs(v) for v in strike_values.values())
                                    if max_abs_value > 0:
                                        for strike in strike_values:
                                            strike_values[strike] = strike_values[strike] / max_abs_value

                                # Fill the heatmap matrix with real data
                                for i, strike in enumerate(strike_range):
                                    # Find the closest strike in our data
                                    closest_strike = min(strike_values.keys(), key=lambda x: abs(x - strike), default=None) if strike_values else None

                                    if closest_strike is not None:
                                        # Get the value for this strike
                                        value = strike_values[closest_strike]

                                        # Fill the row with values that increase with time
                                        for j in range(num_visible_points):
                                            # Time factor increases as we approach expiry
                                            time_factor = (j / num_visible_points) * time_percent
                                            # Value increases with time (more pronounced near expiry)
                                            Z[i, j] = value * (0.5 + time_factor * 0.5)
                            else:
                                # Fallback to synthetic data if no options data is available
                                # Display a warning to the user
                                QMessageBox.warning(self, "Warning", "No options data available. Using synthetic data for visualization.")

                                if heatmap_type == "Gamma":
                                    # Simulate gamma exposure across time and price
                                    for i, strike in enumerate(strike_range):
                                        for j in range(num_visible_points):
                                            distance = abs(strike - self.current_price) / self.current_price
                                            time_factor = (j / num_visible_points) * time_percent

                                            if distance < 0.02:  # Close to current price
                                                Z[i, j] = 0.8 - distance * 30  # Strong positive gamma
                                            elif distance < 0.05:  # Slightly away from current price
                                                Z[i, j] = 0.3 - distance * 8  # Weak positive gamma
                                            else:  # Far from current price
                                                Z[i, j] = -0.3 + distance * 2  # Negative gamma

                                            Z[i, j] *= (0.5 + time_factor * 0.8)

                                elif heatmap_type == "Delta Pressure":
                                    # Use actual delta exposure data
                                    # First, create a dictionary to store DEX by strike
                                    dex_by_strike = {}

                                    # Aggregate DEX by strike from calls and puts
                                    for _, row in calls_filtered.iterrows():
                                        strike = row['strike']
                                        if strike not in dex_by_strike:
                                            dex_by_strike[strike] = 0
                                        dex_by_strike[strike] += row['DEX']

                                    for _, row in puts_filtered.iterrows():
                                        strike = row['strike']
                                        if strike not in dex_by_strike:
                                            dex_by_strike[strike] = 0
                                        dex_by_strike[strike] += row['DEX']

                                    # Normalize DEX values for visualization
                                    max_dex = max(abs(v) for v in dex_by_strike.values()) if dex_by_strike else 1.0
                                    max_dex = max(max_dex, 1.0)  # Avoid division by zero

                                    # Apply DEX values to the heatmap
                                    for i, strike in enumerate(strike_range):
                                        # Find the closest strike in our data
                                        closest_strike = min(dex_by_strike.keys(), key=lambda x: abs(x - strike)) if dex_by_strike else strike
                                        dex_value = dex_by_strike.get(closest_strike, 0) / max_dex

                                        for j in range(num_visible_points):
                                            time_factor = (j / num_visible_points) * time_percent
                                            # Scale the effect by time (stronger closer to expiry)
                                            Z[i, j] = dex_value * (0.3 + time_factor * 0.7)

                                elif heatmap_type == "Charm Pressure":
                                    # Simulate charm pressure
                                    for i, strike in enumerate(strike_range):
                                        for j in range(num_visible_points):
                                            distance = abs(strike - self.current_price) / self.current_price
                                            time_factor = (j / num_visible_points) * time_percent

                                            if distance < 0.02:  # Close to current price
                                                Z[i, j] = -0.8 * (0.2 + time_factor * 0.8)
                                            elif distance < 0.05:  # Slightly away from current price
                                                Z[i, j] = -0.5 * (0.2 + time_factor * 0.8) * (1.0 - distance * 8)
                                            else:  # Far from current price
                                                Z[i, j] = -0.2 * (0.2 + time_factor * 0.8) * (1.0 - distance * 5)

                            # Create the heatmap
                            cmap = plt.cm.coolwarm

                            # Use imshow instead of pcolormesh for better control over the extent
                            # Set zorder to -1 to ensure the heatmap is behind the candlesticks
                            heatmap = heatmap_ax.imshow(Z, cmap=cmap, aspect='auto',
                                                      extent=[x_min, x_max, y_min, y_max],
                                                      origin='lower', interpolation='bilinear',
                                                      zorder=-1)

                            # Add contour lines to highlight gradient changes
                            # This makes it easier to visualize the transitions between different values

                            # Ensure we have a good range of values for the contour levels
                            z_min, z_max = np.min(Z), np.max(Z)
                            # If the range is too small, create a more meaningful range
                            if abs(z_max - z_min) < 0.1:
                                if heatmap_type == "Gamma":
                                    levels = np.linspace(-1.0, 1.0, 10)  # Gamma typically ranges from -1 to 1
                                elif heatmap_type == "Delta Pressure":
                                    levels = np.linspace(-0.5, 0.5, 10)  # Delta pressure ranges
                                elif heatmap_type == "Charm Pressure":
                                    levels = np.linspace(-1.0, 0.0, 10)  # Charm is typically negative
                                else:
                                    levels = np.linspace(-1.0, 1.0, 10)  # Default range
                            else:
                                # Round the min and max to create cleaner level values
                                z_min_rounded = np.floor(z_min * 10) / 10
                                z_max_rounded = np.ceil(z_max * 10) / 10
                                levels = np.linspace(z_min_rounded, z_max_rounded, 10)

                            # Create contour lines with safe_convert to avoid numpy.int64 issues
                            x_range = np.linspace(safe_convert(x_min), safe_convert(x_max), safe_convert(Z.shape[1]))
                            y_range = np.linspace(safe_convert(y_min), safe_convert(y_max), safe_convert(Z.shape[0]))

                            contours = heatmap_ax.contour(x_range, y_range, Z,
                                                       levels=levels, colors='black', alpha=1.0,
                                                       linewidths=0.5, linestyles='dotted', zorder=-0.5)

                            # Skip trying to update contour line styles - they're already set in the contour() call
                            # The linestyles='dotted' parameter in the contour() call is sufficient

                            # Add contour labels for key levels (every other level to avoid clutter)
                            # Format the labels to be concise
                            fmt = {level: f'{level:.2f}' for level in levels[::2]}  # Label every other level with 2 decimal places
                            heatmap_ax.clabel(contours, levels[::2], inline=True, fmt=fmt, fontsize=7, colors='black')

                            # Add grid lines to help with visualization
                            # These grid lines make it easier to see the structure of the data
                            # Use a subtle grid that doesn't overwhelm the visualization
                            heatmap_ax.grid(True, color='black', alpha=0.3, linestyle='dotted', linewidth=0.3, zorder=-0.8)

                            # Add colorbar
                            cbar = self.canvas.fig.colorbar(heatmap, ax=heatmap_ax)
                            if heatmap_type == "Gamma":
                                cbar.set_label('Gamma Exposure', color='white')
                            elif heatmap_type == "Delta Pressure":
                                cbar.set_label('Delta Pressure', color='white')
                            elif heatmap_type == "Charm Pressure":
                                cbar.set_label('Charm Pressure', color='white')

                # If we couldn't add candlesticks, use a flat line and generate a basic heatmap
                if not candlestick_added:
                    print("Using flat price line due to insufficient historical data")
                    heatmap_ax.axhline(self.current_price, color='yellow', linewidth=2,
                                     label='Current Price', zorder=10)

                    # Generate a basic heatmap with default settings
                    num_time_points = 100

                    # Set the heatmap extent
                    # Use explicit float conversion to avoid numpy.int64 issues
                    x_min, x_max = 0.0, float(num_time_points - 1)
                    y_min, y_max = float(min_strike), float(max_strike)

                    # Initialize heatmap data
                    Z = np.zeros((len(strike_range), num_time_points))

                    # Use real options data for the heatmap instead of synthetic data
                    if not self.calls_df.empty and not self.puts_df.empty:
                        # Filter options data to the strike range
                        calls_filtered = self.calls_df[(self.calls_df['strike'] >= min_strike) &
                                                    (self.calls_df['strike'] <= max_strike)].copy()
                        puts_filtered = self.puts_df[(self.puts_df['strike'] >= min_strike) &
                                                   (self.puts_df['strike'] <= max_strike)].copy()

                        # Create a dictionary to store the values for each strike
                        strike_values = {}

                        # Process based on heatmap type
                        if heatmap_type == "Gamma":
                            # Use GEX (Gamma Exposure) from options data
                            for _, row in calls_filtered.iterrows():
                                strike = row['strike']
                                if strike not in strike_values:
                                    strike_values[strike] = 0
                                strike_values[strike] += row['GEX']

                            for _, row in puts_filtered.iterrows():
                                strike = row['strike']
                                if strike not in strike_values:
                                    strike_values[strike] = 0
                                strike_values[strike] += row['GEX']

                        elif heatmap_type == "Delta Pressure":
                            # Use DEX (Delta Exposure) from options data
                            for _, row in calls_filtered.iterrows():
                                strike = row['strike']
                                if strike not in strike_values:
                                    strike_values[strike] = 0
                                strike_values[strike] += row['DEX']

                            for _, row in puts_filtered.iterrows():
                                strike = row['strike']
                                if strike not in strike_values:
                                    strike_values[strike] = 0
                                strike_values[strike] += row['DEX']

                        elif heatmap_type == "Charm Pressure":
                            # Use CEX (Charm Exposure) from options data
                            for _, row in calls_filtered.iterrows():
                                strike = row['strike']
                                if strike not in strike_values:
                                    strike_values[strike] = 0
                                strike_values[strike] += row['CEX']

                            for _, row in puts_filtered.iterrows():
                                strike = row['strike']
                                if strike not in strike_values:
                                    strike_values[strike] = 0
                                strike_values[strike] += row['CEX']

                        # Normalize the values to a reasonable range for visualization
                        if strike_values:
                            max_abs_value = max(abs(v) for v in strike_values.values())
                            if max_abs_value > 0:
                                for strike in strike_values:
                                    strike_values[strike] = strike_values[strike] / max_abs_value

                        # Fill the heatmap matrix with real data
                        for i, strike in enumerate(strike_range):
                            # Find the closest strike in our data
                            closest_strike = min(strike_values.keys(), key=lambda x: abs(x - strike), default=None) if strike_values else None

                            if closest_strike is not None:
                                # Get the value for this strike
                                value = strike_values[closest_strike]

                                # Fill the row with values that increase with time
                                for j in range(num_time_points):
                                    # Time factor increases as we approach expiry
                                    time_factor = (j / num_time_points) * time_percent
                                    # Value increases with time (more pronounced near expiry)
                                    Z[i, j] = value * (0.5 + time_factor * 0.5)
                    else:
                        # Fallback to synthetic data if no options data is available
                        # Display a warning to the user
                        QMessageBox.warning(self, "Warning", "No options data available. Using synthetic data for visualization.")

                        if heatmap_type == "Gamma":
                            # Simulate gamma exposure across time and price
                            for i, strike in enumerate(strike_range):
                                for j in range(num_time_points):
                                    distance = abs(strike - self.current_price) / self.current_price
                                    time_factor = (j / num_time_points) * time_percent

                                    if distance < 0.02:
                                        Z[i, j] = 0.8 - distance * 30
                                    elif distance < 0.05:
                                        Z[i, j] = 0.3 - distance * 8
                                    else:
                                        Z[i, j] = -0.3 + distance * 2

                                    Z[i, j] *= (0.5 + time_factor * 0.8)
                        elif heatmap_type == "Delta Pressure":
                            # Use actual delta exposure data
                            # First, create a dictionary to store DEX by strike
                            dex_by_strike = {}

                            # Aggregate DEX by strike from calls and puts
                            for _, row in calls_filtered.iterrows():
                                strike = row['strike']
                                if strike not in dex_by_strike:
                                    dex_by_strike[strike] = 0
                                dex_by_strike[strike] += row['DEX']

                            for _, row in puts_filtered.iterrows():
                                strike = row['strike']
                                if strike not in dex_by_strike:
                                    dex_by_strike[strike] = 0
                                dex_by_strike[strike] += row['DEX']

                            # Normalize DEX values for visualization
                            max_dex = max(abs(v) for v in dex_by_strike.values()) if dex_by_strike else 1.0
                            max_dex = max(max_dex, 1.0)  # Avoid division by zero

                            # Apply DEX values to the heatmap
                            for i, strike in enumerate(strike_range):
                                # Find the closest strike in our data
                                closest_strike = min(dex_by_strike.keys(), key=lambda x: abs(x - strike)) if dex_by_strike else strike
                                dex_value = dex_by_strike.get(closest_strike, 0) / max_dex

                                for j in range(num_time_points):
                                    time_factor = (j / num_time_points) * time_percent
                                    # Scale the effect by time (stronger closer to expiry)
                                    Z[i, j] = dex_value * (0.3 + time_factor * 0.7)
                        elif heatmap_type == "Charm Pressure":
                            for i, strike in enumerate(strike_range):
                                for j in range(num_time_points):
                                    distance = abs(strike - self.current_price) / self.current_price
                                    time_factor = (j / num_time_points) * time_percent

                                    if distance < 0.02:  # Close to current price
                                        # Strong charm effect near expiry
                                        Z[i, j] = -0.8 * (0.2 + time_factor * 0.8)
                                    elif distance < 0.05:  # Slightly away from current price
                                        # Moderate charm effect
                                        Z[i, j] = -0.5 * (0.2 + time_factor * 0.8) * (1.0 - distance * 8)
                                    else:  # Far from current price
                                        # Weaker charm effect away from the money
                                        Z[i, j] = -0.2 * (0.2 + time_factor * 0.8) * (1.0 - distance * 5)

                    # Create the heatmap
                    cmap = plt.cm.coolwarm
                    heatmap = heatmap_ax.imshow(Z, cmap=cmap, aspect='auto',
                                              extent=[x_min, x_max, y_min, y_max],
                                              origin='lower', interpolation='bilinear',
                                              zorder=-1)

                    # Add contour lines to highlight gradient changes
                    # This makes it easier to visualize the transitions between different values

                    # Ensure we have a good range of values for the contour levels
                    z_min, z_max = np.min(Z), np.max(Z)
                    # If the range is too small, create a more meaningful range
                    if abs(z_max - z_min) < 0.1:
                        if heatmap_type == "Gamma":
                            levels = np.linspace(-1.0, 1.0, 10)  # Gamma typically ranges from -1 to 1
                        elif heatmap_type == "Delta Pressure":
                            levels = np.linspace(-0.5, 0.5, 10)  # Delta pressure ranges
                        elif heatmap_type == "Charm Pressure":
                            levels = np.linspace(-1.0, 0.0, 10)  # Charm is typically negative
                        else:
                            levels = np.linspace(-1.0, 1.0, 10)  # Default range
                    else:
                        # Round the min and max to create cleaner level values
                        z_min_rounded = np.floor(z_min * 10) / 10
                        z_max_rounded = np.ceil(z_max * 10) / 10
                        levels = np.linspace(z_min_rounded, z_max_rounded, 10)

                    # Create contour lines with safe_convert to avoid numpy.int64 issues
                    x_range = np.linspace(safe_convert(x_min), safe_convert(x_max), safe_convert(Z.shape[1]))
                    y_range = np.linspace(safe_convert(y_min), safe_convert(y_max), safe_convert(Z.shape[0]))

                    contours = heatmap_ax.contour(x_range, y_range, Z,
                                               levels=levels, colors='black', alpha=1.0,
                                               linewidths=0.5, linestyles='dotted', zorder=-0.5)

                    # Skip trying to update contour line styles - they're already set in the contour() call
                    # The linestyles='dotted' parameter in the contour() call is sufficient

                    # Add contour labels for key levels (every other level to avoid clutter)
                    # Format the labels to be concise
                    fmt = {level: f'{level:.2f}' for level in levels[::2]}  # Label every other level with 2 decimal places
                    heatmap_ax.clabel(contours, levels[::2], inline=True, fmt=fmt, fontsize=7, colors='black')

                    # Add grid lines to help with visualization
                    # These grid lines make it easier to see the structure of the data
                    # Use a subtle grid that doesn't overwhelm the visualization
                    heatmap_ax.grid(True, color='black', alpha=0.3, linestyle='dotted', linewidth=0.3, zorder=-0.8)

                    # Add colorbar
                    cbar = self.canvas.fig.colorbar(heatmap, ax=heatmap_ax)
                    if heatmap_type == "Gamma":
                        cbar.set_label('Gamma Exposure', color='white')
                    elif heatmap_type == "Delta Pressure":
                        cbar.set_label('Delta Pressure', color='white')
                    elif heatmap_type == "Charm Pressure":
                        cbar.set_label('Charm Pressure', color='white')
            except Exception as e:
                print(f"Error adding candlesticks: {e}")
                # Fallback to flat line at current price
                heatmap_ax.axhline(self.current_price, color='yellow', linewidth=2,
                                 label='Current Price', zorder=10)

                # Generate a basic heatmap with default settings
                num_time_points = 100

                # Set the heatmap extent
                # Use explicit float conversion to avoid numpy.int64 issues
                x_min, x_max = 0.0, float(num_time_points - 1)
                y_min, y_max = float(min_strike), float(max_strike)

                # Initialize heatmap data
                Z = np.zeros((len(strike_range), num_time_points))

                # Use real options data for the heatmap instead of synthetic data
                if not self.calls_df.empty and not self.puts_df.empty:
                    # Filter options data to the strike range
                    calls_filtered = self.calls_df[(self.calls_df['strike'] >= min_strike) &
                                                (self.calls_df['strike'] <= max_strike)].copy()
                    puts_filtered = self.puts_df[(self.puts_df['strike'] >= min_strike) &
                                               (self.puts_df['strike'] <= max_strike)].copy()

                    # Create a dictionary to store the values for each strike
                    strike_values = {}

                    # Process based on heatmap type
                    if heatmap_type == "Gamma":
                        # Use GEX (Gamma Exposure) from options data
                        for _, row in calls_filtered.iterrows():
                            strike = row['strike']
                            if strike not in strike_values:
                                strike_values[strike] = 0
                            strike_values[strike] += row['GEX']

                        for _, row in puts_filtered.iterrows():
                            strike = row['strike']
                            if strike not in strike_values:
                                strike_values[strike] = 0
                            strike_values[strike] += row['GEX']

                    elif heatmap_type == "Delta Pressure":
                        # Use DEX (Delta Exposure) from options data
                        for _, row in calls_filtered.iterrows():
                            strike = row['strike']
                            if strike not in strike_values:
                                strike_values[strike] = 0
                            strike_values[strike] += row['DEX']

                        for _, row in puts_filtered.iterrows():
                            strike = row['strike']
                            if strike not in strike_values:
                                strike_values[strike] = 0
                            strike_values[strike] += row['DEX']

                    elif heatmap_type == "Charm Pressure":
                        # Use CEX (Charm Exposure) from options data
                        for _, row in calls_filtered.iterrows():
                            strike = row['strike']
                            if strike not in strike_values:
                                strike_values[strike] = 0
                            strike_values[strike] += row['CEX']

                        for _, row in puts_filtered.iterrows():
                            strike = row['strike']
                            if strike not in strike_values:
                                strike_values[strike] = 0
                            strike_values[strike] += row['CEX']

                    # Normalize the values to a reasonable range for visualization
                    if strike_values:
                        max_abs_value = max(abs(v) for v in strike_values.values())
                        if max_abs_value > 0:
                            for strike in strike_values:
                                strike_values[strike] = strike_values[strike] / max_abs_value

                    # Fill the heatmap matrix with real data
                    for i, strike in enumerate(strike_range):
                        # Find the closest strike in our data
                        closest_strike = min(strike_values.keys(), key=lambda x: abs(x - strike), default=None) if strike_values else None

                        if closest_strike is not None:
                            # Get the value for this strike
                            value = strike_values[closest_strike]

                            # Fill the row with values that increase with time
                            for j in range(num_time_points):
                                # Time factor increases as we approach expiry
                                time_factor = (j / num_time_points) * time_percent
                                # Value increases with time (more pronounced near expiry)
                                Z[i, j] = value * (0.5 + time_factor * 0.5)
                else:
                    # Fallback to synthetic data if no options data is available
                    # Display a warning to the user
                    QMessageBox.warning(self, "Warning", "No options data available. Using synthetic data for visualization.")

                    if heatmap_type == "Gamma":
                        # Simulate gamma exposure across time and price
                        for i, strike in enumerate(strike_range):
                            for j in range(num_time_points):
                                distance = abs(strike - self.current_price) / self.current_price
                                time_factor = (j / num_time_points) * time_percent

                                if distance < 0.02:
                                    Z[i, j] = 0.8 - distance * 30
                                elif distance < 0.05:
                                    Z[i, j] = 0.3 - distance * 8
                                else:
                                    Z[i, j] = -0.3 + distance * 2

                                Z[i, j] *= (0.5 + time_factor * 0.8)
                    elif heatmap_type == "Delta Pressure":
                        # Use actual delta exposure data
                        # First, create a dictionary to store DEX by strike
                        dex_by_strike = {}

                        # Aggregate DEX by strike from calls and puts
                        for _, row in calls_filtered.iterrows():
                            strike = row['strike']
                            if strike not in dex_by_strike:
                                dex_by_strike[strike] = 0
                            dex_by_strike[strike] += row['DEX']

                        for _, row in puts_filtered.iterrows():
                            strike = row['strike']
                            if strike not in dex_by_strike:
                                dex_by_strike[strike] = 0
                            dex_by_strike[strike] += row['DEX']

                        # Normalize DEX values for visualization
                        max_dex = max(abs(v) for v in dex_by_strike.values()) if dex_by_strike else 1.0
                        max_dex = max(max_dex, 1.0)  # Avoid division by zero

                        # Apply DEX values to the heatmap
                        for i, strike in enumerate(strike_range):
                            # Find the closest strike in our data
                            closest_strike = min(dex_by_strike.keys(), key=lambda x: abs(x - strike)) if dex_by_strike else strike
                            dex_value = dex_by_strike.get(closest_strike, 0) / max_dex

                            for j in range(num_time_points):
                                time_factor = (j / num_time_points) * time_percent
                                # Scale the effect by time (stronger closer to expiry)
                                Z[i, j] = dex_value * (0.3 + time_factor * 0.7)
                    elif heatmap_type == "Charm Pressure":
                        for i, strike in enumerate(strike_range):
                            for j in range(num_time_points):
                                distance = abs(strike - self.current_price) / self.current_price
                                time_factor = (j / num_time_points) * time_percent

                                if distance < 0.02:  # Close to current price
                                    # Strong charm effect near expiry
                                    Z[i, j] = -0.8 * (0.2 + time_factor * 0.8)
                                elif distance < 0.05:  # Slightly away from current price
                                    # Moderate charm effect
                                    Z[i, j] = -0.5 * (0.2 + time_factor * 0.8) * (1.0 - distance * 8)
                                else:  # Far from current price
                                    # Weaker charm effect away from the money
                                    Z[i, j] = -0.2 * (0.2 + time_factor * 0.8) * (1.0 - distance * 5)

                # Create the heatmap
                cmap = plt.cm.coolwarm
                heatmap = heatmap_ax.imshow(Z, cmap=cmap, aspect='auto',
                                          extent=[x_min, x_max, y_min, y_max],
                                          origin='lower', interpolation='bilinear',
                                          zorder=-1)

                # Add contour lines to highlight gradient changes
                # This makes it easier to visualize the transitions between different values

                # Ensure we have a good range of values for the contour levels
                z_min, z_max = np.min(Z), np.max(Z)
                # If the range is too small, create a more meaningful range
                if abs(z_max - z_min) < 0.1:
                    if heatmap_type == "Gamma":
                        levels = np.linspace(-1.0, 1.0, 10)  # Gamma typically ranges from -1 to 1
                    elif heatmap_type == "Delta Pressure":
                        levels = np.linspace(-0.5, 0.5, 10)  # Delta pressure ranges
                    elif heatmap_type == "Charm Pressure":
                        levels = np.linspace(-1.0, 0.0, 10)  # Charm is typically negative
                    else:
                        levels = np.linspace(-1.0, 1.0, 10)  # Default range
                else:
                    # Round the min and max to create cleaner level values
                    z_min_rounded = np.floor(z_min * 10) / 10
                    z_max_rounded = np.ceil(z_max * 10) / 10
                    levels = np.linspace(z_min_rounded, z_max_rounded, 10)

                # Create contour lines with safe_convert to avoid numpy.int64 issues
                x_range = np.linspace(safe_convert(x_min), safe_convert(x_max), safe_convert(Z.shape[1]))
                y_range = np.linspace(safe_convert(y_min), safe_convert(y_max), safe_convert(Z.shape[0]))

                contours = heatmap_ax.contour(x_range, y_range, Z,
                                           levels=levels, colors='black', alpha=1.0,
                                           linewidths=0.5, linestyles='dotted', zorder=-0.5)

                # Skip trying to update contour line styles - they're already set in the contour() call
                # The linestyles='dotted' parameter in the contour() call is sufficient

                # Add contour labels for key levels (every other level to avoid clutter)
                # Format the labels to be concise
                fmt = {level: f'{level:.2f}' for level in levels[::2]}  # Label every other level with 2 decimal places
                heatmap_ax.clabel(contours, levels[::2], inline=True, fmt=fmt, fontsize=7, colors='black')

                # Add grid lines to help with visualization
                # These grid lines make it easier to see the structure of the data
                # Use a subtle grid that doesn't overwhelm the visualization
                heatmap_ax.grid(True, color='black', alpha=0.3, linestyle='dotted', linewidth=0.3, zorder=-0.8)

                # Add colorbar
                cbar = self.canvas.fig.colorbar(heatmap, ax=heatmap_ax)
                if heatmap_type == "Gamma":
                    cbar.set_label('Gamma Exposure', color='white')
                elif heatmap_type == "Delta Pressure":
                    cbar.set_label('Delta Pressure', color='white')
                elif heatmap_type == "Charm Pressure":
                    cbar.set_label('Charm Pressure', color='white')

            # Create the strike plot
            # Prepare data for the strike plot based on the selected type
            strikes = sorted(set(calls_filtered['strike']) | set(puts_filtered['strike']))

            # Get bucket size for time bucketing
            bucket_size = self.bucket_size_selector.currentText()

            # Prepare all data types regardless of selected view for Combined view
            # Prepare GEX data
            call_gex = {}
            put_gex = {}

            # Prepare OI data
            call_oi = {}
            put_oi = {}

            # Prepare Net OI data
            net_oi = {}

            # Prepare flow data for time buckets
            flow_data = {}

            # Get focus strike if specified
            focus_strike = None
            if self.focus_strike_input.text().strip():
                try:
                    focus_strike = float(self.focus_strike_input.text().strip())
                except ValueError:
                    print(f"Invalid focus strike: {self.focus_strike_input.text().strip()}")

            # Calculate all data types
            for strike in strikes:
                call_data = calls_filtered[calls_filtered['strike'] == strike]
                put_data = puts_filtered[puts_filtered['strike'] == strike]

                # Calculate GEX
                call_gex[strike] = safe_convert(call_data['GEX'].sum() if not call_data.empty else 0)
                put_gex[strike] = safe_convert(put_data['GEX'].sum() if not put_data.empty else 0)

                # Calculate OI
                call_oi[strike] = safe_convert(call_data['openInterest'].sum() if not call_data.empty else 0)
                put_oi[strike] = safe_convert(put_data['openInterest'].sum() if not put_data.empty else 0)

                # Calculate Net OI
                net_oi[strike] = safe_convert(call_oi[strike] - put_oi[strike])

                # Generate flow data based on options activity
                # In a real implementation, this would use actual trade data
                # For now, we'll create more realistic synthetic data based on options activity
                flow_data[strike] = []

                # Parse bucket size to minutes
                bucket_minutes = 10  # Default
                if bucket_size.endswith('m'):
                    bucket_minutes = int(bucket_size[:-1])
                elif bucket_size.endswith('h'):
                    bucket_minutes = int(bucket_size[:-1]) * 60

                # Calculate number of buckets based on market hours (6.5 hours = 390 minutes)
                num_buckets = 390 // bucket_minutes

                # Get options data for this strike
                call_data = calls_filtered[calls_filtered['strike'] == strike]
                put_data = puts_filtered[puts_filtered['strike'] == strike]

                # Calculate base flow magnitude from options data
                call_volume = call_data['volume'].sum() if not call_data.empty else 0
                put_volume = put_data['volume'].sum() if not put_data.empty else 0
                call_oi_value = call_data['openInterest'].sum() if not call_data.empty else 0
                put_oi_value = put_data['openInterest'].sum() if not put_data.empty else 0

                # Calculate a base flow magnitude from the options data
                base_magnitude = 0.0
                if call_volume + put_volume > 0:
                    # Use volume as a base for flow magnitude
                    base_magnitude = (call_volume + put_volume) / 1000  # Scale down large numbers
                elif call_oi_value + put_oi_value > 0:
                    # Fallback to open interest if no volume
                    base_magnitude = (call_oi_value + put_oi_value) / 2000  # Scale down large numbers
                else:
                    # If no options data, use distance from current price
                    distance = abs(strike - self.current_price) / self.current_price
                    base_magnitude = max(0.2, 1.0 - distance * 10)

                # Cap the base magnitude to a reasonable range
                base_magnitude = min(2.0, max(0.2, base_magnitude))

                # Generate flow data for each time bucket
                import random
                for i in range(num_buckets):
                    # Add time-based variation (more activity near market open and close)
                    time_factor = 1.0
                    if i < num_buckets * 0.2 or i > num_buckets * 0.8:  # First or last 20% of the day
                        time_factor = 1.5  # More activity near open/close

                    # Add randomness to the magnitude
                    flow_magnitude = base_magnitude * time_factor * random.uniform(0.7, 1.3)

                    # Determine flow direction based on call/put ratio and strike position
                    call_put_ratio = call_volume / max(1, put_volume)  # Avoid division by zero

                    # More positive flow for strikes with higher call volume relative to puts
                    # More negative flow for strikes with higher put volume relative to calls
                    if call_put_ratio > 1.2:  # More calls than puts
                        flow_sign = 1 if random.random() > 0.3 else -1  # 70% chance of positive flow
                    elif call_put_ratio < 0.8:  # More puts than calls
                        flow_sign = -1 if random.random() > 0.3 else 1  # 70% chance of negative flow
                    else:  # Roughly equal calls and puts
                        # Use strike position relative to current price
                        if strike > self.current_price:
                            flow_sign = 1 if random.random() > 0.4 else -1  # 60% chance of positive flow above price
                        else:
                            flow_sign = -1 if random.random() > 0.4 else 1  # 60% chance of negative flow below price

                    # Calculate final flow value
                    flow = flow_sign * flow_magnitude

                    # Make some buckets have zero flow (realistic market behavior)
                    if random.random() < 0.2:  # 20% chance of no flow in a bucket
                        flow = 0.0

                    flow_data[strike].append(flow)

            # Plot based on selected type
            if strike_plot_type == "GEX":
                # Plot GEX
                strike_ax.bar(strikes, [call_gex.get(s, 0) for s in strikes],
                             color='green', alpha=0.7, label='Call GEX')
                strike_ax.bar(strikes, [put_gex.get(s, 0) for s in strikes],
                             color='red', alpha=0.7, label='Put GEX')
                strike_ax.set_title("Gamma Exposure by Strike", color='white')
                strike_ax.set_ylabel("GEX", color='white')

                # Add flow dots if focus strike is specified
                if focus_strike is not None and focus_strike in flow_data:
                    self._plot_flow_dots(strike_ax, focus_strike, flow_data[focus_strike])

            elif strike_plot_type == "OI":
                # Plot OI
                strike_ax.bar(strikes, [call_oi.get(s, 0) for s in strikes],
                             color='orange', alpha=0.7, label='Call OI')
                strike_ax.bar(strikes, [put_oi.get(s, 0) for s in strikes],
                             color='blue', alpha=0.7, label='Put OI')
                strike_ax.set_title("Open Interest by Strike", color='white')
                strike_ax.set_ylabel("OI", color='white')

                # Add flow dots if focus strike is specified
                if focus_strike is not None and focus_strike in flow_data:
                    self._plot_flow_dots(strike_ax, focus_strike, flow_data[focus_strike])

            elif strike_plot_type == "Net OI":
                # Plot Net OI
                colors = ['green' if v >= 0 else 'red' for v in [net_oi.get(s, 0) for s in strikes]]
                strike_ax.bar(strikes, [net_oi.get(s, 0) for s in strikes], color=colors, alpha=0.7)
                strike_ax.set_title("Net Open Interest by Strike", color='white')
                strike_ax.set_ylabel("Net OI", color='white')

                # Add flow dots if focus strike is specified
                if focus_strike is not None and focus_strike in flow_data:
                    self._plot_flow_dots(strike_ax, focus_strike, flow_data[focus_strike])

            elif strike_plot_type == "Combined":
                # Create a figure with 3 subplots for GEX, OI, and Net OI
                self.canvas.fig.clear()
                gs = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3, figure=self.canvas.fig)
                heatmap_ax = self.canvas.fig.add_subplot(gs[0])
                gex_ax = self.canvas.fig.add_subplot(gs[1])
                oi_ax = self.canvas.fig.add_subplot(gs[2])
                net_oi_ax = self.canvas.fig.add_subplot(gs[3])

                # Plot GEX
                gex_ax.bar(strikes, [call_gex.get(s, 0) for s in strikes],
                         color='green', alpha=0.7, label='Call GEX')
                gex_ax.bar(strikes, [put_gex.get(s, 0) for s in strikes],
                         color='red', alpha=0.7, label='Put GEX')
                gex_ax.set_title("Gamma Exposure", color='white', fontsize=10)
                gex_ax.set_ylabel("GEX", color='white', fontsize=8)
                gex_ax.tick_params(axis='both', which='major', labelsize=8)

                # Plot OI
                oi_ax.bar(strikes, [call_oi.get(s, 0) for s in strikes],
                        color='orange', alpha=0.7, label='Call OI')
                oi_ax.bar(strikes, [put_oi.get(s, 0) for s in strikes],
                        color='blue', alpha=0.7, label='Put OI')
                oi_ax.set_title("Open Interest", color='white', fontsize=10)
                oi_ax.set_ylabel("OI", color='white', fontsize=8)
                oi_ax.tick_params(axis='both', which='major', labelsize=8)

                # Plot Net OI
                colors = ['green' if v >= 0 else 'red' for v in [net_oi.get(s, 0) for s in strikes]]
                net_oi_ax.bar(strikes, [net_oi.get(s, 0) for s in strikes], color=colors, alpha=0.7)
                net_oi_ax.set_title("Net Open Interest", color='white', fontsize=10)
                net_oi_ax.set_ylabel("Net OI", color='white', fontsize=8)
                net_oi_ax.set_xlabel("Strike Price", color='white', fontsize=8)
                net_oi_ax.tick_params(axis='both', which='major', labelsize=8)

                # Add current price line to all subplots
                for ax in [gex_ax, oi_ax, net_oi_ax]:
                    ax.axvline(self.current_price, color='white', linestyle='--', alpha=0.7)
                    # Add backtested price line if different from current price
                    backtested_price = self.get_backtested_price()
                    if backtested_price is not None and self.backtesting_slider.value() != 0:
                        ax.axvline(backtested_price, color='yellow', linestyle='-.', alpha=0.9, linewidth=1.5)

                # Add flow dots if focus strike is specified
                if focus_strike is not None and focus_strike in flow_data:
                    self._plot_flow_dots(gex_ax, focus_strike, flow_data[focus_strike])

                # Set dark background for all plots
                for ax in [heatmap_ax, gex_ax, oi_ax, net_oi_ax]:
                    ax.set_facecolor('#1e1e1e')
                    ax.tick_params(colors='white')

                # Continue with heatmap plotting
                strike_ax = heatmap_ax  # Use heatmap_ax for the rest of the function

            # Add current price line to strike plot
            strike_ax.axvline(self.current_price, color='white', linestyle='--', alpha=0.7,
                             label=f'Current Price: {self.current_price:.2f}')

            # Add backtested price line to strike plot if different from current price
            backtested_price = self.get_backtested_price()
            if backtested_price is not None and self.backtesting_slider.value() != 0:
                strike_ax.axvline(backtested_price, color='yellow', linestyle='-.', alpha=0.9, linewidth=1.5,
                                 label=f'Backtested Price: {backtested_price:.2f} ({self.backtesting_slider.value():+d}%)')

            # Add volatility trigger line if enabled
            if self.show_vol_trigger_checkbox.isChecked() and hasattr(self, 'volatility_trigger') and self.volatility_trigger is not None:
                # Use a distinctive purple color for the volatility trigger
                strike_ax.axvline(self.volatility_trigger, color='#9370DB', linestyle='-',
                                alpha=0.9, linewidth=2.0, label=f'Vol Trigger: {self.volatility_trigger:.2f}')

                # Add text annotation above the volatility trigger
                _, y_max = strike_ax.get_ylim()
                strike_ax.text(self.volatility_trigger, y_max * 0.9, f'Vol Trigger: {self.volatility_trigger:.2f}',
                            color='#9370DB', ha='center', va='bottom', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

            # Format strike plot with larger font sizes
            strike_ax.set_xlabel("Strike Price", color='white', fontsize=11)
            strike_ax.tick_params(axis='both', labelsize=10)
            strike_ax.legend(fontsize=10)

            # Set dark background for both plots
            self.canvas.fig.patch.set_facecolor('#121212')
            heatmap_ax.set_facecolor('#1e1e1e')
            strike_ax.set_facecolor('#1e1e1e')

            # Set text colors to white and increase font sizes
            for ax in [heatmap_ax, strike_ax]:
                ax.tick_params(colors='white', labelsize=10)
                ax.xaxis.label.set_color('white')
                ax.xaxis.label.set_fontsize(11)
                ax.yaxis.label.set_color('white')
                ax.yaxis.label.set_fontsize(11)
                ax.title.set_color('white')
                ax.title.set_fontsize(12)

            # Add legend to heatmap with larger font size
            heatmap_ax.legend(loc='upper right', framealpha=0.7, facecolor='#1e1e1e', edgecolor='#444444', fontsize=10)

            # Add overall title
            ticker = self.ticker_input.text().strip().upper()
            expiry = self.expiry_selector.currentText()
            today = datetime.today().date()
            selected_expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
            days_to_expiry = (selected_expiry - today).days

            # Add backtesting indicator to title if backtesting
            backtesting_text = ""
            if self.backtesting_slider.value() != 0:
                backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

            title = f"TRACE - {ticker} (Exp: {expiry}, DTE: {days_to_expiry}){backtesting_text}"
            if zero_dte_only:
                title += " - 0DTE Only"
            self.canvas.fig.suptitle(title, color='white', fontsize=14)

            # Adjust layout - use subplots_adjust instead of tight_layout to avoid warnings
            # Increase the size of the chart by using more of the available space
            self.canvas.fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, hspace=0.3)  # Make room for suptitle and maximize chart area

            # Enable zoom functionality if the zoom checkbox is checked
            if hasattr(self, 'trace_zoom_checkbox') and self.trace_zoom_checkbox.isChecked():
                # Enable zoom functionality for both subplots
                self.enable_trace_zoom()

            # Make sure crosshair is visible for TRACE chart
            # Reset crosshair elements to ensure they're visible
            self.canvas.cursor_hline.set_visible(True)
            self.canvas.cursor_vline.set_visible(True)
            self.canvas.text_box.set_visible(True)

            # Crosshair styling is now applied when creating the new lines in the next step

            # Set initial position for crosshair (middle of the chart)
            # This ensures the crosshair is visible even before mouse movement
            if hasattr(self.canvas.axes, 'get_xlim') and hasattr(self.canvas.axes, 'get_ylim'):
                xlim = self.canvas.axes.get_xlim()
                ylim = self.canvas.axes.get_ylim()
                mid_x = (xlim[0] + xlim[1]) / 2
                mid_y = (ylim[0] + ylim[1]) / 2

                # Safely remove existing lines and create new ones at the middle position
                try:
                    if self.canvas.cursor_hline in self.canvas.axes.lines:
                        self.canvas.cursor_hline.remove()
                    if self.canvas.cursor_vline in self.canvas.axes.lines:
                        self.canvas.cursor_vline.remove()
                except Exception as e:
                    # If we can't remove the lines, just continue
                    print(f"Could not remove crosshair lines in update_trace: {e}")
                    pass

                # Create new lines at the middle position with the trace style
                self.canvas.cursor_hline = self.canvas.axes.axhline(
                    y=mid_y,
                    color=self.canvas.trace_crosshair_style['color'],
                    lw=self.canvas.trace_crosshair_style['lw'],
                    ls=self.canvas.trace_crosshair_style['ls']
                )

                self.canvas.cursor_vline = self.canvas.axes.axvline(
                    x=mid_x,
                    color=self.canvas.trace_crosshair_style['color'],
                    lw=self.canvas.trace_crosshair_style['lw'],
                    ls=self.canvas.trace_crosshair_style['ls']
                )

            # Set the exposure type for the canvas to enable proper crosshair behavior
            self.canvas.exposure_type = "TRACE"

            # Draw the canvas
            self.canvas.draw()

            return True

        except Exception as e:
            import traceback
            print(f"Error in update_trace: {e}")
            print("Detailed traceback:")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error updating TRACE visualization: {str(e)}")
            return False

    def toggle_trace_zoom(self):
        """Toggle zoom functionality for the TRACE component"""
        # Update the chart to apply or remove zoom functionality
        self.update_trace()

    def show_focus_strike_help(self):
        """Show help information about the focus strike feature"""
        QMessageBox.information(self, "Focus Strike Help",
            "<b>Focus Strike Feature</b><br><br>"
            "The Focus Strike feature allows you to visualize options flow activity at a specific strike price:<br><br>"
            "<b>How to use:</b><br>"
            "1. Enter a strike price in the Focus Strike field<br>"
            "2. Press Enter to update the chart<br><br>"
            "<b>What you'll see:</b><br>"
            "- A vertical yellow dotted line at the specified strike<br>"
            "- Colored dots representing trading activity (flow) at that strike<br>"
            "- Green/lime dots indicate buying pressure (positive flow)<br>"
            "- Red/magenta dots indicate selling pressure (negative flow)<br>"
            "- The size of each dot represents the magnitude of the flow<br><br>"
            "<b>Tips:</b><br>"
            "- Try focusing on strikes with high open interest or volume<br>"
            "- Compare flow patterns at different strike prices<br>"
            "- Look for clusters of similar-colored dots indicating consistent directional flow")

    def enable_trace_zoom(self):
        """Enable zoom functionality for the TRACE component"""
        try:
            # Enable zoom functionality for both subplots in the TRACE component
            # This uses matplotlib's built-in zoom and pan tools

            # Get the figure and axes from the canvas
            fig = self.canvas.fig
            axes = fig.get_axes()

            # Create a toolbar for the canvas if it doesn't exist
            if not hasattr(self.canvas, 'toolbar') or self.canvas.toolbar is None:
                # Use the NavigationToolbar2QT that we imported at the top of the file
                self.canvas.toolbar = NavigationToolbar2QT(self.canvas, self)
                # Add the toolbar to the layout
                # Find the canvas in the layout and add the toolbar below it
                for i in range(self.layout().count()):
                    item = self.layout().itemAt(i)
                    if item.widget() == self.canvas_container:
                        # Add the toolbar to the canvas container layout
                        self.canvas_container.layout().addWidget(self.canvas.toolbar)
                        break

            # Make the toolbar visible
            self.canvas.toolbar.setVisible(True)

            # Enable zoom for all axes
            for ax in axes:
                ax.set_navigate(True)
                ax.set_navigate_mode('PAN')  # Default to pan mode

            # Connect mouse events for zooming
            if not hasattr(self, '_trace_zoom_cid'):
                self._trace_zoom_cid = self.canvas.mpl_connect('scroll_event', self._trace_zoom_scroll)

            # Add a message to the plot indicating zoom is enabled
            for ax in axes:
                ax.text(0.02, 0.98, "Zoom Enabled", transform=ax.transAxes,
                       color='white', fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='#444444', alpha=0.7))

            # Redraw the canvas
            self.canvas.draw()
        except Exception as e:
            print(f"Error enabling TRACE zoom: {e}")

    def _plot_flow_dots(self, ax, strike, flow_values):
        """Plot flow dots for a specific strike

        Args:
            ax: The matplotlib axis to plot on
            strike: The strike price to plot flow for
            flow_values: List of flow values for each time bucket
        """
        try:
            # Get the y-range of the plot to position dots
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min

            # Calculate dot positions - make them more visible by positioning higher
            dot_y_base = y_min + y_range * 0.2  # Position dots higher in the plot (20% from bottom)
            dot_spacing = y_range * 0.03  # Increase space between dots

            # Add a vertical line at the focus strike for better visibility
            ax.axvline(strike, color='yellow', linestyle=':', linewidth=1.5, alpha=0.7, zorder=14)

            # Plot dots for each time bucket
            for i, flow in enumerate(flow_values):
                # Skip if flow is too small
                if abs(flow) < 0.1:
                    continue

                # Determine color based on flow direction
                color = 'lime' if flow > 0 else 'magenta'  # More visible colors

                # Calculate dot size based on flow magnitude - make dots larger
                size = min(200, max(50, abs(flow) * 100))  # Increased size

                # Calculate dot position
                dot_x = strike
                dot_y = dot_y_base + i * dot_spacing

                # Plot the dot with a black edge for better visibility
                ax.scatter(dot_x, dot_y, s=size, color=color, edgecolors='black', linewidths=1, alpha=0.8, zorder=15)

            # Add a more visible label to indicate flow dots
            ax.text(strike, dot_y_base + len(flow_values) * dot_spacing, "FLOW ACTIVITY",
                   ha='center', va='bottom', color='yellow', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='#444444', alpha=0.8, edgecolor='yellow', linewidth=1))

            # Add a note about the focus strike at the top of the chart
            ax.set_title(f"Focus Strike: {strike:.2f}", color='yellow', fontsize=10, pad=5)

            # Print a message to confirm the focus strike is being plotted
            print(f"Plotting flow dots for strike {strike} with {len(flow_values)} flow values")

        except Exception as e:
            print(f"Error plotting flow dots: {e}")

    def _trace_zoom_scroll(self, event):
        """Handle mouse scroll events for zooming in the TRACE component"""
        try:
            # Only process if zoom is enabled
            if not self.trace_zoom_checkbox.isChecked():
                return

            # Get the axes that contains the event
            ax = event.inaxes
            if ax is None:
                return

            # Get the current x and y limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Determine the zoom factor based on scroll direction
            zoom_factor = 0.9 if event.button == 'up' else 1.1

            # Calculate new limits centered on the mouse position
            x_range = float(xlim[1] - xlim[0])  # Convert to float to avoid numpy.int64 issues
            y_range = float(ylim[1] - ylim[0])  # Convert to float to avoid numpy.int64 issues

            # Calculate new ranges
            new_x_range = x_range * zoom_factor
            new_y_range = y_range * zoom_factor

            # Calculate new limits centered on the mouse position
            x_center = float(event.xdata) if event.xdata is not None else float(xlim[0] + x_range/2)
            y_center = float(event.ydata) if event.ydata is not None else float(ylim[0] + y_range/2)

            # Set new limits with explicit float conversion
            ax.set_xlim([float(x_center - new_x_range/2), float(x_center + new_x_range/2)])
            ax.set_ylim([float(y_center - new_y_range/2), float(y_center + new_y_range/2)])

            # Redraw the canvas
            self.canvas.draw()
        except Exception as e:
            print(f"Error in TRACE zoom scroll: {e}")

    def fetch_historical_data(self, interval=None):
        """Fetch historical price data for the current ticker

        Args:
            interval (str, optional): Data interval (e.g., '1d', '1h', '1m').
                                     If None, uses the timeframe selector value.
        """
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            return

        try:
            # Format ticker for indices
            ticker = self.format_ticker(ticker)

            # Get interval from selector or use provided interval
            if interval is None:
                interval = self.timeframe_selector.currentText()

            # Get days to load from input field
            try:
                days_to_load = int(self.days_to_load_input.text())
                if days_to_load <= 0:
                    days_to_load = 30  # Default to 30 days if invalid input
            except ValueError:
                days_to_load = 30  # Default to 30 days if not a number
                self.days_to_load_input.setText(str(days_to_load))  # Update the input field

            # Calculate the start date based on days to load
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_to_load)

            # Fetch historical data
            stock = yf.Ticker(ticker)

            # For intraday data (1m, 5m, etc.), Yahoo Finance has limitations
            # 1m data is only available for the last 7 days
            # 5m data is only available for the last 60 days
            # Adjust the start date accordingly
            if interval == '1m':
                # For 1-minute data, limit to last 7 days
                start_date = max(start_date, end_date - timedelta(days=7))
            elif interval in ['2m', '5m', '15m', '30m', '60m', '90m']:
                # For other intraday data, limit to last 60 days
                start_date = max(start_date, end_date - timedelta(days=60))

            try:
                self.historical_data = stock.history(start=start_date, end=end_date, interval=interval)

                if self.historical_data.empty:
                    print(f"No historical data available for {ticker} with interval {interval}")
                    return False

                print(f"Successfully fetched {len(self.historical_data)} data points for {ticker} with interval {interval}")
                return True
            except Exception as e:
                print(f"Error fetching historical data with interval {interval}: {e}")
                return False
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error fetching historical data: {str(e)}")
            return False

    def calculate_heikin_ashi(self, df):
        """Calculate Heikin Ashi candlestick values"""
        ha_df = pd.DataFrame(index=df.index)

        # Calculate Heikin Ashi values
        ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

        # Initialize HA_Open with first candle's opening price
        ha_df['HA_Open'] = pd.Series(index=df.index)
        ha_df.loc[ha_df.index[0], 'HA_Open'] = df['Open'].iloc[0]

        # Calculate subsequent HA_Open values
        for i in range(1, len(df)):
            ha_df.loc[ha_df.index[i], 'HA_Open'] = (ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2

        ha_df['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
        ha_df['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

        return ha_df

    def update_candlestick_chart(self):
        """Update the candlestick chart with current data using PyQtGraph if available, otherwise fallback to matplotlib"""
        try:
            # Wrap the entire method in a try-except to catch any division by zero errors
            # Fetch historical data if needed
            if self.historical_data.empty or self.exposure_type == "CANDLESTICK":
                if not self.fetch_historical_data():
                    return

            # Calculate gamma flip points and top 5 strikes if we have options data
            gamma_flip_points = []
            top_strikes = []
            if not self.calls_df.empty and not self.puts_df.empty:
                # Calculate net gamma exposure
                calls_filtered = self.calls_df[['strike', 'GEX']].copy()
                puts_filtered = self.puts_df[['strike', 'GEX']].copy()

                # Get the price to center around (current or backtested)
                if self.current_price is not None:
                    center_price = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

                    # For 1-minute timeframe, use a wider strike range to ensure all important strikes are included
                    if '1m' in self.timeframe_selector.currentText():
                        # Use a wider range for 1-minute charts (2x the normal range)
                        effective_strike_range = self.strike_range * 2
                    else:
                        effective_strike_range = self.strike_range

                    min_strike = center_price - effective_strike_range
                    max_strike = center_price + effective_strike_range

                    # Filter data based on strike range
                    calls_filtered = calls_filtered[(calls_filtered['strike'] >= min_strike) &
                                                (calls_filtered['strike'] <= max_strike)]
                    puts_filtered = puts_filtered[(puts_filtered['strike'] >= min_strike) &
                                                (puts_filtered['strike'] <= max_strike)]

                    # Create a Series with all strikes
                    all_strikes = pd.Series(sorted(set(calls_filtered['strike']) |
                                                set(puts_filtered['strike'])))

                    # Initialize net exposure with zeros
                    net_exposure = pd.Series(0, index=all_strikes)

                    # Add call and put exposures
                    if not calls_filtered.empty:
                        call_exposure = calls_filtered.groupby('strike')['GEX'].sum()
                        net_exposure = net_exposure.add(call_exposure, fill_value=0)

                    if not puts_filtered.empty:
                        put_exposure = puts_filtered.groupby('strike')['GEX'].sum()
                        net_exposure = net_exposure.add(put_exposure, fill_value=0)

                    # Find where net exposure crosses zero (gamma flip points)
                    strikes = sorted(net_exposure.index)

                    for i in range(len(strikes) - 1):
                        current_strike = strikes[i]
                        next_strike = strikes[i + 1]
                        current_value = net_exposure[current_strike]
                        next_value = net_exposure[next_strike]

                        # Check if there's a sign change (crossing zero)
                        if (current_value * next_value <= 0) and (current_value != 0 or next_value != 0):
                            # Linear interpolation to find the exact zero-crossing point
                            if current_value == next_value or abs(next_value - current_value) < 1e-10:  # Avoid division by zero
                                flip_point = (current_strike + next_strike) / 2
                            else:
                                try:
                                    # Calculate the zero-crossing point using linear interpolation
                                    t = -current_value / (next_value - current_value)
                                    flip_point = current_strike + t * (next_strike - current_strike)
                                except ZeroDivisionError:
                                    # Fallback if division by zero occurs
                                    flip_point = (current_strike + next_strike) / 2

                            gamma_flip_points.append(flip_point)

                    # Find the 5 highest strikes by absolute gamma exposure
                    abs_exposure = net_exposure.abs()
                    # Make sure we get exactly 5 strikes if available
                    num_strikes = min(5, len(abs_exposure))
                    top_strikes_idx = abs_exposure.nlargest(num_strikes).index.tolist()

                    # Print debug info
                    print(f"Found {len(top_strikes_idx)} top strikes: {top_strikes_idx}")

                    # Store both the strike and its sign (positive or negative)
                    top_strikes = []
                    for strike in top_strikes_idx:
                        # Store as tuple: (strike_price, is_positive)
                        top_strikes.append((strike, net_exposure[strike] >= 0))

                    print(f"Final top_strikes list: {top_strikes}")

            # Check if we have the required OHLC columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in self.historical_data.columns for col in required_columns):
                QMessageBox.warning(self, "Warning", "Historical data does not contain required OHLC columns.")
                return

            # Get candlestick type
            candlestick_type = self.candlestick_type_selector.currentText()

            # Prepare data based on candlestick type
            if candlestick_type == "Heikin-Ashi":
                ha_data = self.calculate_heikin_ashi(self.historical_data)
                plot_data = self.historical_data.copy()
                plot_data['Open'] = ha_data['HA_Open']
                plot_data['High'] = ha_data['HA_High']
                plot_data['Low'] = ha_data['HA_Low']
                plot_data['Close'] = ha_data['HA_Close']
            else:
                plot_data = self.historical_data.copy()

            # Check if PyQtGraph is available
            if PYQTGRAPH_AVAILABLE:
                # Hide the matplotlib canvas and create a PyQtGraph widget if it doesn't exist
                self.canvas.hide()

                # Create PyQtGraph widget if it doesn't exist
                if not hasattr(self, 'pg_widget') or self.pg_widget is None:
                    # Create a new layout for the PyQtGraph widgets
                    self.pg_layout = QVBoxLayout()
                    self.pg_widget = QWidget()
                    self.pg_widget.setLayout(self.pg_layout)

                    # Add the widget to the canvas container
                    self.canvas_container.layout().addWidget(self.pg_widget)

                    # Create the plot widget (volume plot removed)
                    self.price_plot = pg.PlotWidget()

                    # Set background colors
                    self.price_plot.setBackground('#121212')

                    # Add plot to layout
                    self.pg_layout.addWidget(self.price_plot)
                else:
                    # Clear existing plot
                    self.price_plot.clear()

                # Show the PyQtGraph widget
                self.pg_widget.show()

                # Convert datetime index to timestamp for PyQtGraph
                timestamps = [i for i in range(len(plot_data))]

                # Prepare data for CandlestickItem
                ohlc_data = []
                for i, (_, row) in enumerate(plot_data.iterrows()):
                    # Use index position as x-coordinate
                    ohlc_data.append((i, row['Open'], row['High'], row['Low'], row['Close']))

                # Set colors based on the dashboard's color scheme
                bull_color = pg.mkColor(self.call_color)
                bear_color = pg.mkColor(self.put_color)

                # Width of candlesticks (adjusted based on data frequency)
                width = 0.6
                if '1m' in self.timeframe_selector.currentText() or '5m' in self.timeframe_selector.currentText():
                    width = 0.4
                elif '15m' in self.timeframe_selector.currentText() or '30m' in self.timeframe_selector.currentText():
                    width = 0.5

                # Create and add the candlestick item
                # For hollow candles, use a transparent bull color
                if candlestick_type == "Hollow":
                    # For hollow candles, we need to modify the CandlestickItem class to handle them specially
                    class HollowCandlestickItem(CandlestickItem):
                        def generatePicture(self):
                            self.picture = QPicture()

                            painter = QPainter(self.picture)

                            w = self.width / 2
                            for (t, open, high, low, close) in self.data:
                                if close >= open:
                                    # Bull candle (hollow)
                                    # Create a pen for hollow candle outlines and wicks with same color
                                    hollow_pen = QPen(pg.mkPen(self.bull_color, width=1.5))
                                    painter.setPen(hollow_pen)
                                    # No fill for hollow candles
                                    painter.setBrush(Qt.BrushStyle.NoBrush)
                                    # Draw the hollow candle body
                                    painter.drawRect(QRectF(t-w, open, w*2, close-open))

                                    # Draw the wick with same color as candle outline
                                    # Draw upper wick if needed
                                    if high > close:
                                        painter.drawLine(QPointF(t, close), QPointF(t, high))
                                    # Draw lower wick if needed
                                    if low < open:
                                        painter.drawLine(QPointF(t, open), QPointF(t, low))
                                else:
                                    # Bear candle (filled)
                                    # Create bear color pen with consistent width for both body and wicks
                                    bear_pen = QPen(pg.mkPen(self.bear_color, width=1.5))
                                    painter.setPen(bear_pen)
                                    # Set bear color brush
                                    painter.setBrush(pg.mkBrush(self.bear_color))
                                    # Draw the candle body without visible border
                                    painter.drawRect(QRectF(t-w, open, w*2, close-open))

                                    # Draw the wick with same color as candle
                                    # Draw upper wick if needed
                                    if high > open:
                                        painter.drawLine(QPointF(t, open), QPointF(t, high))
                                    # Draw lower wick if needed
                                    if low < close:
                                        painter.drawLine(QPointF(t, close), QPointF(t, low))
                            painter.end()

                    # Use the custom hollow candlestick item
                    candlestick_item = HollowCandlestickItem(ohlc_data, width=width, bull_color=bull_color, bear_color=bear_color)
                else:
                    # Use regular candlestick item for filled candles
                    candlestick_item = CandlestickItem(ohlc_data, width=width, bull_color=bull_color, bear_color=bear_color)
                self.price_plot.addItem(candlestick_item)

                # Moving Average functionality removed

                # Volume chart has been removed

                # Set chart title and labels
                ticker = self.ticker_input.text().strip().upper()
                interval = self.timeframe_selector.currentText()

                # Create title
                # Get days to load from input field
                days_to_load = self.days_to_load_input.text()

                # Add backtesting indicator to title if backtesting
                backtesting_text = ""
                if self.backtesting_slider.value() != 0:
                    backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

                title = f"{ticker} - {candlestick_type} Candlestick Chart ({interval}, DTL: {days_to_load}){backtesting_text}"
                if self.current_price is not None:
                    title += f" - Current: ${self.current_price:.2f}"

                # Set labels
                self.price_plot.setTitle(title, color='w')
                self.price_plot.setLabel('left', 'Price', color='w')

                # Add current price line if available
                if self.current_price is not None:
                    min_price = plot_data['Low'].min()
                    max_price = plot_data['High'].max()
                    if min_price <= self.current_price <= max_price:
                        current_price_line = pg.InfiniteLine(
                            pos=self.current_price,
                            angle=0,
                            pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine),
                            label=f'Current: ${self.current_price:.2f}',
                            labelOpts={'color': 'w', 'position': 0.1}
                        )
                        self.price_plot.addItem(current_price_line)

                    # Add backtested price line if different from current price
                    backtested_price = self.get_backtested_price()
                    if backtested_price is not None and self.backtesting_slider.value() != 0:
                        if min_price <= backtested_price <= max_price:
                            backtested_price_line = pg.InfiniteLine(
                                pos=backtested_price,
                                angle=0,
                                pen=pg.mkPen('yellow', width=1.5, style=Qt.PenStyle.DashDotLine),
                                label=f'Backtested: ${backtested_price:.2f} ({self.backtesting_slider.value():+d}%)',
                                labelOpts={'color': 'yellow', 'position': 0.1}
                            )
                            self.price_plot.addItem(backtested_price_line)

                # Add gamma flip points if available
                for i, flip_point in enumerate(gamma_flip_points):
                    # Create a horizontal line at the gamma flip point
                    flip_line = pg.InfiniteLine(
                        pos=flip_point,
                        angle=0,  # Horizontal line
                        pen=pg.mkPen('#5cb8b2', width=1.5),  # Softer teal color
                        label=f'Gamma Flip: {flip_point:.2f}',
                        labelOpts={'color': '#5cb8b2', 'position': 0.05}  # Position label at left
                    )
                    self.price_plot.addItem(flip_line)

                # Add volatility trigger if enabled and available
                if self.show_vol_trigger_checkbox.isChecked() and hasattr(self, 'volatility_trigger') and self.volatility_trigger is not None:
                    # Create a horizontal line at the volatility trigger level
                    vol_trigger_line = pg.InfiniteLine(
                        pos=self.volatility_trigger,
                        angle=0,  # Horizontal line
                        pen=pg.mkPen('#9370DB', width=2.0),  # Purple color
                        label=f'Vol Trigger: {self.volatility_trigger:.2f}',
                        labelOpts={'color': '#9370DB', 'position': 0.05, 'fill': '#000000', 'border': '#9370DB'}  # Position label at left
                    )
                    self.price_plot.addItem(vol_trigger_line)

                # Calculate the maximum absolute GEX value among top strikes for normalization
                max_abs_gex = 0
                strike_gex_values = {}

                # Get the GEX values for each top strike
                for strike, _ in top_strikes:
                    if strike in net_exposure.index:
                        gex_value = abs(net_exposure[strike])
                        strike_gex_values[strike] = gex_value
                        max_abs_gex = max(max_abs_gex, gex_value)

                # Add zone markers for top 5 strikes
                for i, (strike, is_positive) in enumerate(top_strikes):
                    # Always display all top strikes, regardless of visible price range
                    # Set color based on whether the strike has positive or negative exposure
                    if is_positive:
                        color = self.call_color  # Green/bullish for positive exposure
                        sign_text = '(+)'  # Positive sign
                    else:
                        color = self.put_color   # Red/bearish for negative exposure
                        sign_text = '(-)'  # Negative sign

                    # Calculate zone boundaries (±0.1% of strike price)
                    zone_lower = strike * 0.999  # -0.1%
                    zone_upper = strike * 1.001  # +0.1%

                    # Create a rectangle for the zone
                    # First, get the current x-range
                    x_min, x_max = self.price_plot.getViewBox().viewRange()[0]

                    # Create a semi-transparent rectangle for the zone
                    # Convert color to QColor to set alpha
                    zone_color = pg.mkColor(color)

                    # Calculate transparency based on GEX value relative to max GEX
                    # Stronger GEX = more opaque, weaker GEX = more transparent
                    # Scale between 10 (very transparent) and 80 (more opaque)
                    if strike in strike_gex_values and max_abs_gex > 0:
                        relative_strength = strike_gex_values[strike] / max_abs_gex
                        alpha = int(10 + relative_strength * 70)  # Scale to range 10-80 (out of 255)
                    else:
                        alpha = 40  # Default if GEX value not available

                    zone_color.setAlpha(alpha)  # Set transparency (0-255)

                    # Create a rectangle item for the zone
                    zone_rect = pg.QtWidgets.QGraphicsRectItem(x_min, zone_lower, x_max - x_min, zone_upper - zone_lower)
                    zone_rect.setBrush(pg.mkBrush(zone_color))
                    zone_rect.setPen(pg.mkPen(None))  # No border
                    self.price_plot.addItem(zone_rect)

                    # Add horizontal lines at zone boundaries with labels that follow the camera
                    upper_line = pg.InfiniteLine(
                        pos=zone_upper,
                        angle=0,  # Horizontal line
                        pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
                        label=f'{zone_upper:.2f}',  # Add label directly to the line
                        labelOpts={'color': color, 'position': 0.05, 'movable': True}  # Position label at left (5%)
                    )
                    lower_line = pg.InfiniteLine(
                        pos=zone_lower,
                        angle=0,  # Horizontal line
                        pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
                        label=f'{zone_lower:.2f}',  # Add label directly to the line
                        labelOpts={'color': color, 'position': 0.05, 'movable': True}  # Position label at left (5%)
                    )
                    self.price_plot.addItem(upper_line)
                    self.price_plot.addItem(lower_line)

                    # Add a label at the strike price
                    strike_label = pg.InfiniteLine(
                        pos=strike,
                        angle=0,  # Horizontal line
                        pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                        label=f'Top Strike Zone {sign_text}: {strike:.2f} (±0.1%)',
                        labelOpts={'color': color, 'position': 0.05}  # Position label at left
                    )
                    self.price_plot.addItem(strike_label)

                    # Add a star marker at the right edge of the chart
                    # Create a scatter plot item for the star
                    star = pg.ScatterPlotItem([x_max * 0.95], [strike], symbol='star', size=15,
                                             pen=pg.mkPen(color), brush=pg.mkBrush(color))
                    self.price_plot.addItem(star)

                # Set X axis to show dates
                date_strings = [idx.strftime('%Y-%m-%d %H:%M') for idx in plot_data.index]
                date_dict = {i: date for i, date in enumerate(date_strings)}
                self.price_plot.getAxis('bottom').setTicks([[(i, date_dict.get(i, '')) for i in range(0, len(timestamps), max(1, len(timestamps)//10))]])

                # Set axis colors
                self.price_plot.getAxis('left').setPen(pg.mkPen('w'))
                self.price_plot.getAxis('bottom').setPen(pg.mkPen('w'))

                # Set text colors
                self.price_plot.getAxis('left').setTextPen(pg.mkPen('w'))
                self.price_plot.getAxis('bottom').setTextPen(pg.mkPen('w'))

                # Enable mouse interaction
                self.price_plot.setMouseEnabled(x=True, y=True)

                # Adjust y-axis range to ensure all top strikes are visible
                if top_strikes:
                    # Get min and max strike values from top_strikes
                    strike_values = [strike for strike, _ in top_strikes]
                    min_strike_value = min(strike_values)
                    max_strike_value = max(strike_values)

                    # Get current y-range
                    y_min, y_max = self.price_plot.getViewBox().viewRange()[1]

                    # Expand range if needed to include all top strikes
                    new_y_min = min(y_min, min_strike_value * 0.998)  # Add a little extra margin
                    new_y_max = max(y_max, max_strike_value * 1.002)  # Add a little extra margin

                    # Set the new y-range
                    self.price_plot.setYRange(new_y_min, new_y_max)
                else:
                    # If no top strikes, use default auto-range
                    self.price_plot.enableAutoRange()

                # Grid removed as per user request
                self.price_plot.showGrid(x=False, y=False)
            else:
                # Fallback to matplotlib if PyQtGraph is not available
                # Show the matplotlib canvas
                self.canvas.show()

                # Clear the current chart
                self.canvas.fig.clear()

                # Create a single plot for the price chart (volume removed)
                self.canvas.axes = self.canvas.fig.add_subplot(111)

                # Create candlestick chart
                up_idx = plot_data.index[plot_data['Close'] >= plot_data['Open']]
                down_idx = plot_data.index[plot_data['Close'] < plot_data['Open']]

                # Width of candlesticks (adjusted based on data frequency)
                width = 0.6
                if '1m' in self.timeframe_selector.currentText() or '5m' in self.timeframe_selector.currentText():
                    width = 0.4
                elif '15m' in self.timeframe_selector.currentText() or '30m' in self.timeframe_selector.currentText():
                    width = 0.5

                # Plot up candles
                if candlestick_type == "Hollow":
                    # For hollow candles, only draw the outline for up candles
                    self.canvas.axes.bar(up_idx, plot_data.loc[up_idx, 'Close'] - plot_data.loc[up_idx, 'Open'],
                                        width, plot_data.loc[up_idx, 'Open'], color='none',
                                        edgecolor=self.call_color, linewidth=1, zorder=3)
                else:
                    # For regular and Heikin-Ashi, fill the up candles
                    self.canvas.axes.bar(up_idx, plot_data.loc[up_idx, 'Close'] - plot_data.loc[up_idx, 'Open'],
                                        width, plot_data.loc[up_idx, 'Open'], color=self.call_color,
                                        edgecolor=self.call_color, linewidth=1, zorder=3)

                # Plot down candles
                self.canvas.axes.bar(down_idx, plot_data.loc[down_idx, 'Close'] - plot_data.loc[down_idx, 'Open'],
                                    width, plot_data.loc[down_idx, 'Open'], color=self.put_color,
                                    edgecolor=self.put_color, linewidth=1, zorder=3)

                # Add high-low lines
                for idx in plot_data.index:
                    self.canvas.axes.plot([idx, idx], [plot_data.loc[idx, 'Low'], plot_data.loc[idx, 'High']],
                                        color='white' if idx in up_idx else self.put_color, linewidth=1, zorder=2)

                # Moving Average functionality removed

                # Volume chart has been removed

                # Set chart title and labels
                ticker = self.ticker_input.text().strip().upper()
                interval = self.timeframe_selector.currentText()

                # Add current price to title if available
                # Get days to load from input field
                days_to_load = self.days_to_load_input.text()

                # Add backtesting indicator to title if backtesting
                backtesting_text = ""
                if self.backtesting_slider.value() != 0:
                    backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

                title = f"{ticker} - {candlestick_type} Candlestick Chart ({interval}, DTL: {days_to_load}){backtesting_text}"
                if self.current_price is not None:
                    title += f" - Current: ${self.current_price:.2f}"

                    # Add current price line if it's within the chart range
                    min_price = plot_data['Low'].min()
                    max_price = plot_data['High'].max()
                    if min_price <= self.current_price <= max_price:
                        self.canvas.axes.axhline(y=self.current_price, color='white', linestyle='--',
                                               alpha=0.7, label=f'Current: ${self.current_price:.2f}')

                    # Add backtested price line if different from current price
                    backtested_price = self.get_backtested_price()
                    if backtested_price is not None and self.backtesting_slider.value() != 0:
                        if min_price <= backtested_price <= max_price:
                            self.canvas.axes.axhline(y=backtested_price, color='yellow', linestyle='-.',
                                                   alpha=0.9, linewidth=1.5,
                                                   label=f'Backtested: ${backtested_price:.2f} ({self.backtesting_slider.value():+d}%)')

                    # Add gamma flip points if available
                    for i, flip_point in enumerate(gamma_flip_points):
                        # Only add label for the first gamma flip point to avoid duplicate legend entries
                        if i == 0:
                            self.canvas.axes.axhline(y=flip_point, color='#5cb8b2', linestyle='-',
                                                alpha=0.9, linewidth=1.5, label=f'Gamma Flip')
                        else:
                            self.canvas.axes.axhline(y=flip_point, color='#5cb8b2', linestyle='-',
                                                alpha=0.9, linewidth=1.5)

                        # Add text annotation at the left edge of the chart
                        _, x_max = self.canvas.axes.get_xlim()  # Get x-axis limits
                        self.canvas.axes.text(x_max * 0.05, flip_point, f'Gamma Flip: {flip_point:.2f}',
                                            color='#5cb8b2', ha='left', va='center', fontsize=9,
                                            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))

                    # Add volatility trigger if enabled and available
                    if self.show_vol_trigger_checkbox.isChecked() and hasattr(self, 'volatility_trigger') and self.volatility_trigger is not None:
                        # Add a horizontal line at the volatility trigger level
                        self.canvas.axes.axhline(y=self.volatility_trigger, color='#9370DB', linestyle='-',
                                            alpha=0.9, linewidth=2.0, label=f'Vol Trigger')

                        # Add text annotation at the left edge of the chart
                        _, x_max = self.canvas.axes.get_xlim()  # Get x-axis limits
                        self.canvas.axes.text(x_max * 0.05, self.volatility_trigger, f'Vol Trigger: {self.volatility_trigger:.2f}',
                                            color='#9370DB', ha='left', va='center', fontsize=9, fontweight='bold',
                                            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

                    # Calculate the maximum absolute GEX value among top strikes for normalization
                    max_abs_gex = 0
                    strike_gex_values = {}

                    # Get the GEX values for each top strike
                    for strike, _ in top_strikes:
                        if strike in net_exposure.index:
                            gex_value = abs(net_exposure[strike])
                            strike_gex_values[strike] = gex_value
                            max_abs_gex = max(max_abs_gex, gex_value)

                    # Add zone markers for top 5 strikes
                    # Keep track of whether we've added positive and negative labels to the legend
                    added_pos_label = False
                    added_neg_label = False

                    for i, (strike, is_positive) in enumerate(top_strikes):
                        # Always display all top strikes, regardless of visible price range
                        # Set color and label based on whether the strike has positive or negative exposure
                        if is_positive:
                            color = self.call_color  # Green/bullish for positive exposure
                            sign_text = '(+)'  # Positive sign
                            label = 'Top Strike Zone (+)' if not added_pos_label else None
                            added_pos_label = True
                        else:
                            color = self.put_color   # Red/bearish for negative exposure
                            sign_text = '(-)'  # Negative sign
                            label = 'Top Strike Zone (-)' if not added_neg_label else None
                            added_neg_label = True

                        # Calculate zone boundaries (±0.1% of strike price)
                        zone_lower = strike * 0.999  # -0.1%
                        zone_upper = strike * 1.001  # +0.1%

                        # Get the x-axis range
                        x_min, x_max = self.canvas.axes.get_xlim()

                        # Create a semi-transparent rectangle for the zone
                        # Convert color to RGBA with alpha for transparency
                        zone_color = color

                        # Calculate base alpha based on GEX value relative to max GEX
                        # Stronger GEX = more opaque, weaker GEX = more transparent
                        # Scale between 0.05 (very transparent) and 0.3 (more opaque)
                        if strike in strike_gex_values and max_abs_gex > 0:
                            relative_strength = strike_gex_values[strike] / max_abs_gex
                            base_alpha = 0.05 + relative_strength * 0.25  # Scale to range 0.05-0.3
                        else:
                            base_alpha = 0.15  # Default if GEX value not available

                        if isinstance(zone_color, str):
                            # If it's a string like '#00FF00', convert to RGBA
                            if zone_color.startswith('#'):
                                # Convert hex to RGB and add alpha
                                r = int(zone_color[1:3], 16) / 255.0
                                g = int(zone_color[3:5], 16) / 255.0
                                b = int(zone_color[5:7], 16) / 255.0
                                zone_color = (r, g, b, base_alpha)  # Use calculated alpha

                        # Add the zone as a rectangle
                        self.canvas.axes.axhspan(zone_lower, zone_upper, xmin=0, xmax=1,
                                              facecolor=zone_color, alpha=base_alpha, zorder=1, label=label)

                        # Add horizontal lines at zone boundaries with appropriate color
                        # For Matplotlib, we can't make the labels follow the camera as easily as with PyQtGraph
                        # We'll use a different approach by adding the labels as annotations that will be
                        # positioned relative to the axes coordinates rather than data coordinates
                        self.canvas.axes.axhline(y=zone_upper, color=color, linestyle=':',
                                            alpha=0.8, linewidth=1)
                        self.canvas.axes.axhline(y=zone_lower, color=color, linestyle=':',
                                            alpha=0.8, linewidth=1)

                        # Create annotations that will stay in view when panning/zooming
                        # These use axes coordinates (0-1) for x and data coordinates for y

                        # Upper boundary label
                        upper_label = Annotation(f'{zone_upper:.2f}', xy=(0.05, zone_upper),
                                              xycoords=('axes fraction', 'data'),
                                              color=color, ha='left', va='center', fontsize=8)
                        self.canvas.axes.add_artist(upper_label)

                        # Lower boundary label
                        lower_label = Annotation(f'{zone_lower:.2f}', xy=(0.05, zone_lower),
                                              xycoords=('axes fraction', 'data'),
                                              color=color, ha='left', va='center', fontsize=8)
                        self.canvas.axes.add_artist(lower_label)

                        # Add a dashed line at the exact strike price
                        self.canvas.axes.axhline(y=strike, color=color, linestyle='--',
                                            alpha=0.8, linewidth=1)

                        # Add a star marker at the right edge of the chart
                        # Add the star marker with appropriate color
                        self.canvas.axes.plot(x_max * 0.95, strike, '*', color=color, markersize=15, alpha=0.8)

                        # Add text annotation for the strike value with sign indicator
                        self.canvas.axes.text(x_max * 0.95, strike, f'{strike:.2f} {sign_text} (±0.1%)',
                                            color=color, ha='right', va='bottom', fontsize=9,
                                            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))

                self.canvas.axes.set_title(title)
                self.canvas.axes.set_ylabel('Price')

                # Format x-axis to show dates nicely
                self.canvas.fig.autofmt_xdate()

                # Adjust y-axis range to ensure all top strikes are visible
                if top_strikes:
                    # Get min and max strike values from top_strikes
                    strike_values = [strike for strike, _ in top_strikes]
                    min_strike_value = min(strike_values)
                    max_strike_value = max(strike_values)

                    # Get current y-range
                    y_min, y_max = self.canvas.axes.get_ylim()

                    # Expand range if needed to include all top strikes
                    new_y_min = min(y_min, min_strike_value * 0.998)  # Add a little extra margin
                    new_y_max = max(y_max, max_strike_value * 1.002)  # Add a little extra margin

                    # Set the new y-range
                    self.canvas.axes.set_ylim(new_y_min, new_y_max)

                # Set dark background
                self.canvas.fig.patch.set_facecolor('#121212')
                self.canvas.axes.set_facecolor('#1e1e1e')
                self.canvas.axes.tick_params(colors='white')
                self.canvas.axes.xaxis.label.set_color('white')
                self.canvas.axes.yaxis.label.set_color('white')
                self.canvas.axes.title.set_color('white')

                # Remove grid as per user request
                self.canvas.axes.grid(False)

                # Adjust layout
                self.canvas.fig.tight_layout()

                # Draw the canvas
                self.canvas.draw()

        except ZeroDivisionError as e:
            print(f"Division by zero error in update_candlestick_chart: {e}")
            QMessageBox.critical(self, "Error", f"Error fetching data: division by zero. This usually happens with invalid or missing data.")
        except Exception as e:
            print(f"Error in update_candlestick_chart: {e}")
            QMessageBox.critical(self, "Error", f"Error updating candlestick chart: {str(e)}")

    def update_gamma_landscape(self):
        """Update the 3D gamma landscape visualization"""
        try:
            if self.current_price is None:
                return

            # Completely recreate the 3D axes to avoid colorbar issues
            # First, clear the figure
            self.canvas_3d.fig.clear()

            # Create a new 3D axes
            self.canvas_3d.axes = self.canvas_3d.fig.add_subplot(111, projection='3d')

            # Reset the colorbar reference
            self.canvas_3d.colorbar = None

            # Get current date and calculate time to expiration
            today = datetime.today().date()
            selected_expiry = datetime.strptime(self.expiry_selector.currentText(), '%Y-%m-%d').date()
            t_days = max((selected_expiry - today).days, 1)  # Ensure at least 1 day
            t = t_days / 365.0

            # Risk-free rate (could be fetched from a source like ^IRX)
            r = 0.05  # Using 5% as a default

            # Get the price to center around (current or backtested)
            S = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

            # Create strike price range around current price
            strike_range = self.strike_range
            min_strike = S - strike_range
            max_strike = S + strike_range

            # Get grid points from input
            try:
                grid_points = int(self.grid_points_input.text())
                if grid_points < 10:
                    grid_points = 10  # Minimum grid size
                elif grid_points > 100:
                    grid_points = 100  # Maximum grid size for performance
            except ValueError:
                grid_points = 50  # Default if invalid input

            strikes = np.linspace(min_strike, max_strike, grid_points)

            # Get volatility range from inputs
            try:
                min_vol = float(self.min_vol_input.text())
                if min_vol < 0.01:
                    min_vol = 0.01  # Minimum volatility
            except ValueError:
                min_vol = 0.1  # Default if invalid input

            try:
                max_vol = float(self.max_vol_input.text())
                if max_vol > 2.0:
                    max_vol = 2.0  # Maximum volatility
                if max_vol <= min_vol:
                    max_vol = min_vol + 0.1  # Ensure max > min
            except ValueError:
                max_vol = 1.0  # Default if invalid input

            vols = np.linspace(min_vol, max_vol, grid_points)

            # Create meshgrid for 3D surface
            K, V = np.meshgrid(strikes, vols)

            # Calculate gamma values for the grid
            gamma_values = np.zeros_like(K)
            for i in range(len(vols)):
                for j in range(len(strikes)):
                    try:
                        # Calculate gamma using py_vollib
                        # Use selected option type (call or put)
                        option_type = 'c' if self.option_type_selector.currentText() == "Call" else 'p'
                        gamma_val = bs_gamma(option_type, S, K[i, j], t, r, V[i, j])
                        gamma_values[i, j] = gamma_val
                    except Exception:
                        gamma_values[i, j] = 0

            # Create the 3D surface plot
            surf = self.canvas_3d.axes.plot_surface(
                K, V, gamma_values,
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=True,
                alpha=0.8
            )

            # No need to remove previous colorbar since we recreated the axes

            # Add a new color bar
            try:
                # First, adjust the figure layout to make room for the colorbar
                self.canvas_3d.fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)  # Leave space on the right for colorbar

                # Create the colorbar in the reserved space
                cax = self.canvas_3d.fig.add_axes([0.85, 0.2, 0.03, 0.6])  # [left, bottom, width, height]
                self.canvas_3d.colorbar = self.canvas_3d.fig.colorbar(surf, cax=cax)
            except Exception as e:
                print(f"Error creating colorbar: {e}")
                self.canvas_3d.colorbar = None

            # Set labels and title
            ticker = self.ticker_input.text().strip().upper()
            expiry = self.expiry_selector.currentText()

            self.canvas_3d.axes.set_xlabel('Strike Price')
            self.canvas_3d.axes.set_ylabel('Implied Volatility')
            self.canvas_3d.axes.set_zlabel('Gamma')
            option_type = "Call" if self.option_type_selector.currentText() == "Call" else "Put"
            # Add backtesting indicator to title if backtesting
            backtesting_text = ""
            if self.backtesting_slider.value() != 0:
                backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

            self.canvas_3d.axes.set_title(f"Gamma Landscape ({option_type}) - {ticker} (Exp: {expiry}){backtesting_text}\nCurrent Price: {self.current_price:.2f}, Backtested Price: {S:.2f}" if self.backtesting_slider.value() != 0 else f"Gamma Landscape ({option_type}) - {ticker} (Exp: {expiry})\nCurrent Price: {S:.2f}")

            # Mark current price on the surface
            # Find the closest strike to current price
            closest_strike_idx = np.abs(strikes - S).argmin()
            # Add a line at current price through all volatilities
            vol_line = np.linspace(min_vol, max_vol, grid_points)
            strike_line = np.ones_like(vol_line) * strikes[closest_strike_idx]
            gamma_line = np.zeros_like(vol_line)
            for i, vol in enumerate(vol_line):
                try:
                    option_type = 'c' if self.option_type_selector.currentText() == "Call" else 'p'
                    gamma_line[i] = bs_gamma(option_type, S, strikes[closest_strike_idx], t, r, vol)
                except Exception:
                    gamma_line[i] = 0

            self.canvas_3d.axes.plot(strike_line, vol_line, gamma_line, 'r-', linewidth=2)

            # Set dark background
            self.canvas_3d.fig.patch.set_facecolor('#121212')
            self.canvas_3d.axes.set_facecolor('#1e1e1e')

            # Set text colors to white
            self.canvas_3d.axes.xaxis.label.set_color('white')
            self.canvas_3d.axes.yaxis.label.set_color('white')
            self.canvas_3d.axes.zaxis.label.set_color('white')
            self.canvas_3d.axes.title.set_color('white')

            # Set tick colors to white
            self.canvas_3d.axes.tick_params(axis='x', colors='white')
            self.canvas_3d.axes.tick_params(axis='y', colors='white')
            self.canvas_3d.axes.tick_params(axis='z', colors='white')

            # Set view angle based on selection
            view_angle = self.view_angle_selector.currentText()
            if view_angle == "Top":
                self.canvas_3d.axes.view_init(elev=90, azim=-90)
            elif view_angle == "Side":
                self.canvas_3d.axes.view_init(elev=0, azim=0)
            elif view_angle == "Front":
                self.canvas_3d.axes.view_init(elev=0, azim=90)
            else:  # Default
                self.canvas_3d.axes.view_init(elev=30, azim=30)

            # No need for tight_layout since we're manually setting the layout

            # Draw the canvas
            try:
                self.canvas_3d.draw()
            except Exception as e:
                print(f"Error drawing 3D canvas: {e}")

        except Exception as e:
            print(f"Error in update_gamma_landscape: {e}")

    def update_chart(self):
        """Update the chart with current data"""
        if self.calls_df.empty and self.puts_df.empty:
            return

        try:
            # Completely recreate the figure and axes to avoid colorbar issues
            self.canvas.fig.clear()
            self.canvas.axes = self.canvas.fig.add_subplot(111)

            # Reset colorbar reference
            if hasattr(self.canvas, 'colorbar'):
                self.canvas.colorbar = None

            # Store filtered data for crosshair lookup
            self.filtered_calls = pd.DataFrame()
            self.filtered_puts = pd.DataFrame()

            # Get chart type
            chart_type = self.chart_type_selector.currentText()

            # Get the current exposure type
            exposure_type = self.exposure_type  # VEX, GEX, DEX, CEX, MPX, OI, DELTA_PROFILE, etc.

            # Handle special cases for profile visualizations, TRACE, and CANDLESTICK
            if exposure_type == "DELTA_PROFILE":
                return self.update_delta_profile()
            elif exposure_type == "VEGA_PROFILE":
                return self.update_vega_profile()
            elif exposure_type == "VOMMA_PROFILE":
                return self.update_vomma_profile()
            elif exposure_type == "TRACE":
                return self.update_trace()  # Handle TRACE separately
            elif exposure_type == "CANDLESTICK":
                return self.update_candlestick_chart()  # Handle CANDLESTICK separately

            # Handle OI, PCR, IVSKEW, and ACTIVITY_MAP differently than other exposure types
            if exposure_type == "OI":
                # Filter data for Open Interest
                calls_filtered = self.calls_df[['strike', 'openInterest']].copy()
                calls_filtered = calls_filtered[calls_filtered['openInterest'] > 0]

                puts_filtered = self.puts_df[['strike', 'openInterest']].copy()
                puts_filtered = puts_filtered[puts_filtered['openInterest'] > 0]
            elif exposure_type == "PCR":
                # Filter data for Put/Call Ratio
                calls_filtered = self.calls_df[['strike', 'PCR']].copy()
                calls_filtered = calls_filtered[~calls_filtered['PCR'].isna()]

                puts_filtered = self.puts_df[['strike', 'PCR']].copy()
                puts_filtered = puts_filtered[~puts_filtered['PCR'].isna()]
            elif exposure_type == "IVSKEW":
                # Filter data for IV Skew
                calls_filtered = self.calls_df[['strike', 'IVSKEW']].copy()
                calls_filtered = calls_filtered[~calls_filtered['IVSKEW'].isna()]

                puts_filtered = self.puts_df[['strike', 'IVSKEW']].copy()
                puts_filtered = puts_filtered[~puts_filtered['IVSKEW'].isna()]
            elif exposure_type == "ACTIVITY_MAP":
                # For activity map, we need both volume and open interest
                calls_filtered = self.calls_df[['strike', 'volume', 'openInterest']].copy()
                puts_filtered = self.puts_df[['strike', 'volume', 'openInterest']].copy()
            else:
                # Filter out zero values for other exposure types
                calls_filtered = self.calls_df[['strike', exposure_type]].copy()
                calls_filtered = calls_filtered[calls_filtered[exposure_type] != 0]

                puts_filtered = self.puts_df[['strike', exposure_type]].copy()
                puts_filtered = puts_filtered[puts_filtered[exposure_type] != 0]

            # Get the price to center around (current or backtested)
            center_price = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

            # Calculate strike range around center price
            min_strike = center_price - self.strike_range
            max_strike = center_price + self.strike_range

            # Filter data based on strike range
            calls_filtered = calls_filtered[(calls_filtered['strike'] >= min_strike) &
                                           (calls_filtered['strike'] <= max_strike)]
            puts_filtered = puts_filtered[(puts_filtered['strike'] >= min_strike) &
                                         (puts_filtered['strike'] <= max_strike)]

            # Special handling for activity map
            if exposure_type == "ACTIVITY_MAP":
                return self.update_activity_map(calls_filtered, puts_filtered)

            # Store filtered data for crosshair lookup
            self.filtered_calls = calls_filtered.copy()
            self.filtered_puts = puts_filtered.copy()

            # Calculate net exposure
            if not calls_filtered.empty or not puts_filtered.empty:
                # Create a Series with all strikes
                all_strikes = pd.Series(sorted(set(calls_filtered['strike']) |
                                              set(puts_filtered['strike'])))

                # Initialize net exposure with zeros
                net_exposure = pd.Series(0, index=all_strikes)

                # For OI, PCR, and IVSKEW, use specific columns instead of exposure_type
                if exposure_type == 'OI':
                    y_column = 'openInterest'
                elif exposure_type == 'PCR':
                    y_column = 'PCR'
                elif exposure_type == 'IVSKEW':
                    y_column = 'IVSKEW'
                else:
                    y_column = exposure_type

                # Add call and put exposures
                if not calls_filtered.empty:
                    call_exposure = calls_filtered.groupby('strike')[y_column].sum()
                    net_exposure = net_exposure.add(call_exposure, fill_value=0)

                if not puts_filtered.empty:
                    # For OI, we subtract put OI from call OI to get net OI
                    # For PCR, we don't need to calculate net exposure (it's already a ratio)
                    # For other exposure types, we add them as usual
                    put_exposure = puts_filtered.groupby('strike')[y_column].sum()
                    if exposure_type == 'OI':
                        # For OI, we want to show the difference between calls and puts
                        net_exposure = net_exposure.subtract(put_exposure, fill_value=0)
                    elif exposure_type == 'PCR' or exposure_type == 'IVSKEW':
                        # For PCR and IVSKEW, we don't need to calculate net exposure (they're already ratios/differences)
                        # Just use the values from calls (they're the same as puts for these metrics)
                        pass
                    else:
                        net_exposure = net_exposure.add(put_exposure, fill_value=0)
            else:
                net_exposure = pd.Series()

            # Calculate total exposure values
            if exposure_type == 'OI':
                total_call_value = calls_filtered['openInterest'].sum() if not calls_filtered.empty else 0
                total_put_value = puts_filtered['openInterest'].sum() if not puts_filtered.empty else 0
            elif exposure_type == 'PCR':
                # For PCR, calculate average ratio instead of sum
                total_call_value = calls_filtered['PCR'].mean() if not calls_filtered.empty else 0
                total_put_value = puts_filtered['PCR'].mean() if not puts_filtered.empty else 0
            elif exposure_type == 'IVSKEW':
                # For IVSKEW, calculate average skew instead of sum
                total_call_value = calls_filtered['IVSKEW'].mean() if not calls_filtered.empty else 0
                total_put_value = puts_filtered['IVSKEW'].mean() if not puts_filtered.empty else 0
            else:
                total_call_value = calls_filtered[exposure_type].sum() if not calls_filtered.empty else 0
                total_put_value = puts_filtered[exposure_type].sum() if not puts_filtered.empty else 0

            # Plot based on chart type and display options
            if self.show_calls_checkbox.isChecked() and not calls_filtered.empty:
                # For OI, PCR, and IVSKEW, use specific columns instead of exposure_type
                if exposure_type == 'OI':
                    y_column = 'openInterest'
                elif exposure_type == 'PCR':
                    y_column = 'PCR'
                elif exposure_type == 'IVSKEW':
                    y_column = 'IVSKEW'
                else:
                    y_column = exposure_type

                if chart_type == 'Bar':
                    self.canvas.axes.bar(calls_filtered['strike'], calls_filtered[y_column],
                                        color=self.call_color, alpha=0.7, label='Call')
                elif chart_type == 'Line':
                    self.canvas.axes.plot(calls_filtered['strike'], calls_filtered[y_column],
                                         color=self.call_color, label='Call')
                elif chart_type == 'Scatter':
                    self.canvas.axes.scatter(calls_filtered['strike'], calls_filtered[y_column],
                                            color=self.call_color, label='Call')
                elif chart_type == 'Area':
                    self.canvas.axes.fill_between(calls_filtered['strike'], calls_filtered[y_column],
                                                 color=self.call_color, alpha=0.5, label='Call')

            if self.show_puts_checkbox.isChecked() and not puts_filtered.empty:
                # For OI, PCR, and IVSKEW, use specific columns instead of exposure_type
                if exposure_type == 'OI':
                    y_column = 'openInterest'
                elif exposure_type == 'PCR':
                    y_column = 'PCR'
                elif exposure_type == 'IVSKEW':
                    y_column = 'IVSKEW'
                else:
                    y_column = exposure_type

                if chart_type == 'Bar':
                    self.canvas.axes.bar(puts_filtered['strike'], puts_filtered[y_column],
                                        color=self.put_color, alpha=0.7, label='Put')
                elif chart_type == 'Line':
                    self.canvas.axes.plot(puts_filtered['strike'], puts_filtered[y_column],
                                         color=self.put_color, label='Put')
                elif chart_type == 'Scatter':
                    self.canvas.axes.scatter(puts_filtered['strike'], puts_filtered[y_column],
                                            color=self.put_color, label='Put')
                elif chart_type == 'Area':
                    self.canvas.axes.fill_between(puts_filtered['strike'], puts_filtered[y_column],
                                                 color=self.put_color, alpha=0.5, label='Put')

            # Add Net if enabled
            if self.show_net_checkbox.isChecked() and not net_exposure.empty:
                # For OI, PCR, and IVSKEW, we use different labels
                if exposure_type == 'OI':
                    net_positive_label = 'Net OI (Call > Put)'
                    net_negative_label = 'Net OI (Put > Call)'
                elif exposure_type == 'PCR':
                    net_positive_label = 'High P/C Ratio (Bearish)'
                    net_negative_label = 'Low P/C Ratio (Bullish)'
                elif exposure_type == 'IVSKEW':
                    net_positive_label = 'Positive Skew (Call IV > Put IV)'
                    net_negative_label = 'Negative Skew (Put IV > Call IV)'
                else:
                    net_positive_label = 'Net (Positive)'
                    net_negative_label = 'Net (Negative)'

                if chart_type == 'Bar':
                    # Use different colors for positive and negative values
                    positive_mask = net_exposure.values >= 0
                    negative_mask = ~positive_mask

                    if any(positive_mask):
                        self.canvas.axes.bar(net_exposure.index[positive_mask],
                                            net_exposure.values[positive_mask],
                                            color=self.call_color, alpha=0.7, label=net_positive_label)

                    if any(negative_mask):
                        self.canvas.axes.bar(net_exposure.index[negative_mask],
                                            net_exposure.values[negative_mask],
                                            color=self.put_color, alpha=0.7, label=net_negative_label)
                elif chart_type in ['Line', 'Scatter']:
                    # Split into positive and negative series
                    positive_mask = net_exposure.values >= 0
                    negative_mask = ~positive_mask

                    if any(positive_mask):
                        if chart_type == 'Line':
                            self.canvas.axes.plot(net_exposure.index[positive_mask],
                                                 net_exposure.values[positive_mask],
                                                 color=self.call_color, label=net_positive_label)
                        else:  # Scatter
                            self.canvas.axes.scatter(net_exposure.index[positive_mask],
                                                    net_exposure.values[positive_mask],
                                                    color=self.call_color, label=net_positive_label)

                    if any(negative_mask):
                        if chart_type == 'Line':
                            self.canvas.axes.plot(net_exposure.index[negative_mask],
                                                 net_exposure.values[negative_mask],
                                                 color=self.put_color, label=net_negative_label)
                        else:  # Scatter
                            self.canvas.axes.scatter(net_exposure.index[negative_mask],
                                                    net_exposure.values[negative_mask],
                                                    color=self.put_color, label=net_negative_label)
                elif chart_type == 'Area':
                    # Split into positive and negative series
                    positive_mask = net_exposure.values >= 0
                    negative_mask = ~positive_mask

                    if any(positive_mask):
                        self.canvas.axes.fill_between(net_exposure.index[positive_mask],
                                                     net_exposure.values[positive_mask],
                                                     color=self.call_color, alpha=0.5,
                                                     label=net_positive_label)

                    if any(negative_mask):
                        self.canvas.axes.fill_between(net_exposure.index[negative_mask],
                                                     net_exposure.values[negative_mask],
                                                     color=self.put_color, alpha=0.5,
                                                     label=net_negative_label)

            # Add current price line and backtested price line if different
            if self.current_price is not None:
                backtested_price = self.get_backtested_price()

                # Always show the current price line
                self.canvas.axes.axvline(x=self.current_price, color='white', linestyle='--',
                                        alpha=0.7, label=f'Current Price: {self.current_price}')

                # Show backtested price line if it's different from current price
                if backtested_price is not None and abs(backtested_price - self.current_price) > 0.01:
                    self.canvas.axes.axvline(x=backtested_price, color='yellow', linestyle='-.',
                                            alpha=0.9, linewidth=1.5,
                                            label=f'Backtested Price: {backtested_price:.2f} ({self.backtesting_slider.value():+d}%)')

            # Add max pain line for MPX chart
            if self.exposure_type == "MPX" and hasattr(self, 'max_pain_strike'):
                self.canvas.axes.axvline(x=self.max_pain_strike, color='yellow', linestyle='-.',
                                        alpha=0.9, linewidth=1.5, label=f'Max Pain: {self.max_pain_strike}')

            # Add volatility trigger line if enabled
            if self.show_vol_trigger_checkbox.isChecked() and hasattr(self, 'volatility_trigger') and self.volatility_trigger is not None:
                # Use a distinctive purple color for the volatility trigger
                self.canvas.axes.axvline(x=self.volatility_trigger, color='#9370DB', linestyle='-',
                                        alpha=0.9, linewidth=2.0, label=f'Vol Trigger: {self.volatility_trigger:.2f}')

                # Add text annotation above the volatility trigger
                _, y_max = self.canvas.axes.get_ylim()
                text_y = y_max * 0.95  # Position text at 95% of the y-axis height
                self.canvas.axes.text(self.volatility_trigger, text_y, f'Vol Trigger: {self.volatility_trigger:.2f}',
                                    color='#9370DB', ha='center', va='bottom', fontweight='bold',
                                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

            # Add gamma flip lines for GEX chart
            if self.exposure_type == "GEX" and self.show_net_checkbox.isChecked() and not net_exposure.empty:
                # Find where net exposure crosses zero (gamma flip points)
                gamma_flip_points = []
                strikes = sorted(net_exposure.index)

                for i in range(len(strikes) - 1):
                    current_strike = strikes[i]
                    next_strike = strikes[i + 1]
                    current_value = net_exposure[current_strike]
                    next_value = net_exposure[next_strike]

                    # Check if there's a sign change (crossing zero)
                    if (current_value * next_value <= 0) and (current_value != 0 or next_value != 0):
                        # Linear interpolation to find the exact zero-crossing point
                        if current_value == next_value or abs(next_value - current_value) < 1e-10:  # Avoid division by zero
                            flip_point = (current_strike + next_strike) / 2
                        else:
                            try:
                                # Calculate the zero-crossing point using linear interpolation
                                t = -current_value / (next_value - current_value)
                                flip_point = current_strike + t * (next_strike - current_strike)
                            except ZeroDivisionError:
                                # Fallback if division by zero occurs
                                flip_point = (current_strike + next_strike) / 2

                        gamma_flip_points.append(flip_point)

                # Add vertical lines at gamma flip points
                for i, flip_point in enumerate(gamma_flip_points):
                    # Only add label for the first gamma flip point to avoid duplicate legend entries
                    if i == 0:
                        self.canvas.axes.axvline(x=flip_point, color='#5cb8b2', linestyle='-',
                                            alpha=0.9, linewidth=1.5, label=f'Gamma Flip')
                    else:
                        self.canvas.axes.axvline(x=flip_point, color='#5cb8b2', linestyle='-',
                                            alpha=0.9, linewidth=1.5)

                    # Add text annotation above the flip point
                    _, y_max = self.canvas.axes.get_ylim()
                    text_y = y_max * 0.9  # Position text at 90% of the y-axis height
                    self.canvas.axes.text(flip_point, text_y, f'{flip_point:.2f}',
                                        color='#5cb8b2', ha='center', va='bottom',
                                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))

                # Add volatility trigger line if enabled
                if self.show_vol_trigger_checkbox.isChecked() and hasattr(self, 'volatility_trigger') and self.volatility_trigger is not None:
                    # Use a distinctive purple color for the volatility trigger
                    self.canvas.axes.axvline(x=self.volatility_trigger, color='#9370DB', linestyle='-',
                                            alpha=0.9, linewidth=2.0, label=f'Vol Trigger')

                    # Add text annotation above the volatility trigger
                    _, y_max = self.canvas.axes.get_ylim()
                    text_y = y_max * 0.95  # Position text at 95% of the y-axis height
                    self.canvas.axes.text(self.volatility_trigger, text_y, f'Vol Trigger: {self.volatility_trigger:.2f}',
                                        color='#9370DB', ha='center', va='bottom', fontweight='bold',
                                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

            # Highlight the 5 highest strikes on all exposure charts
            if self.exposure_type in ["GEX", "VEX", "DEX", "CEX", "TEX", "VEGX", "VOMX", "OI"] and not net_exposure.empty:
                # Determine which data to use for finding highest strikes
                if self.show_net_checkbox.isChecked():
                    # Use absolute values of net exposure to find highest strikes
                    abs_exposure = net_exposure.abs()
                    top_strikes = abs_exposure.nlargest(5).index.tolist()
                else:
                    # If net is not shown, combine call and put data
                    combined_data = pd.DataFrame()

                    # For OI, use specific column
                    if self.exposure_type == 'OI':
                        y_column = 'openInterest'
                    else:
                        y_column = self.exposure_type

                    # Add call data if shown
                    if self.show_calls_checkbox.isChecked() and not calls_filtered.empty:
                        call_data = calls_filtered.groupby('strike')[y_column].sum().abs()
                        combined_data = pd.concat([combined_data, call_data])

                    # Add put data if shown
                    if self.show_puts_checkbox.isChecked() and not puts_filtered.empty:
                        put_data = puts_filtered.groupby('strike')[y_column].sum().abs()
                        combined_data = pd.concat([combined_data, put_data])

                    # Group by strike and sum the absolute values
                    if not combined_data.empty:
                        combined_data = combined_data.groupby(level=0).sum()
                        # Check if combined_data is a DataFrame or Series
                        if isinstance(combined_data, pd.DataFrame):
                            # If it's a DataFrame, we need to specify the column
                            if y_column in combined_data.columns:
                                top_strikes = combined_data.nlargest(5, y_column).index.tolist()
                            else:
                                # If the column doesn't exist, use the first column
                                top_strikes = combined_data.nlargest(5, combined_data.columns[0]).index.tolist()
                        else:
                            # If it's a Series, we don't need to specify a column
                            top_strikes = combined_data.nlargest(5).index.tolist()
                    else:
                        top_strikes = []

                # Add markers for top strikes
                for i, strike in enumerate(top_strikes):
                    # Find the corresponding y-value
                    if self.show_net_checkbox.isChecked():
                        y_value = net_exposure[strike]
                    elif self.show_calls_checkbox.isChecked() and strike in calls_filtered['strike'].values:
                        if self.exposure_type == 'OI':
                            y_value = calls_filtered[calls_filtered['strike'] == strike]['openInterest'].iloc[0]
                        else:
                            y_value = calls_filtered[calls_filtered['strike'] == strike][self.exposure_type].iloc[0]
                    elif self.show_puts_checkbox.isChecked() and strike in puts_filtered['strike'].values:
                        if self.exposure_type == 'OI':
                            y_value = puts_filtered[puts_filtered['strike'] == strike]['openInterest'].iloc[0]
                        else:
                            y_value = puts_filtered[puts_filtered['strike'] == strike][self.exposure_type].iloc[0]
                    else:
                        continue

                    # Add a star marker at the strike
                    if i == 0:  # Only add label for the first marker to avoid duplicate legend entries
                        self.canvas.axes.plot(strike, y_value, 'y*', markersize=15, alpha=0.8, label='Top 5 Strikes')
                    else:
                        self.canvas.axes.plot(strike, y_value, 'y*', markersize=15, alpha=0.8)

                    # Add text annotation for the strike value
                    self.canvas.axes.text(strike, y_value, f'{strike:.2f}',
                                        color='yellow', ha='center', va='bottom', fontsize=9,
                                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))

            # Set chart title and labels
            ticker = self.ticker_input.text().strip().upper()
            expiry = self.expiry_selector.currentText()

            # Add backtesting indicator to title if backtesting
            backtesting_text = ""
            if self.backtesting_slider.value() != 0:
                backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

            # Set title based on exposure type
            if self.exposure_type == "VEX":
                title = f"Vanna Exposure - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call VEX: {total_call_value:.0f} | Put VEX: {total_put_value:.0f}"
                y_label = "Vanna Exposure"
            elif self.exposure_type == "GEX":
                title = f"Gamma Exposure - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call GEX: {total_call_value:.0f} | Put GEX: {total_put_value:.0f}"
                y_label = "Gamma Exposure"
            elif self.exposure_type == "DEX":
                title = f"Delta Exposure - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call DEX: {total_call_value:.0f} | Put DEX: {total_put_value:.0f}"
                y_label = "Delta Exposure"
            elif self.exposure_type == "CEX":
                title = f"Charm Exposure - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call CEX: {total_call_value:.0f} | Put CEX: {total_put_value:.0f}"
                y_label = "Charm Exposure (Delta Decay)"
            elif self.exposure_type == "TEX":
                title = f"Theta Exposure - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call TEX: {total_call_value:.0f} | Put TEX: {total_put_value:.0f}"
                y_label = "Theta Exposure (Time Decay)"
            elif self.exposure_type == "VEGX":
                title = f"Vega Exposure - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call VEGX: {total_call_value:.0f} | Put VEGX: {total_put_value:.0f}"
                y_label = "Vega Exposure (Volatility Sensitivity)"
            elif self.exposure_type == "VOMX":
                title = f"Vomma Exposure - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call VOMX: {total_call_value:.0f} | Put VOMX: {total_put_value:.0f}"
                y_label = "Vomma Exposure (Vega Convexity)"
            elif self.exposure_type == "OI":
                title = f"Open Interest - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call OI: {self.calls_df['openInterest'].sum():.0f} | Put OI: {self.puts_df['openInterest'].sum():.0f}"
                y_label = "Open Interest"
            elif self.exposure_type == "PCR":
                title = f"Put/Call Ratio by Strike - {ticker} (Exp: {expiry}){backtesting_text}"
                avg_pcr = total_call_value  # We stored the average PCR in total_call_value
                # Determine sentiment based on PCR value
                if avg_pcr > 1.5:
                    sentiment = "Bearish (high put/call ratio)"
                elif avg_pcr < 0.7:
                    sentiment = "Bullish (low put/call ratio)"
                else:
                    sentiment = "Neutral"
                subtitle = f"Average P/C Ratio: {avg_pcr:.2f} - {sentiment}"
                y_label = "Put/Call Ratio"
            elif self.exposure_type == "IVSKEW":
                title = f"IV Skew (Call IV - Put IV) - {ticker} (Exp: {expiry}){backtesting_text}"
                avg_skew = total_call_value  # We stored the average IV skew in total_call_value
                # Determine sentiment based on IV skew value
                if avg_skew > 0.02:  # More than 2% difference
                    sentiment = "Call IV Premium"
                elif avg_skew < -0.02:  # Less than -2% difference
                    sentiment = "Put IV Premium"
                else:
                    sentiment = "Neutral"
                subtitle = f"Average IV Skew: {avg_skew*100:.2f}% - {sentiment}"
                y_label = "IV Skew (Call IV - Put IV)"
            elif self.exposure_type == "ACTIVITY_MAP":
                title = f"Options Chain Activity Map - {ticker} (Exp: {expiry}){backtesting_text}"
                subtitle = f"Call Volume: {self.calls_df['volume'].sum():.0f} | Put Volume: {self.puts_df['volume'].sum():.0f}"
                y_label = "Activity Level"
            else:  # MPX
                max_pain_text = f", Max Pain: {self.max_pain_strike:.2f}" if hasattr(self, 'max_pain_strike') else ""
                title = f"Max Pain - {ticker} (Exp: {expiry}){max_pain_text}{backtesting_text}"
                subtitle = f"Call MPX: {total_call_value:.0f} | Put MPX: {total_put_value:.0f}"
                y_label = "Option Writer Pain"

            self.canvas.axes.set_title(f"{title}\n{subtitle}")
            self.canvas.axes.set_xlabel('Strike Price')
            self.canvas.axes.set_ylabel(y_label)

            # Set x-axis range with padding
            padding = self.strike_range * 0.1
            self.canvas.axes.set_xlim(min_strike - padding, max_strike + padding)

            # Add legend
            self.canvas.axes.legend()

            # Set dark background
            self.canvas.fig.patch.set_facecolor('#121212')
            self.canvas.axes.set_facecolor('#1e1e1e')
            self.canvas.axes.tick_params(colors='white')
            self.canvas.axes.xaxis.label.set_color('white')
            self.canvas.axes.yaxis.label.set_color('white')
            self.canvas.axes.title.set_color('white')

            # Draw the canvas
            self.canvas.draw()
        except ZeroDivisionError as e:
            print(f"Division by zero error in update_chart: {e}")
            QMessageBox.critical(self, "Error", "Error fetching data: division by zero. This usually happens with invalid or missing data.")
        except Exception as e:
            print(f"Error in update_chart: {e}")
            QMessageBox.critical(self, "Error", f"Error updating chart: {str(e)}")

    # Crosshair functionality removed

    def update_activity_map(self, calls_filtered, puts_filtered):
        """Create a heatmap visualization of options chain activity"""
        if self.calls_df.empty and self.puts_df.empty:
            return

        # Clear the current chart
        self.canvas.axes.clear()

        # Get all strikes in the range
        all_strikes = sorted(set(
            list(calls_filtered['strike']) +
            list(puts_filtered['strike'])
        ))

        # Create a matrix for the heatmap
        # Rows: Call ITM, Call ATM, Call OTM, Put OTM, Put ATM, Put ITM
        # Columns: Strike prices

        # Define moneyness categories
        categories = ['Call ITM', 'Call ATM', 'Call OTM', 'Put OTM', 'Put ATM', 'Put ITM']

        # Create empty matrix
        activity_matrix = np.zeros((len(categories), len(all_strikes)))

        # Current price for determining moneyness
        current_price = self.current_price

        # ATM threshold (consider strikes within 1% of current price as ATM)
        atm_threshold = current_price * 0.01

        # Get activity metric (volume or open interest)
        activity_metric = 'volume' if not hasattr(self, 'activity_metric_selector') or \
                          self.activity_metric_selector.currentText() == "Volume" else 'openInterest'

        # Process calls
        for i, strike in enumerate(all_strikes):
            # Find calls at this strike
            call_data = calls_filtered[calls_filtered['strike'] == strike]

            if not call_data.empty:
                # Determine moneyness
                if strike < current_price - atm_threshold:  # ITM
                    row_idx = 0  # Call ITM
                elif abs(strike - current_price) <= atm_threshold:  # ATM
                    row_idx = 1  # Call ATM
                else:  # OTM
                    row_idx = 2  # Call OTM

                # Use selected activity metric
                activity_matrix[row_idx, i] = call_data[activity_metric].sum()

        # Process puts
        for i, strike in enumerate(all_strikes):
            # Find puts at this strike
            put_data = puts_filtered[puts_filtered['strike'] == strike]

            if not put_data.empty:
                # Determine moneyness
                if strike > current_price + atm_threshold:  # ITM
                    row_idx = 5  # Put ITM
                elif abs(strike - current_price) <= atm_threshold:  # ATM
                    row_idx = 4  # Put ATM
                else:  # OTM
                    row_idx = 3  # Put OTM

                # Use selected activity metric
                activity_matrix[row_idx, i] = put_data[activity_metric].sum()

        # Normalize the data for better visualization
        # Add a small value to avoid division by zero
        max_activity = np.max(activity_matrix)
        if max_activity > 0:
            try:
                normalized_matrix = activity_matrix / max_activity
            except ZeroDivisionError:
                # If division by zero occurs, don't normalize
                normalized_matrix = activity_matrix
        else:
            normalized_matrix = activity_matrix

        # Get colormap from selector
        colormap = 'hot' if not hasattr(self, 'colormap_selector') else self.colormap_selector.currentText()

        # No need to clear colorbars here as we're recreating the figure in update_chart

        # Create the heatmap
        im = self.canvas.axes.imshow(
            normalized_matrix,
            aspect='auto',
            cmap=colormap,
            interpolation='nearest'
        )

        # Add colorbar and store reference to it
        self.canvas.colorbar = self.canvas.fig.colorbar(im, ax=self.canvas.axes)
        self.canvas.colorbar.set_label(f'Normalized Activity ({activity_metric.capitalize()})', color='white')
        self.canvas.colorbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(self.canvas.colorbar.ax.axes, 'yticklabels'), color='white')

        # Set labels
        self.canvas.axes.set_xticks(np.arange(len(all_strikes)))
        self.canvas.axes.set_xticklabels([f"{s:.1f}" for s in all_strikes], rotation=45, ha='right')
        self.canvas.axes.set_yticks(np.arange(len(categories)))
        self.canvas.axes.set_yticklabels(categories)

        # Add grid lines
        self.canvas.axes.set_xticks(np.arange(-.5, len(all_strikes), 1), minor=True)
        self.canvas.axes.set_yticks(np.arange(-.5, len(categories), 1), minor=True)
        self.canvas.axes.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.2)

        # Highlight current price
        if self.current_price is not None:
            # Find the closest strike to current price
            closest_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - self.current_price))
            self.canvas.axes.axvline(x=closest_idx, color='#5cb8b2', linestyle='--', alpha=0.7, linewidth=2)

        # Get annotation threshold
        try:
            threshold = float(self.threshold_input.text()) if hasattr(self, 'threshold_input') else 0.5
            if threshold < 0 or threshold > 1:
                threshold = 0.5
        except ValueError:
            threshold = 0.5

        # Add annotations for high activity cells
        for i in range(len(categories)):
            for j in range(len(all_strikes)):
                if normalized_matrix[i, j] > threshold:  # Only annotate cells with significant activity
                    text_color = 'black' if normalized_matrix[i, j] > 0.7 else 'white'
                    self.canvas.axes.text(j, i, f"{int(activity_matrix[i, j])}",
                                         ha="center", va="center", color=text_color, fontsize=8)

        # Set title
        ticker = self.ticker_input.text().strip().upper()
        expiry = self.expiry_selector.currentText()
        title = f"Options Chain Activity Map - {ticker} (Exp: {expiry})"
        subtitle = f"Call Volume: {self.calls_df['volume'].sum():.0f} | Put Volume: {self.puts_df['volume'].sum():.0f}"
        self.canvas.axes.set_title(f"{title}\n{subtitle}", color='white')

        # Adjust figure layout to prevent overlapping elements
        self.canvas.fig.subplots_adjust(left=0.1, right=0.85, bottom=0.15, top=0.9)

        # Set dark background
        self.canvas.fig.patch.set_facecolor('#121212')
        self.canvas.axes.set_facecolor('#1e1e1e')
        self.canvas.axes.tick_params(colors='white')
        self.canvas.axes.xaxis.label.set_color('white')
        self.canvas.axes.yaxis.label.set_color('white')
        self.canvas.axes.title.set_color('white')

        # Draw the canvas
        self.canvas.draw()

        # Return early since we've already updated the chart
        return True

    def update_delta_profile(self):
        """Create a visualization of delta values across different strikes"""
        if self.calls_df.empty and self.puts_df.empty:
            return

        # Clear the current chart
        self.canvas.axes.clear()

        # Calculate strike range around current price
        min_strike = self.current_price - self.strike_range
        max_strike = self.current_price + self.strike_range

        # Filter data based on strike range and non-zero delta values
        calls_filtered = self.calls_df[['strike', 'calc_delta', 'impliedVolatility']].copy()
        calls_filtered = calls_filtered[(calls_filtered['strike'] >= min_strike) &
                                       (calls_filtered['strike'] <= max_strike) &
                                       (calls_filtered['calc_delta'].notna())]

        puts_filtered = self.puts_df[['strike', 'calc_delta', 'impliedVolatility']].copy()
        puts_filtered = puts_filtered[(puts_filtered['strike'] >= min_strike) &
                                     (puts_filtered['strike'] <= max_strike) &
                                     (puts_filtered['calc_delta'].notna())]

        # Create strike price range for smooth curve
        strike_range = np.linspace(min_strike, max_strike, 100)

        # Get current date and calculate time to expiration
        today = datetime.today().date()
        selected_expiry = datetime.strptime(self.expiry_selector.currentText(), '%Y-%m-%d').date()
        t_days = max((selected_expiry - today).days, 1)  # Ensure at least 1 day
        t = t_days / 365.0

        # Risk-free rate
        r = 0.05  # Using 5% as a default

        # Get the price to center around (current or backtested)
        S = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

        # Calculate theoretical delta values for a range of strikes
        call_deltas = []
        put_deltas = []

        # Get average IV for calls and puts to use for theoretical lines
        avg_call_iv = calls_filtered['impliedVolatility'].mean() if not calls_filtered.empty else 0.3
        avg_put_iv = puts_filtered['impliedVolatility'].mean() if not puts_filtered.empty else 0.3

        # Use overall average IV if either is missing
        if np.isnan(avg_call_iv) or avg_call_iv == 0:
            avg_call_iv = 0.3
        if np.isnan(avg_put_iv) or avg_put_iv == 0:
            avg_put_iv = 0.3

        # Calculate theoretical delta values
        for K in strike_range:
            try:
                # Call delta
                call_delta = bs_delta('c', S, K, t, r, avg_call_iv)
                call_deltas.append(call_delta)

                # Put delta
                put_delta = bs_delta('p', S, K, t, r, avg_put_iv)
                put_deltas.append(put_delta)
            except Exception:
                # Use approximate delta if calculation fails
                call_delta = max(0, min(1, 1 - (K - S) / (S * 0.1)))
                put_delta = -max(0, min(1, (K - S) / (S * 0.1)))
                call_deltas.append(call_delta)
                put_deltas.append(put_delta)

        # Plot theoretical lines
        self.canvas.axes.plot(strike_range, call_deltas, color='green', linestyle='--',
                             alpha=0.7, linewidth=2, label='Call Delta (Theoretical)')
        self.canvas.axes.plot(strike_range, put_deltas, color='red', linestyle='--',
                             alpha=0.7, linewidth=2, label='Put Delta (Theoretical)')

        # Plot actual data points if available
        if not calls_filtered.empty:
            self.canvas.axes.scatter(calls_filtered['strike'], calls_filtered['calc_delta'],
                                   color='green', alpha=0.7, label='Call Delta (Market)')

        if not puts_filtered.empty:
            self.canvas.axes.scatter(puts_filtered['strike'], puts_filtered['calc_delta'],
                                   color='red', alpha=0.7, label='Put Delta (Market)')

        # Add current price line
        if self.current_price is not None:
            self.canvas.axes.axvline(x=self.current_price, color='white', linestyle='--',
                                    alpha=0.7, label=f'Current Price: {self.current_price}')

        # Add horizontal line at delta = 0
        self.canvas.axes.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Add horizontal lines at delta = 0.5 and delta = -0.5 (ATM approximate)
        self.canvas.axes.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        self.canvas.axes.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.3)

        # Set chart title and labels
        ticker = self.ticker_input.text().strip().upper()
        expiry = self.expiry_selector.currentText()
        days_to_expiry = (selected_expiry - today).days

        # Add backtesting indicator to title if backtesting
        backtesting_text = ""
        if self.backtesting_slider.value() != 0:
            backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

        title = f"Delta Profile - {ticker} (Exp: {expiry}, DTE: {days_to_expiry}){backtesting_text}"
        subtitle = f"Call IV: {avg_call_iv:.2f} | Put IV: {avg_put_iv:.2f}"

        self.canvas.axes.set_title(f"{title}\n{subtitle}")
        self.canvas.axes.set_xlabel('Strike Price')
        self.canvas.axes.set_ylabel('Delta')

        # Set y-axis range
        self.canvas.axes.set_ylim(-1.1, 1.1)

        # Set x-axis range with padding
        padding = self.strike_range * 0.1
        self.canvas.axes.set_xlim(min_strike - padding, max_strike + padding)

        # Add legend
        self.canvas.axes.legend()

        # Set dark background
        self.canvas.fig.patch.set_facecolor('#121212')
        self.canvas.axes.set_facecolor('#1e1e1e')
        self.canvas.axes.tick_params(colors='white')
        self.canvas.axes.xaxis.label.set_color('white')
        self.canvas.axes.yaxis.label.set_color('white')
        self.canvas.axes.title.set_color('white')

        # Draw the canvas
        self.canvas.draw()

        # Return early since we've already updated the chart
        return True

    def update_vega_profile(self):
        """Create a visualization of vega values across different strikes"""
        if self.calls_df.empty and self.puts_df.empty:
            return

        # Clear the current chart
        self.canvas.axes.clear()

        # Calculate strike range around current price
        min_strike = self.current_price - self.strike_range
        max_strike = self.current_price + self.strike_range

        # Filter data based on strike range and non-zero vega values
        calls_filtered = self.calls_df[['strike', 'calc_vega', 'impliedVolatility']].copy()
        calls_filtered = calls_filtered[(calls_filtered['strike'] >= min_strike) &
                                       (calls_filtered['strike'] <= max_strike) &
                                       (calls_filtered['calc_vega'].notna())]

        puts_filtered = self.puts_df[['strike', 'calc_vega', 'impliedVolatility']].copy()
        puts_filtered = puts_filtered[(puts_filtered['strike'] >= min_strike) &
                                     (puts_filtered['strike'] <= max_strike) &
                                     (puts_filtered['calc_vega'].notna())]

        # Create strike price range for smooth curve
        strike_range = np.linspace(min_strike, max_strike, 100)

        # Get current date and calculate time to expiration
        today = datetime.today().date()
        selected_expiry = datetime.strptime(self.expiry_selector.currentText(), '%Y-%m-%d').date()
        t_days = max((selected_expiry - today).days, 1)  # Ensure at least 1 day
        t = t_days / 365.0

        # Risk-free rate
        r = 0.05  # Using 5% as a default

        # Get the price to center around (current or backtested)
        S = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

        # Calculate theoretical vega values for a range of strikes
        call_vegas = []
        put_vegas = []

        # Get average IV for calls and puts to use for theoretical lines
        avg_call_iv = calls_filtered['impliedVolatility'].mean() if not calls_filtered.empty else 0.3
        avg_put_iv = puts_filtered['impliedVolatility'].mean() if not puts_filtered.empty else 0.3

        # Use overall average IV if either is missing
        if np.isnan(avg_call_iv) or avg_call_iv == 0:
            avg_call_iv = 0.3
        if np.isnan(avg_put_iv) or avg_put_iv == 0:
            avg_put_iv = 0.3

        # Calculate theoretical vega values
        for K in strike_range:
            try:
                # Call vega
                call_vega = bs_vega('c', S, K, t, r, avg_call_iv)
                call_vegas.append(call_vega)

                # Put vega
                put_vega = bs_vega('p', S, K, t, r, avg_put_iv)
                put_vegas.append(put_vega)
            except Exception:
                # Use approximate vega if calculation fails
                # Vega is highest at-the-money and decreases as strike moves away from current price
                approx_vega = max(0, 0.4 * (1 - abs(K - S) / (S * 0.2)))
                call_vegas.append(approx_vega)
                put_vegas.append(approx_vega)

        # Plot theoretical lines
        self.canvas.axes.plot(strike_range, call_vegas, color='green', linestyle='--',
                             alpha=0.7, linewidth=2, label='Call Vega (Theoretical)')
        self.canvas.axes.plot(strike_range, put_vegas, color='red', linestyle='--',
                             alpha=0.7, linewidth=2, label='Put Vega (Theoretical)')

        # Plot actual data points if available
        if not calls_filtered.empty:
            self.canvas.axes.scatter(calls_filtered['strike'], calls_filtered['calc_vega'],
                                   color='green', alpha=0.7, label='Call Vega (Market)')

        if not puts_filtered.empty:
            self.canvas.axes.scatter(puts_filtered['strike'], puts_filtered['calc_vega'],
                                   color='red', alpha=0.7, label='Put Vega (Market)')

        # Add current price line
        if self.current_price is not None:
            self.canvas.axes.axvline(x=self.current_price, color='white', linestyle='--',
                                    alpha=0.7, label=f'Current Price: {self.current_price}')

        # Add horizontal line at vega = 0
        self.canvas.axes.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Set chart title and labels
        ticker = self.ticker_input.text().strip().upper()
        expiry = self.expiry_selector.currentText()
        days_to_expiry = (selected_expiry - today).days

        # Add backtesting indicator to title if backtesting
        backtesting_text = ""
        if self.backtesting_slider.value() != 0:
            backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

        title = f"Vega Profile - {ticker} (Exp: {expiry}, DTE: {days_to_expiry}){backtesting_text}"
        subtitle = f"Call IV: {avg_call_iv:.2f} | Put IV: {avg_put_iv:.2f}"

        self.canvas.axes.set_title(f"{title}\n{subtitle}")
        self.canvas.axes.set_xlabel('Strike Price')
        self.canvas.axes.set_ylabel('Vega')

        # Set y-axis range with some padding
        if call_vegas and put_vegas:
            max_vega = max(max(call_vegas), max(put_vegas)) * 1.1
            self.canvas.axes.set_ylim(0, max_vega)

        # Set x-axis range with padding
        padding = self.strike_range * 0.1
        self.canvas.axes.set_xlim(min_strike - padding, max_strike + padding)

        # Add legend
        self.canvas.axes.legend()

        # Set dark background
        self.canvas.fig.patch.set_facecolor('#121212')
        self.canvas.axes.set_facecolor('#1e1e1e')
        self.canvas.axes.tick_params(colors='white')
        self.canvas.axes.xaxis.label.set_color('white')
        self.canvas.axes.yaxis.label.set_color('white')
        self.canvas.axes.title.set_color('white')

        # Draw the canvas
        self.canvas.draw()

        # Return early since we've already updated the chart
        return True

    def update_vomma_profile(self):
        """Create a visualization of vomma values across different strikes"""
        try:
            if self.calls_df.empty and self.puts_df.empty:
                return

            # Clear the current chart
            self.canvas.axes.clear()

            # Calculate strike range around current price
            min_strike = self.current_price - self.strike_range
            max_strike = self.current_price + self.strike_range

            # Filter data based on strike range and non-zero vomma values
            calls_filtered = self.calls_df[['strike', 'calc_vomma', 'impliedVolatility']].copy()
            calls_filtered = calls_filtered[(calls_filtered['strike'] >= min_strike) &
                                           (calls_filtered['strike'] <= max_strike) &
                                           (calls_filtered['calc_vomma'].notna())]

            puts_filtered = self.puts_df[['strike', 'calc_vomma', 'impliedVolatility']].copy()
            puts_filtered = puts_filtered[(puts_filtered['strike'] >= min_strike) &
                                         (puts_filtered['strike'] <= max_strike) &
                                         (puts_filtered['calc_vomma'].notna())]

            # Create strike price range for smooth curve
            strike_range = np.linspace(min_strike, max_strike, 100)

            # Get current date and calculate time to expiration
            today = datetime.today().date()
            selected_expiry = datetime.strptime(self.expiry_selector.currentText(), '%Y-%m-%d').date()
            t_days = max((selected_expiry - today).days, 1)  # Ensure at least 1 day
            t = t_days / 365.0

            # Risk-free rate
            r = 0.05  # Using 5% as a default

            # Get the price to center around (current or backtested)
            S = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

            # Calculate theoretical vomma values for a range of strikes
            call_vommas = []
            put_vommas = []

            # Get average IV for calls and puts to use for theoretical lines
            avg_call_iv = calls_filtered['impliedVolatility'].mean() if not calls_filtered.empty else 0.3
            avg_put_iv = puts_filtered['impliedVolatility'].mean() if not puts_filtered.empty else 0.3

            # Use overall average IV if either is missing
            if np.isnan(avg_call_iv) or avg_call_iv == 0:
                avg_call_iv = 0.3
            if np.isnan(avg_put_iv) or avg_put_iv == 0:
                avg_put_iv = 0.3

            # Calculate theoretical vomma values
            for K in strike_range:
                try:
                    # Calculate d1 and d2 for call
                    d1_call = (log(S / K) + (r + 0.5 * avg_call_iv**2) * t) / (avg_call_iv * sqrt(t))
                    d2_call = d1_call - avg_call_iv * sqrt(t)

                    # Calculate vomma for call
                    vega_call = S * sqrt(t) * norm.pdf(d1_call)
                    try:
                        vomma_call = vega_call * (d1_call*d2_call - 1) / avg_call_iv
                    except ZeroDivisionError:
                        # Handle division by zero
                        vomma_call = 0
                    call_vommas.append(vomma_call)

                    # Calculate d1 and d2 for put
                    d1_put = (log(S / K) + (r + 0.5 * avg_put_iv**2) * t) / (avg_put_iv * sqrt(t))
                    d2_put = d1_put - avg_put_iv * sqrt(t)

                    # Calculate vomma for put
                    vega_put = S * sqrt(t) * norm.pdf(d1_put)
                    try:
                        vomma_put = vega_put * (d1_put*d2_put - 1) / avg_put_iv
                    except ZeroDivisionError:
                        # Handle division by zero
                        vomma_put = 0
                    put_vommas.append(vomma_put)
                except Exception as e:
                    # Use approximate vomma if calculation fails
                    print(f"Error calculating theoretical vomma: {e}")
                    # Vomma is highest for OTM options and increases with IV
                    moneyness = abs(K - S) / S
                    approx_vomma = max(0, 0.2 * (1 + moneyness * 5))
                    call_vommas.append(approx_vomma)
                    put_vommas.append(approx_vomma)

            # Plot theoretical lines
            self.canvas.axes.plot(strike_range, call_vommas, color='green', linestyle='--',
                                 alpha=0.7, linewidth=2, label='Call Vomma (Theoretical)')
            self.canvas.axes.plot(strike_range, put_vommas, color='red', linestyle='--',
                                 alpha=0.7, linewidth=2, label='Put Vomma (Theoretical)')

            # Plot actual data points if available
            if not calls_filtered.empty:
                self.canvas.axes.scatter(calls_filtered['strike'], calls_filtered['calc_vomma'],
                                       color='green', alpha=0.7, label='Call Vomma (Market)')

            if not puts_filtered.empty:
                self.canvas.axes.scatter(puts_filtered['strike'], puts_filtered['calc_vomma'],
                                       color='red', alpha=0.7, label='Put Vomma (Market)')

            # Add current price line
            if self.current_price is not None:
                self.canvas.axes.axvline(x=self.current_price, color='white', linestyle='--',
                                        alpha=0.7, label=f'Current Price: {self.current_price}')

            # Add horizontal line at vomma = 0
            self.canvas.axes.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

            # Set chart title and labels
            ticker = self.ticker_input.text().strip().upper()
            expiry = self.expiry_selector.currentText()
            days_to_expiry = (selected_expiry - today).days

            # Add backtesting indicator to title if backtesting
            backtesting_text = ""
            if self.backtesting_slider.value() != 0:
                backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

            title = f"Vomma Profile - {ticker} (Exp: {expiry}, DTE: {days_to_expiry}){backtesting_text}"
            subtitle = f"Call IV: {avg_call_iv:.2f} | Put IV: {avg_put_iv:.2f}"

            self.canvas.axes.set_title(f"{title}\n{subtitle}")
            self.canvas.axes.set_xlabel('Strike Price')
            self.canvas.axes.set_ylabel('Vomma (Vega Convexity)')

            # Set y-axis range with some padding
            if call_vommas and put_vommas:
                max_vomma = max(max(call_vommas), max(put_vommas)) * 1.1
                min_vomma = min(min(call_vommas), min(put_vommas)) * 1.1
                self.canvas.axes.set_ylim(min_vomma, max_vomma)

            # Set x-axis range with padding
            padding = self.strike_range * 0.1
            self.canvas.axes.set_xlim(min_strike - padding, max_strike + padding)

            # Add legend
            self.canvas.axes.legend()

            # Set dark background
            self.canvas.fig.patch.set_facecolor('#121212')
            self.canvas.axes.set_facecolor('#1e1e1e')
            self.canvas.axes.tick_params(colors='white')
            self.canvas.axes.xaxis.label.set_color('white')
            self.canvas.axes.yaxis.label.set_color('white')
            self.canvas.axes.title.set_color('white')

            # Draw the canvas
            self.canvas.draw()

            # Return early since we've already updated the chart
            return True
        except ZeroDivisionError as e:
            print(f"Division by zero error in update_vomma_profile: {e}")
            QMessageBox.critical(self, "Error", "Error fetching data: division by zero. This usually happens with invalid or missing data.")
            return False
        except Exception as e:
            print(f"Error in update_vomma_profile: {e}")
            QMessageBox.critical(self, "Error", f"Error updating vomma profile: {str(e)}")
            return False

    def update_greek_landscape(self):
        """Create a visualization of Greek values across different strikes and time to expiration"""
        if self.current_price is None:
            return

        try:
            # Completely recreate the 3D axes to avoid colorbar issues
            # First, clear the figure
            self.canvas_3d.fig.clear()

            # Create a new 3D axes
            self.canvas_3d.axes = self.canvas_3d.fig.add_subplot(111, projection='3d')

            # Reset the colorbar reference
            self.canvas_3d.colorbar = None

            # Get current date and calculate time to expiration
            today = datetime.today().date()
            selected_expiry = datetime.strptime(self.expiry_selector.currentText(), '%Y-%m-%d').date()
            # We'll use the actual days to expiration for the title
            days_to_expiry = max((selected_expiry - today).days, 1)  # Ensure at least 1 day

            # Risk-free rate
            r = 0.05  # Using 5% as a default

            # Get the price to center around (current or backtested)
            S = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

            # Create strike price range around current price
            strike_range = self.strike_range
            min_strike = S - strike_range
            max_strike = S + strike_range

            # Get grid points from input
            try:
                grid_points = int(self.grid_points_input_greek.text())
                if grid_points < 10:
                    grid_points = 10  # Minimum grid size
                elif grid_points > 100:
                    grid_points = 100  # Maximum grid size for performance
            except ValueError:
                grid_points = 50  # Default if invalid input

            strikes = np.linspace(min_strike, max_strike, grid_points)

            # Get time range from inputs
            try:
                min_days = int(self.min_time_input.text())
                if min_days < 1:
                    min_days = 1  # Minimum 1 day
            except ValueError:
                min_days = 1  # Default if invalid input

            try:
                max_days = int(self.max_time_input.text())
                if max_days > 365:
                    max_days = 365  # Maximum 1 year
                if max_days <= min_days:
                    max_days = min_days + 7  # Ensure max > min by at least a week
            except ValueError:
                max_days = 30  # Default if invalid input

            days_range = np.linspace(min_days, max_days, grid_points)

            # Create meshgrid for 3D surface
            K, T = np.meshgrid(strikes, days_range)

            # Get average IV to use for theoretical calculations
            avg_iv = 0.3  # Default
            if not self.calls_df.empty and not self.puts_df.empty:
                calls_iv = self.calls_df['impliedVolatility'].mean()
                puts_iv = self.puts_df['impliedVolatility'].mean()
                if not np.isnan(calls_iv) and not np.isnan(puts_iv):
                    avg_iv = (calls_iv + puts_iv) / 2

            # Get selected Greek type
            greek_type = self.greek_type_selector.currentText()

            # Get option type
            option_type = 'c' if self.option_type_selector_greek.currentText() == "Call" else 'p'

            # Calculate Greek values for the grid
            greek_values = np.zeros_like(K)

            for i in range(len(days_range)):
                for j in range(len(strikes)):
                    try:
                        # Convert days to years for BS calculation
                        t = days_range[i] / 365.0

                        # Calculate the selected Greek using py_vollib
                        if greek_type == "Delta":
                            greek_val = bs_delta(option_type, S, K[i, j], t, r, avg_iv)
                        elif greek_type == "Gamma":
                            greek_val = bs_gamma(option_type, S, K[i, j], t, r, avg_iv)
                        elif greek_type == "Vega":
                            greek_val = bs_vega(option_type, S, K[i, j], t, r, avg_iv)
                        elif greek_type == "Theta":
                            greek_val = bs_theta(option_type, S, K[i, j], t, r, avg_iv)
                            # Convert to daily theta
                            greek_val = greek_val / 365.0
                        elif greek_type == "Vomma":
                            # Calculate d1 and d2
                            d1 = (log(S / K[i, j]) + (r + 0.5 * avg_iv**2) * t) / (avg_iv * sqrt(t))
                            d2 = d1 - avg_iv * sqrt(t)

                            # Calculate vomma
                            vega_val = S * sqrt(t) * norm.pdf(d1)
                            try:
                                greek_val = vega_val * (d1*d2 - 1) / avg_iv
                            except ZeroDivisionError:
                                # Handle division by zero
                                greek_val = 0
                        else:
                            greek_val = 0

                        greek_values[i, j] = greek_val
                    except Exception as e:
                        print(f"Error calculating {greek_type}: {e}")
                        greek_values[i, j] = 0

            # Create the 3D surface plot
            surf = self.canvas_3d.axes.plot_surface(
                K, T, greek_values,
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=True,
                alpha=0.8
            )

            # Add a new color bar
            try:
                # First, adjust the figure layout to make room for the colorbar
                self.canvas_3d.fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)  # Leave space on the right for colorbar

                # Create the colorbar in the reserved space
                cax = self.canvas_3d.fig.add_axes([0.85, 0.2, 0.03, 0.6])  # [left, bottom, width, height]
                self.canvas_3d.colorbar = self.canvas_3d.fig.colorbar(surf, cax=cax)
            except Exception as e:
                print(f"Error creating colorbar: {e}")
                self.canvas_3d.colorbar = None

            # Set labels and title
            ticker = self.ticker_input.text().strip().upper()
            expiry = self.expiry_selector.currentText()

            self.canvas_3d.axes.set_xlabel('Strike Price')
            self.canvas_3d.axes.set_ylabel('Days to Expiration')
            self.canvas_3d.axes.set_zlabel(greek_type)
            option_type_str = "Call" if option_type == 'c' else "Put"
            # Add backtesting indicator to title if backtesting
            backtesting_text = ""
            if self.backtesting_slider.value() != 0:
                backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

            self.canvas_3d.axes.set_title(f"{greek_type} Landscape ({option_type_str}) - {ticker} (Exp: {expiry}, DTE: {days_to_expiry}){backtesting_text}\nCurrent Price: {self.current_price:.2f}, Backtested Price: {S:.2f}" if self.backtesting_slider.value() != 0 else f"{greek_type} Landscape ({option_type_str}) - {ticker} (Exp: {expiry}, DTE: {days_to_expiry})\nCurrent Price: {S:.2f}")

            # Mark current price on the surface
            # Find the closest strike to current price
            closest_strike_idx = np.abs(strikes - S).argmin()
            # Add a line at current price through all days
            days_line = np.linspace(min_days, max_days, grid_points)
            strike_line = np.ones_like(days_line) * strikes[closest_strike_idx]
            greek_line = np.zeros_like(days_line)

            for i, days in enumerate(days_line):
                try:
                    # Convert days to years for BS calculation
                    t = days / 365.0

                    # Calculate the selected Greek using py_vollib
                    if greek_type == "Delta":
                        greek_line[i] = bs_delta(option_type, S, strikes[closest_strike_idx], t, r, avg_iv)
                    elif greek_type == "Gamma":
                        greek_line[i] = bs_gamma(option_type, S, strikes[closest_strike_idx], t, r, avg_iv)
                    elif greek_type == "Vega":
                        greek_line[i] = bs_vega(option_type, S, strikes[closest_strike_idx], t, r, avg_iv)
                    elif greek_type == "Theta":
                        greek_line[i] = bs_theta(option_type, S, strikes[closest_strike_idx], t, r, avg_iv) / 365.0
                    elif greek_type == "Vomma":
                        # Calculate d1 and d2
                        d1 = (log(S / strikes[closest_strike_idx]) + (r + 0.5 * avg_iv**2) * t) / (avg_iv * sqrt(t))
                        d2 = d1 - avg_iv * sqrt(t)

                        # Calculate vomma
                        vega_val = S * sqrt(t) * norm.pdf(d1)
                        try:
                            greek_line[i] = vega_val * (d1*d2 - 1) / avg_iv
                        except ZeroDivisionError:
                            # Handle division by zero
                            greek_line[i] = 0
                    else:
                        greek_line[i] = 0
                except Exception:
                    greek_line[i] = 0

            self.canvas_3d.axes.plot(strike_line, days_line, greek_line, 'r-', linewidth=2)

            # Set dark background
            self.canvas_3d.fig.patch.set_facecolor('#121212')
            self.canvas_3d.axes.set_facecolor('#1e1e1e')

            # Set text colors to white
            self.canvas_3d.axes.xaxis.label.set_color('white')
            self.canvas_3d.axes.yaxis.label.set_color('white')
            self.canvas_3d.axes.zaxis.label.set_color('white')
            self.canvas_3d.axes.title.set_color('white')

            # Set tick colors to white
            self.canvas_3d.axes.tick_params(axis='x', colors='white')
            self.canvas_3d.axes.tick_params(axis='y', colors='white')
            self.canvas_3d.axes.tick_params(axis='z', colors='white')

            # Set view angle based on selection
            view_angle = self.view_angle_selector_greek.currentText()
            if view_angle == "Top":
                self.canvas_3d.axes.view_init(elev=90, azim=-90)
            elif view_angle == "Side":
                self.canvas_3d.axes.view_init(elev=0, azim=0)
            elif view_angle == "Front":
                self.canvas_3d.axes.view_init(elev=0, azim=90)
            else:  # Default
                self.canvas_3d.axes.view_init(elev=30, azim=30)

            # Draw the canvas
            try:
                self.canvas_3d.draw()
            except Exception as e:
                print(f"Error drawing 3D canvas: {e}")

        except Exception as e:
            print(f"Error in update_greek_landscape: {e}")

        return True

    def update_iv_surface(self):
        """Create a visualization of implied volatility across different strikes and time to expiration"""
        if self.current_price is None:
            return

        try:
            # Completely recreate the 3D axes to avoid colorbar issues
            # First, clear the figure
            self.canvas_3d.fig.clear()

            # Create a new 3D axes
            self.canvas_3d.axes = self.canvas_3d.fig.add_subplot(111, projection='3d')

            # Reset the colorbar reference
            self.canvas_3d.colorbar = None

            # Get current date and calculate time to expiration
            today = datetime.today().date()
            selected_expiry = datetime.strptime(self.expiry_selector.currentText(), '%Y-%m-%d').date()
            # We'll use the actual days to expiration for the title
            days_to_expiry = max((selected_expiry - today).days, 1)  # Ensure at least 1 day

            # Get the price to center around (current or backtested)
            S = self.get_backtested_price() if self.backtesting_slider.value() != 0 else self.current_price

            # Create strike price range around current price
            strike_range = self.strike_range
            min_strike = S - strike_range
            max_strike = S + strike_range

            # Get grid points from input
            try:
                grid_points = int(self.grid_points_input_iv.text())
                if grid_points < 10:
                    grid_points = 10  # Minimum grid size
                elif grid_points > 100:
                    grid_points = 100  # Maximum grid size for performance
            except ValueError:
                grid_points = 50  # Default if invalid input

            strikes = np.linspace(min_strike, max_strike, grid_points)

            # Get time range from inputs
            try:
                min_days = int(self.min_time_input_iv.text())
                if min_days < 1:
                    min_days = 1  # Minimum 1 day
            except ValueError:
                min_days = 1  # Default if invalid input

            try:
                max_days = int(self.max_time_input_iv.text())
                if max_days > 365:
                    max_days = 365  # Maximum 1 year
                if max_days <= min_days:
                    max_days = min_days + 7  # Ensure max > min by at least a week
            except ValueError:
                max_days = 30  # Default if invalid input

            days_range = np.linspace(min_days, max_days, grid_points)

            # Create meshgrid for 3D surface
            K, T = np.meshgrid(strikes, days_range)

            # Get option type selection
            option_type = self.option_type_selector_iv.currentText()

            # Get colormap selection
            colormap = self.colormap_selector_iv.currentText()

            # Calculate IV values for the grid
            iv_values = np.zeros_like(K)

            # Note: We don't need risk-free rate for IV surface calculation
            # but we keep this comment for consistency with other methods

            # Get current options data for IV reference
            calls_filtered = self.calls_df.copy()
            puts_filtered = self.puts_df.copy()

            # Filter by strike range
            calls_filtered = calls_filtered[(calls_filtered['strike'] >= min_strike) &
                                          (calls_filtered['strike'] <= max_strike)]
            puts_filtered = puts_filtered[(puts_filtered['strike'] >= min_strike) &
                                        (puts_filtered['strike'] <= max_strike)]

            # Create dictionaries to store IV by strike
            call_iv_by_strike = {}
            put_iv_by_strike = {}

            # Aggregate IV by strike
            for _, row in calls_filtered.iterrows():
                strike = row['strike']
                iv = row.get('impliedVolatility', 0)
                if strike not in call_iv_by_strike:
                    call_iv_by_strike[strike] = []
                call_iv_by_strike[strike].append(iv)

            for _, row in puts_filtered.iterrows():
                strike = row['strike']
                iv = row.get('impliedVolatility', 0)
                if strike not in put_iv_by_strike:
                    put_iv_by_strike[strike] = []
                put_iv_by_strike[strike].append(iv)

            # Calculate average IV for each strike
            call_avg_iv = {}
            put_avg_iv = {}

            for strike, ivs in call_iv_by_strike.items():
                call_avg_iv[strike] = np.mean(ivs)

            for strike, ivs in put_iv_by_strike.items():
                put_avg_iv[strike] = np.mean(ivs)

            # Get all unique strikes from both calls and puts
            all_strikes = sorted(set(list(call_avg_iv.keys()) + list(put_avg_iv.keys())))

            # Create a function to estimate IV based on strike and days to expiration
            def estimate_iv(strike, days):
                # Find the closest available strike
                if not all_strikes:
                    return 0.3  # Default IV if no data

                closest_strike = min(all_strikes, key=lambda x: abs(x - strike))

                # Get IV based on option type
                if option_type == "Call":
                    iv = call_avg_iv.get(closest_strike, 0.3)
                elif option_type == "Put":
                    iv = put_avg_iv.get(closest_strike, 0.3)
                else:  # Average
                    call_iv = call_avg_iv.get(closest_strike, 0.3)
                    put_iv = put_avg_iv.get(closest_strike, 0.3)
                    iv = (call_iv + put_iv) / 2

                # Apply a time decay factor to IV (IV typically decreases as expiration approaches)
                # This is a simple model - in reality, the term structure of IV is more complex
                days_factor = np.sqrt(days / days_to_expiry) if days_to_expiry > 0 else 1

                # Apply a skew factor based on distance from ATM
                # IV typically forms a smile or skew pattern across strikes
                moneyness = abs(strike - S) / S
                skew_factor = 1 + 0.1 * moneyness  # Simple skew model

                return iv * days_factor * skew_factor

            # Calculate IV values for the grid
            for i in range(len(days_range)):
                for j in range(len(strikes)):
                    try:
                        iv_values[i, j] = estimate_iv(strikes[j], days_range[i])
                    except Exception as e:
                        print(f"Error calculating IV: {e}")
                        iv_values[i, j] = 0.3  # Default IV

            # Create the 3D surface plot
            surf = self.canvas_3d.axes.plot_surface(
                K, T, iv_values,
                cmap=colormap,
                linewidth=0,
                antialiased=True,
                alpha=0.8
            )

            # Add a new color bar
            try:
                # First, adjust the figure layout to make room for the colorbar
                self.canvas_3d.fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)  # Leave space on the right for colorbar

                # Create the colorbar in the reserved space
                cax = self.canvas_3d.fig.add_axes([0.85, 0.2, 0.03, 0.6])  # [left, bottom, width, height]
                self.canvas_3d.colorbar = self.canvas_3d.fig.colorbar(surf, cax=cax)
                self.canvas_3d.colorbar.set_label('Implied Volatility', color='white')
            except Exception as e:
                print(f"Error creating colorbar: {e}")
                self.canvas_3d.colorbar = None

            # Set labels and title
            ticker = self.ticker_input.text().strip().upper()
            expiry = self.expiry_selector.currentText()

            self.canvas_3d.axes.set_xlabel('Strike Price')
            self.canvas_3d.axes.set_ylabel('Days to Expiration')
            self.canvas_3d.axes.set_zlabel('Implied Volatility')
            # Add backtesting indicator to title if backtesting
            backtesting_text = ""
            if self.backtesting_slider.value() != 0:
                backtesting_text = f" [BACKTESTING: {self.backtesting_slider.value():+d}%]"

            self.canvas_3d.axes.set_title(f"IV Surface ({option_type}) - {ticker} (Exp: {expiry}, DTE: {days_to_expiry}){backtesting_text}\nCurrent Price: {self.current_price:.2f}, Backtested Price: {S:.2f}" if self.backtesting_slider.value() != 0 else f"IV Surface ({option_type}) - {ticker} (Exp: {expiry}, DTE: {days_to_expiry})\nCurrent Price: {S:.2f}")

            # Mark current price on the surface
            # Find the closest strike to current price
            closest_strike_idx = np.abs(strikes - S).argmin()
            # Add a line at current price through all days
            days_line = np.linspace(min_days, max_days, grid_points)
            strike_line = np.ones_like(days_line) * strikes[closest_strike_idx]
            iv_line = np.zeros_like(days_line)

            for i, days in enumerate(days_line):
                try:
                    iv_line[i] = estimate_iv(strikes[closest_strike_idx], days)
                except Exception:
                    iv_line[i] = 0.3  # Default IV

            self.canvas_3d.axes.plot(strike_line, days_line, iv_line, 'r-', linewidth=2)

            # Set dark background
            self.canvas_3d.fig.patch.set_facecolor('#121212')
            self.canvas_3d.axes.set_facecolor('#1e1e1e')

            # Set text colors to white
            self.canvas_3d.axes.xaxis.label.set_color('white')
            self.canvas_3d.axes.yaxis.label.set_color('white')
            self.canvas_3d.axes.zaxis.label.set_color('white')
            self.canvas_3d.axes.title.set_color('white')

            # Set tick colors to white
            self.canvas_3d.axes.tick_params(axis='x', colors='white')
            self.canvas_3d.axes.tick_params(axis='y', colors='white')
            self.canvas_3d.axes.tick_params(axis='z', colors='white')

            # Set view angle based on selection
            view_angle = self.view_angle_selector_iv.currentText()
            if view_angle == "Top":
                self.canvas_3d.axes.view_init(elev=90, azim=-90)
            elif view_angle == "Side":
                self.canvas_3d.axes.view_init(elev=0, azim=0)
            elif view_angle == "Front":
                self.canvas_3d.axes.view_init(elev=0, azim=90)
            else:  # Default
                self.canvas_3d.axes.view_init(elev=30, azim=30)

            # Draw the canvas
            try:
                self.canvas_3d.draw()
            except Exception as e:
                print(f"Error drawing 3D canvas: {e}")

        except Exception as e:
            print(f"Error in update_iv_surface: {e}")

        return True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a more modern look

    # Set dark theme
    dark_palette = app.palette()
    dark_palette.setColor(dark_palette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(dark_palette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(dark_palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(dark_palette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(dark_palette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(dark_palette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(dark_palette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(dark_palette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(dark_palette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(dark_palette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)

    window = OptionsExposureDashboard()
    window.show()
    sys.exit(app.exec())

        # Add Options Activity tab
    options_activity_tab = QWidget()
    options_activity_idx = window.exposure_tabs.addTab(options_activity_tab, "Options Activity")
    window.exposure_tabs.setTabToolTip(options_activity_idx, "Options Activity: Heatmap and charts for options volume, open interest, and implied volatility.")
    window.setup_options_activity_tab()
    
    def update_options_activity(self, calls_df, puts_df, current_price, max_pain):
        """Update the Options Activity tab with heatmap and charts."""
        # Generate the heatmap
        self.generate_heatmap(calls_df, puts_df, current_price, max_pain)
    
        # Generate the Open Interest by Strike chart
        self.generate_open_interest_chart(calls_df, puts_df)
    
        # Generate the Implied Volatility Smile chart
        self.generate_iv_smile_chart(calls_df, puts_df)

    def generate_heatmap(self, calls_df, puts_df, current_price, max_pain):
        """Generate the Options Volume Heat Map."""
        # Prepare data for the heatmap
        strikes = sorted(set(calls_df['strike']).union(puts_df['strike']))
        heatmap_data = np.zeros((6, len(strikes)))  # 6 rows for Calls ITM, ATM, OTM, Puts ITM, ATM, OTM

        for i, strike in enumerate(strikes):
            # Calls
            heatmap_data[0, i] = calls_df[(calls_df['strike'] == strike) & (calls_df['moneyness'] == 'ITM')]['volume'].sum()
            heatmap_data[1, i] = calls_df[(calls_df['strike'] == strike) & (calls_df['moneyness'] == 'ATM')]['volume'].sum()
            heatmap_data[2, i] = calls_df[(calls_df['strike'] == strike) & (calls_df['moneyness'] == 'OTM')]['volume'].sum()

            # Puts
            heatmap_data[3, i] = puts_df[(puts_df['strike'] == strike) & (puts_df['moneyness'] == 'ITM')]['volume'].sum()
            heatmap_data[4, i] = puts_df[(puts_df['strike'] == strike) & (puts_df['moneyness'] == 'ATM')]['volume'].sum()
            heatmap_data[5, i] = puts_df[(puts_df['strike'] == strike) & (puts_df['moneyness'] == 'OTM')]['volume'].sum()

        # Plot the heatmap
        ax = self.heatmap_canvas.axes
        ax.clear()
        heatmap = ax.imshow(heatmap_data, cmap="coolwarm", aspect="auto")

        # Add labels
        ax.set_xticks(range(len(strikes)))
        ax.set_xticklabels([f"${strike:.2f}" for strike in strikes], rotation=45, ha="right")
        ax.set_yticks(range(6))
        ax.set_yticklabels(["Call ITM", "Call ATM", "Call OTM", "Put ITM", "Put ATM", "Put OTM"])
        ax.set_title(f"Options Volume Heat Map\nCurrent Price: ${current_price:.2f}, Max Pain: ${max_pain:.2f}")

        # Add color bar
        cbar = self.heatmap_canvas.fig.colorbar(heatmap, ax=ax, orientation="vertical")
        cbar.set_label("Volume", fontsize=12)

        self.heatmap_canvas.draw()
