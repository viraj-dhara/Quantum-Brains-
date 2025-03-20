import pandas as pd
import numpy as np
import talib
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class CryptoTradingStrategy:
    def __init__(self, symbol='BTC/USDT', timeframe='1h', initial_capital=10000):
        """
        Initialize the trading strategy with parameters
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Candle timeframe
        initial_capital (float): Starting capital in USDT
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.position_price = 0
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        
    def load_data(self, file_path):
        """Load historical OHLCV data from CSV"""
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    def preprocess_data(self, df):
        """Calculate technical indicators and create features"""
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Calculate MACD
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Calculate RSI
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        
        # Calculate Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(
            data['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # Calculate ATR for volatility measurement
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Calculate Stochastic Oscillator
        data['slowk'], data['slowd'] = talib.STOCH(
            data['high'], data['low'], data['close'], 
            fastk_period=14, slowk_period=3, slowk_matype=0, 
            slowd_period=3, slowd_matype=0)
        
        # Calculate Exponential Moving Averages
        data['ema_9'] = talib.EMA(data['close'], timeperiod=9)
        data['ema_21'] = talib.EMA(data['close'], timeperiod=21)
        data['ema_50'] = talib.EMA(data['close'], timeperiod=50)
        data['ema_200'] = talib.EMA(data['close'], timeperiod=200)
        
        # Calculate price momentum
        data['mom_1d'] = data['close'].pct_change(1)
        data['mom_3d'] = data['close'].pct_change(3)
        data['mom_7d'] = data['close'].pct_change(7)
        data['mom_14d'] = data['close'].pct_change(14)
        
        # Calculate volume indicators
        data['volume_change'] = data['volume'].pct_change()
        data['volume_sma'] = talib.SMA(data['volume'], timeperiod=20)
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Calculate ADX (Average Directional Index)
        data['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Calculate OBV (On-Balance Volume)
        data['obv'] = talib.OBV(data['close'], data['volume'])
        
        # Market regime features (bull/bear)
        data['bull_market'] = (data['close'] > data['ema_200']).astype(int)
        data['price_above_ema50'] = (data['close'] > data['ema_50']).astype(int)
        
        # Advanced features
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['rsi_ma'] = talib.SMA(data['rsi'], timeperiod=5)
        data['macd_cross'] = ((data['macd'] > data['macd_signal']) & 
                             (data['macd'].shift(1) <= data['macd_signal'].shift(1))).astype(int)
        
        # Percent to upper/lower bands
        data['pct_to_upper'] = (data['bb_upper'] - data['close']) / (data['bb_upper'] - data['bb_lower'])
        data['pct_to_lower'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Label for price direction (target)
        data['next_return'] = data['close'].pct_change(1).shift(-1)
        data['direction'] = np.where(data['next_return'] > 0, 1, 0)
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data

    def train_ml_model(self, df, test_size=0.2):
        """Train Gradient Boosting classifier model"""
        features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'slowk', 'slowd',
            'mom_1d', 'mom_3d', 'mom_7d', 'adx', 'bb_width', 
            'pct_to_upper', 'pct_to_lower', 'volume_ratio',
            'bull_market', 'price_above_ema50'
        ]
        
        X = df[features]
        y = df['direction']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False
        )
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Test model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Model Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return model, scaler, features

    def generate_signals(self, df, model, scaler, features):
        """Generate trading signals using technical indicators and ML predictions"""
        data = df.copy()
        
        # Get ML model predictions
        X = data[features]
        X_scaled = scaler.transform(X)
        data['ml_signal'] = model.predict(X_scaled)
        data['ml_proba'] = model.predict_proba(X_scaled)[:, 1]
        
        # Traditional signals
        # MACD crossover
        data['macd_long'] = ((data['macd'] > data['macd_signal']) & 
                            (data['macd'].shift(1) <= data['macd_signal'].shift(1)))
        data['macd_short'] = ((data['macd'] < data['macd_signal']) & 
                             (data['macd'].shift(1) >= data['macd_signal'].shift(1)))
        
        # RSI signals
        data['rsi_oversold'] = (data['rsi'] < 30) & (data['rsi'].shift(1) < 30) & (data['rsi'] > data['rsi'].shift(1))
        data['rsi_overbought'] = (data['rsi'] > 70) & (data['rsi'].shift(1) > 70) & (data['rsi'] < data['rsi'].shift(1))
        
        # EMA crosses
        data['ema_cross_up'] = (data['ema_9'] > data['ema_21']) & (data['ema_9'].shift(1) <= data['ema_21'].shift(1))
        data['ema_cross_down'] = (data['ema_9'] < data['ema_21']) & (data['ema_9'].shift(1) >= data['ema_21'].shift(1))
        
        # Bollinger Band signals
        data['bb_lower_touch'] = (data['close'] <= data['bb_lower']) & (data['close'].shift(1) > data['bb_lower'].shift(1))
        data['bb_upper_touch'] = (data['close'] >= data['bb_upper']) & (data['close'].shift(1) < data['bb_upper'].shift(1))
        
        # Volume confirmation
        data['high_volume'] = data['volume'] > (1.5 * data['volume_sma'])
        
        # Combined signals
        # Long signal: ML prediction + technical confirmation
        data['long_signal'] = (
            (data['ml_signal'] == 1) & 
            (data['ml_proba'] > 0.65) &
            (
                (data['macd_long']) | 
                (data['rsi_oversold']) |
                (data['ema_cross_up'])
            ) &
            (data['bull_market'] == 1)
        )
        
        # Short signal: ML prediction + technical confirmation
        data['short_signal'] = (
            (data['ml_signal'] == 0) & 
            (data['ml_proba'] < 0.35) &
            (
                (data['macd_short']) | 
                (data['rsi_overbought']) |
                (data['ema_cross_down'])
            )
        )
        
        # Exit conditions
        data['exit_long'] = (
            (data['ml_signal'] == 0) |
            (data['rsi'] > 75) |
            (data['close'] < data['ema_50'])
        )
        
        data['exit_short'] = (
            (data['ml_signal'] == 1) |
            (data['rsi'] < 25) |
            (data['close'] > data['ema_50'])
        )
        
        return data

    def backtest(self, df, model, scaler, features):
        """Run backtest on the strategy"""
        signals = self.generate_signals(df, model, scaler, features)
        
        # Initialize variables
        self.capital = self.initial_capital
        self.position = 0
        self.position_price = 0
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        
        max_capital = self.capital
        peak_capital = self.capital
        max_drawdown = 0
        mae_values = []  # Maximum Adverse Excursion
        trade_count_by_month = {}
        trade_count_by_quarter = {}
        profit_by_quarter = {}
        
        for idx, row in signals.iterrows():
            current_date = idx
            year_month = f"{current_date.year}-{current_date.month:02d}"
            quarter = f"{current_date.year}-Q{(current_date.month-1)//3+1}"
            
            # Initialize tracking dictionaries if new period
            if year_month not in trade_count_by_month:
                trade_count_by_month[year_month] = 0
            if quarter not in trade_count_by_quarter:
                trade_count_by_quarter[quarter] = 0
                profit_by_quarter[quarter] = 0
            
            # Record equity at this point
            self.equity_curve.append((current_date, self.capital))
            
            # Update max capital
            if self.capital > max_capital:
                max_capital = self.capital
            
            # Calculate current drawdown
            if self.capital < peak_capital:
                current_drawdown = (peak_capital - self.capital) / peak_capital * 100
                self.drawdowns.append((current_date, current_drawdown))
                max_drawdown = max(max_drawdown, current_drawdown)
            else:
                peak_capital = self.capital
                self.drawdowns.append((current_date, 0))
            
            # Calculate MAE for open positions
            if self.position != 0:
                price = row['close']
                if self.position > 0:  # Long position
                    adverse_move = (self.position_price - min(price, self.position_price)) / self.position_price * 100
                else:  # Short position
                    adverse_move = (max(price, self.position_price) - self.position_price) / self.position_price * 100
                mae_values.append(adverse_move)
            
            # Trading logic
            if self.position == 0:  # No position
                # Long entry
                if row['long_signal']:
                    self.position = 1
                    self.position_price = row['close']
                    entry_date = current_date
                    trade_count_by_month[year_month] += 1
                    trade_count_by_quarter[quarter] += 1
                    self.trades.append({
                        'entry_date': entry_date,
                        'entry_price': self.position_price,
                        'position': 'LONG',
                        'exit_date': None,
                        'exit_price': None,
                        'pnl': None,
                        'pnl_pct': None
                    })
                
                # Short entry
                elif row['short_signal']:
                    self.position = -1
                    self.position_price = row['close']
                    entry_date = current_date
                    trade_count_by_month[year_month] += 1
                    trade_count_by_quarter[quarter] += 1
                    self.trades.append({
                        'entry_date': entry_date,
                        'entry_price': self.position_price,
                        'position': 'SHORT',
                        'exit_date': None,
                        'exit_price': None,
                        'pnl': None,
                        'pnl_pct': None
                    })
            
            elif self.position > 0:  # Long position
                # Exit long
                if row['exit_long']:
                    exit_price = row['close']
                    exit_date = current_date
                    pnl_pct = (exit_price - self.position_price) / self.position_price * 100
                    pnl = self.capital * pnl_pct / 100
                    
                    self.capital += pnl
                    profit_by_quarter[quarter] += pnl
                    
                    # Update last trade
                