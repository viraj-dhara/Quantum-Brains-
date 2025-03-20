# Calculate buy and hold return for ETH
    eth_data = pd.read_csv('eth_usdt_data.csv')
    eth_buy_hold_return = (eth_data['close'].iloc[-1] - eth_data['close'].iloc[0]) / eth_data['close'].iloc[0] * 100
    compare_with_buy_hold(eth_metrics, eth_buy_hold_return)

# Additional class for ensemble strategy
class EnsembleStrategy:
    def __init__(self, initial_capital=10000):
        """Initialize ensemble strategy that combines multiple sub-strategies"""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategies = []
        self.weights = []
        self.equity_curve = []
        self.drawdowns = []
        
    def add_strategy(self, strategy_class, params, weight):
        """Add a strategy to the ensemble with a specific weight"""
        self.strategies.append((strategy_class, params))
        self.weights.append(weight)
        
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def run_ensemble(self, data):
        """Run the ensemble strategy on the data"""
        # Normalize weights
        self.normalize_weights()
        
        strategy_results = []
        for (strategy_class, params), weight in zip(self.strategies, self.weights):
            # Initialize and run strategy
            strategy = strategy_class(**params)
            result = strategy.run_strategy(data)
            strategy_results.append(result)
        
        # Combine equity curves with weights
        combined_equity = []
        combined_drawdowns = []
        
        # Get the longest equity curve
        max_length = max(len(strategy.equity_curve) for strategy in [s[0] for s, _ in zip(self.strategies, self.weights)])
        
        for i in range(max_length):
            weighted_equity = 0
            weighted_drawdown = 0
            current_date = None
            
            for strategy, weight in zip(self.strategies, self.weights):
                if i < len(strategy[0].equity_curve):
                    date, equity = strategy[0].equity_curve[i]
                    current_date = date
                    normalized_equity = equity / strategy[0].initial_capital * self.initial_capital * weight
                    weighted_equity += normalized_equity
                    
                    if i < len(strategy[0].drawdowns):
                        _, drawdown = strategy[0].drawdowns[i]
                        weighted_drawdown += drawdown * weight
            
            if current_date:
                combined_equity.append((current_date, weighted_equity))
                combined_drawdowns.append((current_date, weighted_drawdown))
        
        self.equity_curve = combined_equity
        self.drawdowns = combined_drawdowns
        
        # Calculate ensemble metrics
        self.capital = combined_equity[-1][1] if combined_equity else self.initial_capital
        
        # Calculate performance metrics for ensemble
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Calculate max drawdown
        max_drawdown = max([dd for _, dd in combined_drawdowns]) if combined_drawdowns else 0
        
        # Calculate Sharpe ratio
        daily_returns = []
        for i in range(1, len(combined_equity)):
            daily_return = (combined_equity[i][1] - combined_equity[i-1][1]) / combined_equity[i-1][1]
            daily_returns.append(daily_return)
        
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(365) if daily_returns else 0
        
        print("\nEnsemble Strategy Results:")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_ensemble_results(self):
        """Plot the ensemble strategy results"""
        if not self.equity_curve:
            print("Run ensemble first before plotting results.")
            return
        
        dates = [date for date, _ in self.equity_curve]
        equity = [capital for _, capital in self.equity_curve]
        
        dd_dates = [date for date, _ in self.drawdowns]
        dd_values = [dd for _, dd in self.drawdowns]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(dates, equity, label='Ensemble Equity Curve', color='purple')
        ax1.set_title('Ensemble Strategy Equity Curve')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdowns
        ax2.fill_between(dd_dates, dd_values, 0, alpha=0.3, color='red')
        ax2.plot(dd_dates, dd_values, color='red', label='Drawdown %')
        ax2.set_title('Ensemble Drawdowns')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True)
        
        # Format x-axis for better readability
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()

# Advanced Breakout Strategy
class BreakoutStrategy(CryptoTradingStrategy):
    def __init__(self, symbol='BTC/USDT', timeframe='1h', initial_capital=10000, lookback=20):
        super().__init__(symbol, timeframe, initial_capital)
        self.lookback = lookback
    
    def preprocess_data(self, df):
        """Calculate technical indicators specific for breakout strategy"""
        data = df.copy()
        
        # Calculate ATR for volatility measurement
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=self.lookback)
        
        # Calculate Donchian Channel
        data['dc_upper'] = data['high'].rolling(window=self.lookback).max()
        data['dc_lower'] = data['low'].rolling(window=self.lookback).min()
        data['dc_middle'] = (data['dc_upper'] + data['dc_lower']) / 2
        
        # Calculate Volume Indicators
        data['volume_sma'] = talib.SMA(data['volume'], timeperiod=self.lookback)
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Calculate price momentum
        data['mom_1d'] = data['close'].pct_change(1)
        data['mom_3d'] = data['close'].pct_change(3)
        data['mom_7d'] = data['close'].pct_change(7)
        
        # Calculate RSI
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        
        # Calculate MACD
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Label for price direction (target)
        data['next_return'] = data['close'].pct_change(1).shift(-1)
        data['direction'] = np.where(data['next_return'] > 0, 1, 0)
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data
    
    def generate_signals(self, df):
        """Generate trading signals using breakout strategy"""
        data = df.copy()
        
        # Calculate new highs/lows
        data['new_high'] = data['close'] > data['dc_upper'].shift(1)
        data['new_low'] = data['close'] < data['dc_lower'].shift(1)
        
        # Long signal: Breakout of upper Donchian Channel with volume confirmation
        data['long_signal'] = (
            data['new_high'] & 
            (data['volume_ratio'] > 1.5) &
            (data['rsi'] < 70)  # Avoid overbought conditions
        )
        
        # Short signal: Breakdown of lower Donchian Channel with volume confirmation
        data['short_signal'] = (
            data['new_low'] & 
            (data['volume_ratio'] > 1.5) &
            (data['rsi'] > 30)  # Avoid oversold conditions
        )
        
        # Exit conditions
        data['exit_long'] = (
            (data['close'] < data['dc_middle']) |  # Price falls below middle band
            (data['rsi'] > 75)  # RSI indicates overbought
        )
        
        data['exit_short'] = (
            (data['close'] > data['dc_middle']) |  # Price rises above middle band
            (data['rsi'] < 25)  # RSI indicates oversold
        )
        
        return data
    
    def backtest(self, df):
        """Run backtest specific to the breakout strategy"""
        signals = self.generate_signals(df)
        
        # Initialize variables
        self.capital = self.initial_capital
        self.position = 0
        self.position_price = 0
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        
        # Run the same backtest logic as the parent class
        max_capital = self.capital
        peak_capital = self.capital
        max_drawdown = 0
        mae_values = []
        trade_count_by_month = {}
        trade_count_by_quarter = {}
        profit_by_quarter = {}
        
        # Rest of the backtest logic follows the same pattern as in CryptoTradingStrategy
        # ... (same implementation as in the parent class)
        
        # The complete backtest method would be the same as in the parent class
        # To avoid redundancy, we're not repeating the entire method here
        
        return signals  # Return the signals for now as a placeholder

# Mean Reversion Strategy
class MeanReversionStrategy(CryptoTradingStrategy):
    def __init__(self, symbol='BTC/USDT', timeframe='1h', initial_capital=10000):
        super().__init__(symbol, timeframe, initial_capital)
    
    def preprocess_data(self, df):
        """Calculate technical indicators specific for mean reversion strategy"""
        data = df.copy()
        
        # Calculate Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(
            data['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # Calculate RSI
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        
        # Calculate distance from moving average
        data['ema_50'] = talib.EMA(data['close'], timeperiod=50)
        data['ema_200'] = talib.EMA(data['close'], timeperiod=200)
        data['dist_from_ema50'] = (data['close'] - data['ema_50']) / data['ema_50'] * 100
        
        # Calculate stochastic oscillator
        data['slowk'], data['slowd'] = talib.STOCH(
            data['high'], data['low'], data['close'], 
            fastk_period=14, slowk_period=3, slowk_matype=0, 
            slowd_period=3, slowd_matype=0)
        
        # Market regime features
        data['bull_market'] = (data['close'] > data['ema_200']).astype(int)
        
        # Label for price direction (target)
        data['next_return'] = data['close'].pct_change(1).shift(-1)
        data['direction'] = np.where(data['next_return'] > 0, 1, 0)
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data
    
    def generate_signals(self, df):
        """Generate trading signals using mean reversion strategy"""
        data = df.copy()
        
        # Long signal: Price below lower Bollinger Band with oversold RSI
        data['long_signal'] = (
            (data['close'] < data['bb_lower']) & 
            (data['rsi'] < 30) &
            (data['slowk'] < 20) &
            (data['slowk'] > data['slowk'].shift(1))  # Stochastic turning up
        )
        
        # Short signal: Price above upper Bollinger Band with overbought RSI
        data['short_signal'] = (
            (data['close'] > data['bb_upper']) & 
            (data['rsi'] > 70) &
            (data['slowk'] > 80) &
            (data['slowk'] < data['slowk'].shift(1))  # Stochastic turning down
        )
        
        # Exit conditions
        data['exit_long'] = (
            (data['close'] > data['bb_middle']) |  # Price rises to middle band
            (data['rsi'] > 60)  # RSI no longer oversold
        )
        
        data['exit_short'] = (
            (data['close'] < data['bb_middle']) |  # Price falls to middle band
            (data['rsi'] < 40)  # RSI no longer overbought
        )
        
        return data

# Custom features for ML model
def add_custom_features(df):
    """Add custom features for ML model training"""
    data = df.copy()
    
    # Price patterns
    data['gap_up'] = (data['open'] > data['close'].shift(1)) & (data['open'] > data['open'].shift(1))
    data['gap_down'] = (data['open'] < data['close'].shift(1)) & (data['open'] < data['open'].shift(1))
    
    # Candlestick patterns
    data['doji'] = abs(data['open'] - data['close']) < (0.1 * (data['high'] - data['low']))
    data['hammer'] = ((data['high'] - data['low']) > 3 * (data['open'] - data['close'])) & \
                    ((data['close'] - data['low']) > (0.6 * (data['high'] - data['low'])))
    data['shooting_star'] = ((data['high'] - data['low']) > 3 * (data['open'] - data['close'])) & \
                           ((data['high'] - data['close']) > (0.6 * (data['high'] - data['low'])))
    
    # Advanced trend features
    data['trend_strength'] = abs(data['close'].pct_change(20))
    data['volatility_ratio'] = data['high'].rolling(5).std() / data['high'].rolling(20).std()
    
    # Volume patterns
    data['volume_surge'] = data['volume']