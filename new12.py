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
                # Update last trade
                    self.trades[-1]['exit_date'] = exit_date
                    self.trades[-1]['exit_price'] = exit_price
                    self.trades[-1]['pnl'] = pnl
                    self.trades[-1]['pnl_pct'] = pnl_pct
                    
                    # Reset position
                    self.position = 0
                    self.position_price = 0
                    
            elif self.position < 0:  # Short position
                # Exit short
                if row['exit_short']:
                    exit_price = row['close']
                    exit_date = current_date
                    pnl_pct = (self.position_price - exit_price) / self.position_price * 100
                    pnl = self.capital * pnl_pct / 100
                    
                    self.capital += pnl
                    profit_by_quarter[quarter] += pnl
                    
                    # Update last trade
                    self.trades[-1]['exit_date'] = exit_date
                    self.trades[-1]['exit_price'] = exit_price
                    self.trades[-1]['pnl'] = pnl
                    self.trades[-1]['pnl_pct'] = pnl_pct
                    
                    # Reset position
                    self.position = 0
                    self.position_price = 0
        
        # Close any remaining position at the end of the period
        if self.position != 0:
            last_price = signals.iloc[-1]['close']
            if self.position > 0:  # Long position
                pnl_pct = (last_price - self.position_price) / self.position_price * 100
            else:  # Short position
                pnl_pct = (self.position_price - last_price) / self.position_price * 100
            
            pnl = self.capital * pnl_pct / 100
            self.capital += pnl
            
            # Update last trade
            self.trades[-1]['exit_date'] = signals.index[-1]
            self.trades[-1]['exit_price'] = last_price
            self.trades[-1]['pnl'] = pnl
            self.trades[-1]['pnl_pct'] = pnl_pct
            
            # Update quarterly profit
            last_quarter = f"{signals.index[-1].year}-Q{(signals.index[-1].month-1)//3+1}"
            profit_by_quarter[last_quarter] += pnl
        
        # Calculate performance metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in self.trades if trade['pnl'] <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average profit and loss
        avg_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(trade['pnl'] for trade in self.trades if trade['pnl'] <= 0) / losing_trades if losing_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trades if trade['pnl'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate annualized return
        start_date = signals.index[0]
        end_date = signals.index[-1]
        years = (end_date - start_date).days / 365.25
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        
        # Calculate sharpe ratio
        daily_returns = []
        for i in range(1, len(self.equity_curve)):
            daily_return = (self.equity_curve[i][1] - self.equity_curve[i-1][1]) / self.equity_curve[i-1][1]
            daily_returns.append(daily_return)
        
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(365)
        
        # Calculate maximum adverse excursion
        max_mae = max(mae_values) if mae_values else 0
        
        # Calculate time to recovery
        recovery_times = []
        in_drawdown = False
        start_drawdown = None
        peak_capital = self.initial_capital
        
        for date, capital in self.equity_curve:
            if capital >= peak_capital:
                peak_capital = capital
                if in_drawdown:
                    recovery_time = (date - start_drawdown).days
                    recovery_times.append(recovery_time)
                    in_drawdown = False
            elif not in_drawdown and capital < peak_capital:
                in_drawdown = True
                start_drawdown = date
        
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        # Calculate profitable quarters
        profitable_quarters = sum(1 for quarter, profit in profit_by_quarter.items() if profit > 0)
        total_quarters = len(profit_by_quarter)
        
        # Calculate yearly performance
        yearly_performance = {}
        for year in set(date.year for date, _ in self.equity_curve):
            year_start = None
            year_end = None
            
            for date, capital in self.equity_curve:
                if date.year == year:
                    if year_start is None:
                        year_start = (date, capital)
                    year_end = (date, capital)
            
            if year_start and year_end:
                yearly_return = (year_end[1] - year_start[1]) / year_start[1] * 100
                yearly_performance[year] = yearly_return
        
        # Calculate monthly trade frequency
        avg_monthly_trades = sum(trade_count_by_month.values()) / len(trade_count_by_month)
        
        # Print performance metrics
        print(f"\nPerformance Metrics for {self.symbol}:")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {annualized_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Maximum Adverse Excursion: {max_mae:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Profit: ${avg_profit:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Average Recovery Time: {avg_recovery_time:.2f} days")
        print(f"Profitable Quarters: {profitable_quarters}/{total_quarters}")
        print(f"Average Monthly Trades: {avg_monthly_trades:.2f}")
        
        print("\nYearly Performance:")
        for year, perf in yearly_performance.items():
            print(f"{year}: {perf:.2f}%")
        
        # Check if the strategy meets the competition criteria
        meets_criteria = True
        criteria_results = {
            "Sharpe Ratio > 6": sharpe_ratio > 6,
            "Double Digits Time to Recovery": avg_recovery_time < 100,
            "Max Drawdown < 15%": max_drawdown < 15,
            "MAE < 15%": max_mae < 15,
            "Quarterly Profitability (12/16 quarters)": profitable_quarters >= 12,
            "Yearly Profitability (all 4 years)": all(yearly_performance.values()) > 0,
            "Minimum Trade Frequency (≥ 4 trades/month)": avg_monthly_trades >= 4
        }
        
        print("\nCompetition Criteria Check:")
        for criterion, result in criteria_results.items():
            print(f"{criterion}: {'PASS' if result else 'FAIL'}")
            if not result:
                meets_criteria = False
        
        print(f"\nOverall: {'PASS' if meets_criteria else 'FAIL'}")
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_mae': max_mae,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_recovery_time': avg_recovery_time,
            'profitable_quarters': profitable_quarters,
            'total_quarters': total_quarters,
            'yearly_performance': yearly_performance,
            'avg_monthly_trades': avg_monthly_trades,
            'meets_criteria': meets_criteria,
            'criteria_results': criteria_results
        }

    def plot_equity_curve(self):
        """Plot equity curve and drawdowns"""
        if not self.equity_curve:
            print("Backtest first before plotting equity curve.")
            return
        
        dates = [date for date, _ in self.equity_curve]
        equity = [capital for _, capital in self.equity_curve]
        
        dd_dates = [date for date, _ in self.drawdowns]
        dd_values = [dd for _, dd in self.drawdowns]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(dates, equity, label='Equity Curve', color='blue')
        ax1.set_title(f'{self.symbol} Equity Curve')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdowns
        ax2.fill_between(dd_dates, dd_values, 0, alpha=0.3, color='red')
        ax2.plot(dd_dates, dd_values, color='red', label='Drawdown %')
        ax2.set_title('Drawdowns')
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

    def plot_trade_distribution(self):
        """Plot trade distribution and statistics"""
        if not self.trades:
            print("Backtest first before plotting trade distribution.")
            return
        
        # Extract PnL percentages
        pnl_pcts = [trade['pnl_pct'] for trade in self.trades if trade['pnl_pct'] is not None]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot histogram of PnL percentages
        ax1.hist(pnl_pcts, bins=20, alpha=0.7, color='blue')
        ax1.axvline(0, color='r', linestyle='dashed', linewidth=1)
        ax1.set_title('Trade PnL Distribution')
        ax1.set_xlabel('PnL %')
        ax1.set_ylabel('Frequency')
        ax1.grid(True)
        
        # Plot trade PnL over time
        dates = [trade['exit_date'] for trade in self.trades if trade['exit_date'] is not None]
        pnls = [trade['pnl'] for trade in self.trades if trade['pnl'] is not None]
        
        colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors)
        ax2.set_title('Trade PnL Over Time')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('PnL ($)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def run_strategy(self, file_path):
        """Run the complete strategy"""
        # Load data
        print(f"Loading data for {self.symbol}...")
        data = self.load_data(file_path)
        
        # Preprocess data
        print("Preprocessing data and calculating indicators...")
        processed_data = self.preprocess_data(data)
        
        # Train model
        print("Training ML model...")
        model, scaler, features = self.train_ml_model(processed_data)
        
        # Run backtest
        print("Running backtest...")
        metrics = self.backtest(processed_data, model, scaler, features)
        
        # Plot results
        self.plot_equity_curve()
        self.plot_trade_distribution()
        
        return metrics

# Helper function to compare strategy with buy and hold
def compare_with_buy_hold(strategy_metrics, buy_hold_return):
    """Compare strategy with buy and hold"""
    print("\nComparison with Buy and Hold:")
    print(f"Strategy Total Return: {strategy_metrics['total_return']:.2f}%")
    print(f"Buy and Hold Return: {buy_hold_return:.2f}%")
    print(f"Outperformance: {strategy_metrics['total_return'] - buy_hold_return:.2f}%")
    
    if strategy_metrics['total_return'] > buy_hold_return:
        print("Strategy OUTPERFORMS Buy and Hold")
    else:
        print("Strategy UNDERPERFORMS Buy and Hold")

# Main execution
if __name__ == "__main__":
    # BTC/USDT Strategy
    btc_strategy = CryptoTradingStrategy(symbol='BTC/USDT', initial_capital=10000)
    btc_metrics = btc_strategy.run_strategy('btc_usdt_data.csv')
    
    # Calculate buy and hold return for BTC
    btc_data = pd.read_csv('btc_usdt_data.csv')
    btc_buy_hold_return = (btc_data['close'].iloc[-1] - btc_data['close'].iloc[0]) / btc_data['close'].iloc[0] * 100
    compare_with_buy_hold(btc_metrics, btc_buy_hold_return)
    
    # ETH/USDT Strategy
    eth_strategy = CryptoTradingStrategy(symbol='ETH/USDT', initial_capital=10000)
    eth_metrics = eth_strategy.run_strategy('eth_usdt_data.csv')
    
    # Calculate buy and hold return for ETH
    eth_data = pd.rea
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
    # Volume patterns
    data['volume_surge'] = data['volume'] > (data['volume'].rolling(20).mean() * 2)
    data['volume_climax'] = (data['volume'] > data['volume'].rolling(20).max()) & (data['close'] < data['close'].shift(1))
    data['volume_trend'] = data['volume'].rolling(10).mean() / data['volume'].rolling(30).mean()
    
    # Market regime indicators
    data['bull_regime'] = (data['close'] > data['ema_50']) & (data['ema_50'] > data['ema_200'])
    data['bear_regime'] = (data['close'] < data['ema_50']) & (data['ema_50'] < data['ema_200'])
    data['chop_regime'] = ~(data['bull_regime'] | data['bear_regime'])
    
    # Momentum and volatility
    data['close_to_atr_ratio'] = data['close'] / data['atr']
    data['momentum_strength'] = abs(data['mom_7d']) / data['close'].rolling(7).std()
    
    # Advanced oscillator metrics
    data['rsi_divergence'] = (data['close'] > data['close'].shift(5)) & (data['rsi'] < data['rsi'].shift(5))
    data['overbought'] = (data['rsi'] > 70) & (data['slowk'] > 80)
    data['oversold'] = (data['rsi'] < 30) & (data['slowk'] < 20)
    
    return data

# Main execution with ensemble approach
def run_ensemble_strategy():
    print("Running Ensemble Strategy for Crypto Trading...")
    
    # Load data
    btc_data = pd.read_csv('btc_usdt_data.csv')
    eth_data = pd.read_csv('eth_usdt_data.csv')
    
    # Create ensemble strategy
    ensemble = EnsembleStrategy(initial_capital=10000)
    
    # Add main ML-based strategy for BTC with 40% weight
    ensemble.add_strategy(
        CryptoTradingStrategy,
        {'symbol': 'BTC/USDT', 'initial_capital': 10000},
        0.4
    )
    
    # Add breakout strategy for BTC with 20% weight
    ensemble.add_strategy(
        BreakoutStrategy,
        {'symbol': 'BTC/USDT', 'initial_capital': 10000, 'lookback': 20},
        0.2
    )
    
    # Add mean reversion strategy for BTC with 10% weight
    ensemble.add_strategy(
        MeanReversionStrategy,
        {'symbol': 'BTC/USDT', 'initial_capital': 10000},
        0.1
    )
    
    # Add main ML-based strategy for ETH with 30% weight
    ensemble.add_strategy(
        CryptoTradingStrategy,
        {'symbol': 'ETH/USDT', 'initial_capital': 10000},
        0.3
    )
    
    # Run ensemble strategy
    ensemble_metrics = ensemble.run_ensemble([btc_data, eth_data])
    
    # Plot ensemble results
    ensemble.plot_ensemble_results()
    
    return ensemble_metrics

# Adaptive Risk Management Class
class AdaptiveRiskManager:
    def __init__(self, strategy, max_risk_per_trade=0.02):
        """
        Initialize risk manager with a trading strategy
        
        Parameters:
        strategy (CryptoTradingStrategy): Trading strategy to manage risk for
        max_risk_per_trade (float): Maximum risk per trade as fraction of capital
        """
        self.strategy = strategy
        self.max_risk_per_trade = max_risk_per_trade
        self.volatility_lookback = 20
        self.max_drawdown_threshold = 0.1  # 10% max drawdown threshold
        self.current_drawdown = 0
        self.position_sizes = []
    
    def calculate_position_size(self, price, stop_loss_pct):
        """Calculate position size based on risk parameters"""
        # Calculate risk amount
        risk_amount = self.strategy.capital * self.max_risk_per_trade
        
        # Calculate position size
        stop_loss_amount = price * stop_loss_pct
        position_size = risk_amount / stop_loss_amount
        
        # Adjust for current drawdown
        drawdown_factor = 1 - (self.current_drawdown / self.max_drawdown_threshold)
        drawdown_factor = max(0.25, min(1.0, drawdown_factor))  # Limit to 25%-100%
        
        # Adjust position size
        adjusted_position = position_size * drawdown_factor
        
        # Record position size
        self.position_sizes.append(adjusted_position)
        
        return adjusted_position
    
    def update_drawdown(self, current_capital):
        """Update current drawdown based on peak capital"""
        if not hasattr(self, 'peak_capital'):
            self.peak_capital = current_capital
        
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
    
    def calculate_stop_loss(self, data, is_long):
        """Calculate adaptive stop loss based on market volatility"""
        # Use ATR for volatility-based stop loss
        current_atr = data['atr'].iloc[-1]
        
        if is_long:
            # Long position - ATR-based stop below entry
            stop_multiplier = 2.0  # Adjust based on your risk tolerance
            stop_loss_pct = (stop_multiplier * current_atr) / data['close'].iloc[-1]
        else:
            # Short position - ATR-based stop above entry
            stop_multiplier = 2.0
            stop_loss_pct = (stop_multiplier * current_atr) / data['close'].iloc[-1]
        
        # Ensure stop loss is at least 1% and at most 5%
        stop_loss_pct = max(0.01, min(0.05, stop_loss_pct))
        
        return stop_loss_pct
    
    def adjust_for_correlation(self, other_position):
        """Adjust position size based on correlation with other assets"""
        # If we already have a position in a correlated asset, reduce this position
        if other_position != 0:
            correlation_factor = 0.7  # Assume 70% correlation between BTC and ETH
            return 1 - (abs(other_position) * correlation_factor)
        return 1.0

# Strategy optimization with walk-forward analysis
def perform_walk_forward_optimization(strategy_class, params, data, test_periods=4):
    """
    Perform walk-forward optimization on strategy parameters
    
    Parameters:
    strategy_class: Class of the strategy to optimize
    params: Dictionary of parameters to optimize
    data: Historical data DataFrame
    test_periods: Number of test periods to use
    
    Returns:
    Optimized parameters and out-of-sample performance metrics
    """
    # Split data into chunks for walk-forward analysis
    chunk_size = len(data) // (test_periods + 1)
    best_params = []
    oos_metrics = []
    
    for i in range(test_periods):
        # Define in-sample and out-of-sample data
        train_start = 0
        train_end = (i + 1) * chunk_size
        test_start = train_end
        test_end = test_start + chunk_size
        
        train_data = data.iloc[train_start:train_end].copy()
        test_data = data.iloc[test_start:test_end].copy()
        
        # Grid search for best parameters using in-sample data
        best_sharpe = 0
        optimal_params = None
        
        # Define parameter grid
        param_grid = {
            'lookback': [10, 15, 20, 25, 30],
            'fast_period': [8, 12, 16],
            'slow_period': [21, 26, 30],
            'signal_period': [7, 9, 11]
        }
        
        # Generate parameter combinations
        from itertools import product
        param_combinations = list(product(
            param_grid['lookback'],
            param_grid['fast_period'],
            param_grid['slow_period'],
            param_grid['signal_period']
        ))
        
        # Optimize parameters on in-sample data
        for lookback, fast, slow, signal in param_combinations:
            # Skip invalid combinations
            if fast >= slow:
                continue
                
            # Update parameters
            test_params = params.copy()
            test_params.update({
                'lookback': lookback,
                'fast_period': fast,
                'slow_period': slow,
                'signal_period': signal
            })
            
            # Initialize strategy with parameters
            strategy = strategy_class(**test_params)
            
            # Process data with these parameters
            processed_data = strategy.preprocess_data(train_data)
            
            # For ML-based strategy, train model
            if hasattr(strategy, 'train_ml_model'):
                model, scaler, features = strategy.train_ml_model(processed_data)
                metrics = strategy.backtest(processed_data, model, scaler, features)
            else:
                # For traditional strategies, just backtest
                metrics = strategy.backtest(processed_data)
            
            # Check if this parameter set is better
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                optimal_params = test_params
        
        # Test optimal parameters on out-of-sample data
        strategy = strategy_class(**optimal_params)
        test_processed = strategy.preprocess_data(test_data)
        
        # For ML-based strategy, train model on in-sample and test on out-of-sample
        if hasattr(strategy, 'train_ml_model'):
            train_processed = strategy.preprocess_data(train_data)
            model, scaler, features = strategy.train_ml_model(train_processed)
            oos_performance = strategy.backtest(test_processed, model, scaler, features)
        else:
            # For traditional strategies, just backtest out-of-sample
            oos_performance = strategy.backtest(test_processed)
        
        # Record results
        best_params.append(optimal_params)
        oos_metrics.append(oos_performance)
        
        print(f"Period {i+1} Optimal Parameters: {optimal_params}")
        print(f"Period {i+1} Out-of-Sample Sharpe: {oos_performance['sharpe_ratio']:.2f}")
        print(f"Period {i+1} Out-of-Sample Return: {oos_performance['total_return']:.2f}%")
        print(f"Period {i+1} Out-of-Sample Max Drawdown: {oos_performance['max_drawdown']:.2f}%")
        print("-" * 50)
    
    # Aggregate results
    avg_oos_sharpe = sum(m['sharpe_ratio'] for m in oos_metrics) / len(oos_metrics)
    avg_oos_return = sum(m['total_return'] for m in oos_metrics) / len(oos_metrics)
    avg_oos_drawdown = sum(m['max_drawdown'] for m in oos_metrics) / len(oos_metrics)
    
    print("\nWalk-Forward Optimization Results:")
    print(f"Average Out-of-Sample Sharpe: {avg_oos_sharpe:.2f}")
    print(f"Average Out-of-Sample Return: {avg_oos_return:.2f}%")
    print(f"Average Out-of-Sample Max Drawdown: {avg_oos_drawdown:.2f}%")
    
    # Choose final parameters (could use most recent or average)
    final_params = best_params[-1]  # Use most recent optimal parameters
    
    return final_params, oos_metrics

# Statistical analysis functions
def perform_strategy_analysis(trades, equity_curve):
    """Perform detailed statistical analysis on strategy performance"""
    # Extract trade data
    pnl_pcts = [trade['pnl_pct'] for trade in trades if trade['pnl_pct'] is not None]
    trade_durations = [(trade['exit_date'] - trade['entry_date']).total_seconds() / 86400 
                      for trade in trades if trade['exit_date'] is not None]
    
    # Calculate basic statistics
    win_rate = len([p for p in pnl_pcts if p > 0]) / len(pnl_pcts) if pnl_pcts else 0
    avg_win = np.mean([p for p in pnl_pcts if p > 0]) if any(p > 0 for p in pnl_pcts) else 0
    avg_loss = np.mean([p for p in pnl_pcts if p <= 0]) if any(p <= 0 for p in pnl_pcts) else 0
    
    # Calculate expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Calculate trade statistics
    avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
    max_consecutive_wins = max_consecutive_sequence(pnl_pcts, lambda x: x > 0)
    max_consecutive_losses = max_consecutive_sequence(pnl_pcts, lambda x: x <= 0)
    
    # Calculate equity curve statistics
    equity_values = [eq for _, eq in equity_curve]
    returns = [equity_values[i] / equity_values[i-1] - 1 for i in range(1, len(equity_values))]
    
    # Calculate advanced metrics
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    cagr = (equity_values[-1] / equity_values[0]) ** (252 / len(returns)) - 1
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    sortino_ratio = (np.mean(returns) / np.std([r for r in returns if r < 0])) * np.sqrt(252) if np.std([r for r in returns if r < 0]) > 0 else 0
    
    # Calculate drawdown statistics
    underwater = [1 - (equity_values[i] / max(equity_values[:i+1])) for i in range(len(equity_values))]
    max_drawdown = max(underwater) * 100
    
    # Calculate time to recovery
    in_drawdown = False
    drawdown_start = 0
    recovery_times = []
    
    for i, uwr in enumerate(underwater):
        if not in_drawdown and uwr > 0:
            in_drawdown = True
            drawdown_start = i
        elif in_drawdown and uwr == 0:
            in_drawdown = False
            recovery_time = i - drawdown_start
            recovery_times.append(recovery_time)
    
    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
    max_recovery_time = max(recovery_times) if recovery_times else 0
    
    # Print statistics
    print("\nDetailed Strategy Analysis:")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Profit Factor: {abs(win_rate * avg_win / ((1 - win_
                                                       # Helper function for calculating consecutive sequences
def max_consecutive_sequence(values, condition_func):
    """Calculate the maximum consecutive sequence that satisfies a condition"""
    max_streak = 0
    current_streak = 0
    
    for value in values:
        if condition_func(value):
            current_streak += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak = 0
    
    return max(max_streak, current_streak)

# Statistical analysis functions
def perform_strategy_analysis(trades, equity_curve):
    """Perform detailed statistical analysis on strategy performance"""
    # Extract trade data
    pnl_pcts = [trade['pnl_pct'] for trade in trades if trade['pnl_pct'] is not None]
    trade_durations = [(trade['exit_date'] - trade['entry_date']).total_seconds() / 86400 
                      for trade in trades if trade['exit_date'] is not None]
    
    # Calculate basic statistics
    win_rate = len([p for p in pnl_pcts if p > 0]) / len(pnl_pcts) if pnl_pcts else 0
    avg_win = np.mean([p for p in pnl_pcts if p > 0]) if any(p > 0 for p in pnl_pcts) else 0
    avg_loss = np.mean([p for p in pnl_pcts if p <= 0]) if any(p <= 0 for p in pnl_pcts) else 0
    
    # Calculate expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Calculate trade statistics
    avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
    max_consecutive_wins = max_consecutive_sequence(pnl_pcts, lambda x: x > 0)
    max_consecutive_losses = max_consecutive_sequence(pnl_pcts, lambda x: x <= 0)
    
    # Calculate equity curve statistics
    equity_values = [eq for _, eq in equity_curve]
    returns = [equity_values[i] / equity_values[i-1] - 1 for i in range(1, len(equity_values))]
    
    # Calculate advanced metrics
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    cagr = (equity_values[-1] / equity_values[0]) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    sortino_ratio = (np.mean(returns) / np.std([r for r in returns if r < 0])) * np.sqrt(252) if np.std([r for r in returns if r < 0]) > 0 else 0
    
    # Calculate drawdown statistics
    underwater = [1 - (equity_values[i] / max(equity_values[:i+1])) for i in range(len(equity_values))]
    max_drawdown = max(underwater) * 100 if underwater else 0
    
    # Calculate time to recovery
    in_drawdown = False
    drawdown_start = 0
    recovery_times = []
    
    for i, uwr in enumerate(underwater):
        if not in_drawdown and uwr > 0:
            in_drawdown = True
            drawdown_start = i
        elif in_drawdown and uwr == 0:
            in_drawdown = False
            recovery_time = i - drawdown_start
            recovery_times.append(recovery_time)
    
    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
    max_recovery_time = max(recovery_times) if recovery_times else 0
    
    # Calculate Calmar ratio
    calmar_ratio = cagr / (max_drawdown/100) if max_drawdown > 0 else float('inf')
    
    # Calculate Kelly criterion
    if avg_loss != 0:
        kelly_percentage = win_rate - ((1 - win_rate) / (abs(avg_win / avg_loss)))
    else:
        kelly_percentage = 1.0 if win_rate > 0 else 0.0
    
    # Calculate Profit Factor
    gross_profit = sum([t['pnl'] for t in trades if t['pnl'] is not None and t['pnl'] > 0])
    gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] is not None and t['pnl'] < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Statistical significance tests
    from scipy import stats
    
    # Test if returns are significantly different from zero
    t_stat, p_value = stats.ttest_1samp(returns, 0) if returns else (0, 1)
    returns_significant = p_value < 0.05
    
    # Test if win rate is significantly different from 50%
    win_counts = [1 if p > 0 else 0 for p in pnl_pcts]
    binom_test = stats.binom_test(sum(win_counts), len(win_counts), p=0.5) if win_counts else 1
    winrate_significant = binom_test < 0.05
    
    # Calculate autocorrelation of returns (for independence test)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    try:
        lb_stat, lb_pvalue = acorr_ljungbox(returns, lags=10, return_df=False) if len(returns) > 10 else ([0], [1])
        returns_independent = all(p > 0.05 for p in lb_pvalue)
    except:
        returns_independent = True
    
    # Print statistics
    print("\nDetailed Strategy Analysis:")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Expectancy per Trade: {expectancy:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Trade Duration: {avg_trade_duration:.2f} days")
    print(f"Max Consecutive Wins: {max_consecutive_wins}")
    print(f"Max Consecutive Losses: {max_consecutive_losses}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    print(f"Kelly Percentage: {kelly_percentage:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Average Recovery Time: {avg_recovery_time:.2f} days")
    print(f"Maximum Recovery Time: {max_recovery_time:.2f} days")
    
    print("\nStatistical Significance Tests:")
    print(f"Returns significantly different from zero: {'Yes' if returns_significant else 'No'} (p-value: {p_value:.4f})")
    print(f"Win rate significantly different from 50%: {'Yes' if winrate_significant else 'No'} (p-value: {binom_test:.4f})")
    print(f"Returns appear to be independent: {'Yes' if returns_independent else 'No'}")
    
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'avg_trade_duration': avg_trade_duration,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'kelly_percentage': kelly_percentage,
        'max_drawdown': max_drawdown,
        'avg_recovery_time': avg_recovery_time,
        'max_recovery_time': max_recovery_time,
        'returns_significant': returns_significant,
        'returns_pvalue': p_value,
        'winrate_significant': winrate_significant,
        'winrate_pvalue': binom_test,
        'returns_independent': returns_independent
    }

def monte_carlo_simulation(trades, initial_capital, num_simulations=1000):
    """
    Run Monte Carlo simulation by resampling trades
    
    Parameters:
    trades (list): List of trade dictionaries with PnL information
    initial_capital (float): Initial capital amount
    num_simulations (int): Number of Monte Carlo simulations to run
    
    Returns:
    dict: Monte Carlo simulation results
    """
    import random
    from tqdm import tqdm
    
    # Extract completed trades with PnL
    completed_trades = [t for t in trades if t['pnl'] is not None and t['pnl_pct'] is not None]
    
    if len(completed_trades) < 10:
        print("Not enough trades for Monte Carlo simulation")
        return None
    
    # Extract PnL percentages for resampling
    pnl_percentages = [t['pnl_pct'] for t in completed_trades]
    
    # Initialize simulation results
    final_capitals = []
    max_drawdowns = []
    cagrs = []
    
    print(f"Running {num_simulations} Monte Carlo simulations...")
    
    for _ in tqdm(range(num_simulations)):
        # Resample trades with replacement
        sampled_returns = random.choices(pnl_percentages, k=len(pnl_percentages))
        
        # Initialize equity curve for this simulation
        equity = [initial_capital]
        underwater = [0]
        peak_equity = initial_capital
        
        # Simulate equity curve
        for ret_pct in sampled_returns:
            # Apply return to current equity
            current_equity = equity[-1] * (1 + ret_pct/100)
            equity.append(current_equity)
            
            # Update peak equity and underwater
            peak_equity = max(peak_equity, current_equity)
            underwater.append((peak_equity - current_equity) / peak_equity * 100)
        
        # Calculate metrics
        final_capital = equity[-1]
        max_drawdown = max(underwater)
        # Calculate CAGR - assume 1 year for simplicity, can be adjusted based on trade frequencies
        cagr = (final_capital / initial_capital) ** (1 / (len(equity) / 252)) - 1
        
        # Store results
        final_capitals.append(final_capital)
        max_drawdowns.append(max_drawdown)
        cagrs.append(cagr)
    
    # Calculate statistics from simulations
    final_capital_mean = np.mean(final_capitals)
    final_capital_std = np.std(final_capitals)
    final_capital_median = np.median(final_capitals)
    
    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 95]
    capital_percentiles = {p: np.percentile(final_capitals, p) for p in percentiles}
    drawdown_percentiles = {p: np.percentile(max_drawdowns, p) for p in percentiles}
    cagr_percentiles = {p: np.percentile(cagrs, p) for p in percentiles}
    
    # Calculate probability of profit
    profit_probability = sum(1 for c in final_capitals if c > initial_capital) / num_simulations
    
    # Calculate value at risk (VaR)
    var_95 = np.percentile(final_capitals, 5)
    var_95_pct = (var_95 - initial_capital) / initial_capital * 100
    
    # Calculate conditional value at risk (CVaR / Expected Shortfall)
    cvar_95_values = [c for c in final_capitals if c <= var_95]
    cvar_95 = np.mean(cvar_95_values) if cvar_95_values else var_95
    cvar_95_pct = (cvar_95 - initial_capital) / initial_capital * 100
    
    # Print results
    print("\nMonte Carlo Simulation Results:")
    print(f"Number of Simulations: {num_simulations}")
    print(f"Number of Trades per Simulation: {len(pnl_percentages)}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Mean Final Capital: ${final_capital_mean:.2f} ({(final_capital_mean-initial_capital)/initial_capital*100:.2f}%)")
    print(f"Median Final Capital: ${final_capital_median:.2f} ({(final_capital_median-initial_capital)/initial_capital*100:.2f}%)")
    print(f"Standard Deviation of Final Capital: ${final_capital_std:.2f}")
    print(f"Probability of Profit: {profit_probability:.2%}")
    
    print("\nFinal Capital Percentiles:")
    for p, value in capital_percentiles.items():
        change = (value - initial_capital) / initial_capital * 100
        print(f"{p}th Percentile: ${value:.2f} ({change:.2f}%)")
    
    print("\nMaximum Drawdown Percentiles:")
    for p, value in drawdown_percentiles.items():
        print(f"{p}th Percentile: {value:.2f}%")
    
    print("\nCAGR Percentiles:")
    for p, value in cagr_percentiles.items():
        print(f"{p}th Percentile: {value:.2%}")
    
    print(f"\nValue at Risk (95%): ${var_95:.2f} ({var_95_pct:.2f}%)")
    print(f"Conditional Value at Risk (95%): ${cvar_95:.2f} ({cvar_95_pct:.2f}%)")
    
    return {
        'initial_capital': initial_capital,
        'num_simulations': num_simulations,
        'final_capitals': final_capitals,
        'max_drawdowns': max_drawdowns,
        'cagrs': cagrs,
        'final_capital_mean': final_capital_mean,
        'final_capital_median': final_capital_median,
        'final_capital_std': final_capital_std,
        'capital_percentiles': capital_percentiles,
        'drawdown_percentiles': drawdown_percentiles,
        'cagr_percentiles': cagr_percentiles,
        'profit_probability': profit_probability,
        'var_95': var_95,
        'var_95_pct': var_95_pct,
        'cvar_95': cvar_95,
        'cvar_95_pct': cvar_95_pct
    }

def plot_monte_carlo_results(mc_results):
    """
    Plot Monte Carlo simulation results
    
    Parameters:
    mc_results (dict): Results from monte_carlo_simulation function
    """
    plt.figure(figsize=(15, 10))
    
    # Create 2
    def plot_monte_carlo_results(mc_results):
    """
    Plot Monte Carlo simulation results
    
    Parameters:
    mc_results (dict): Results from monte_carlo_simulation function
    """
    plt.figure(figsize=(15, 10))
    
    # Create 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Distribution of Final Capital
    axs[0, 0].hist(mc_results['final_capitals'], bins=50, alpha=0.7, color='blue')
    axs[0, 0].axvline(mc_results['initial_capital'], color='red', linestyle='--', 
                     label=f'Initial Capital: ${mc_results["initial_capital"]:.2f}')
    axs[0, 0].axvline(mc_results['final_capital_mean'], color='green', linestyle='-', 
                     label=f'Mean: ${mc_results["final_capital_mean"]:.2f}')
    axs[0, 0].axvline(mc_results['capital_percentiles'][5], color='orange', linestyle='-.', 
                     label=f'5th Percentile: ${mc_results["capital_percentiles"][5]:.2f}')
    axs[0, 0].axvline(mc_results['capital_percentiles'][95], color='purple', linestyle='-.', 
                     label=f'95th Percentile: ${mc_results["capital_percentiles"][95]:.2f}')
    axs[0, 0].set_title('Distribution of Final Capital')
    axs[0, 0].set_xlabel('Final Capital ($)')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].legend()
    
    # Plot 2: Distribution of Maximum Drawdowns
    axs[0, 1].hist(mc_results['max_drawdowns'], bins=50, alpha=0.7, color='red')
    axs[0, 1].axvline(mc_results['drawdown_percentiles'][50], color='black', linestyle='-', 
                     label=f'Median: {mc_results["drawdown_percentiles"][50]:.2f}%')
    axs[0, 1].axvline(mc_results['drawdown_percentiles'][95], color='darkred', linestyle='--', 
                     label=f'95th Percentile: {mc_results["drawdown_percentiles"][95]:.2f}%')
    axs[0, 1].set_title('Distribution of Maximum Drawdowns')
    axs[0, 1].set_xlabel('Maximum Drawdown (%)')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].legend()
    
    # Plot 3: Distribution of CAGRs
    axs[1, 0].hist(mc_results['cagrs'], bins=50, alpha=0.7, color='green')
    axs[1, 0].axvline(0, color='red', linestyle='--', label='Break Even (0%)')
    axs[1, 0].axvline(np.mean(mc_results['cagrs']), color='darkgreen', linestyle='-', 
                     label=f'Mean: {np.mean(mc_results["cagrs"]):.2%}')
    axs[1, 0].axvline(mc_results['cagr_percentiles'][5], color='orange', linestyle='-.', 
                     label=f'5th Percentile: {mc_results["cagr_percentiles"][5]:.2%}')
    axs[1, 0].set_title('Distribution of CAGRs')
    axs[1, 0].set_xlabel('CAGR')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].legend()
    
    # Plot 4: Sample of equity curves
    # Generate random sample of simulation indices
    import random
    num_curves = min(50, mc_results['num_simulations'])
    sample_indices = random.sample(range(mc_results['num_simulations']), num_curves)
    
    # Reconstruct sample equity curves
    initial_capital = mc_results['initial_capital']
    
    for idx in sample_indices:
        # Get the returns for this simulation
        pnl_pcts = mc_results['final_capitals'][idx] / initial_capital - 1
        
        # Create a simple equity curve assuming steady growth
        x = np.linspace(0, 1, 100)  # Normalized time periods
        y = initial_capital * (1 + x * pnl_pcts)
        
        # Plot with low alpha for visualization
        axs[1, 1].plot(x, y, alpha=0.1, color='blue')
    
    # Plot key percentiles as thicker lines
    for percentile in [5, 50, 95]:
        pct_return = mc_results['capital_percentiles'][percentile] / initial_capital - 1
        x = np.linspace(0, 1, 100)
        y = initial_capital * (1 + x * pct_return)
        
        if percentile == 5:
            color, label = 'red', '5th Percentile'
        elif percentile == 50:
            color, label = 'black', 'Median'
        else:
            color, label = 'green', '95th Percentile'
            
        axs[1, 1].plot(x, y, color=color, linewidth=2, label=label)
    
    axs[1, 1].axhline(initial_capital, color='gray', linestyle='--', 
                     label=f'Initial: ${initial_capital:.2f}')
    axs[1, 1].set_title('Sample of Simulated Equity Curves')
    axs[1, 1].set_xlabel('Normalized Time')
    axs[1, 1].set_ylabel('Account Value ($)')
    axs[1, 1].legend()
    
    # Adjust layout and show
    plt.tight_layout()
    plt.suptitle(f'Monte Carlo Simulation Results ({mc_results["num_simulations"]} Simulations)', 
                fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Create additional plot for risk metrics
    plt.figure(figsize=(10, 6))
    
    # Calculate return bins for histogram
    returns = [(cap - mc_results['initial_capital']) / mc_results['initial_capital'] * 100 
              for cap in mc_results['final_capitals']]
    
    plt.hist(returns, bins=50, alpha=0.7, color='blue')
    plt.axvline(0, color='red', linestyle='--', label='Break Even (0%)')
    plt.axvline(mc_results['var_95_pct'], color='orange', linestyle='-', 
               label=f'VaR 95%: {mc_results["var_95_pct"]:.2f}%')
    plt.axvline(mc_results['cvar_95_pct'], color='red', linestyle='-', 
               label=f'CVaR 95%: {mc_results["cvar_95_pct"]:.2f}%')
    
    # Add vertical lines for mean and median
    mean_return = (mc_results['final_capital_mean'] - mc_results['initial_capital']) / mc_results['initial_capital'] * 100
    median_return = (mc_results['final_capital_median'] - mc_results['initial_capital']) / mc_results['initial_capital'] * 100
    
    plt.axvline(mean_return, color='green', linestyle='-', 
               label=f'Mean: {mean_return:.2f}%')
    plt.axvline(median_return, color='black', linestyle=':', 
               label=f'Median: {median_return:.2f}%')
    
    plt.title('Distribution of Returns with Risk Metrics')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print risk-reward summary
    print("\nRisk-Reward Summary:")
    print(f"Probability of Profit: {mc_results['profit_probability']:.2%}")
    print(f"Median Return: {median_return:.2f}%")
    print(f"Value at Risk (95%): {mc_results['var_95_pct']:.2f}%")
    print(f"Conditional VaR (95%): {mc_results['cvar_95_pct']:.2f}%")
    print(f"Risk-Reward Ratio: {median_return / abs(mc_results['var_95_pct']):.2f}")
