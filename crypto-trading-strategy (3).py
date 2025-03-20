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