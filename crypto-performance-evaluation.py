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