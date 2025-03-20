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