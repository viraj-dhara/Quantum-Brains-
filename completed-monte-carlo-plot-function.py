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
