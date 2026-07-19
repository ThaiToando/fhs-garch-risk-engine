import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_crisis_overlay(csv_path='results/backtest_comparison.csv'):
    # Load the backtest data
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Isolate the crisis window
    crisis_df = df.loc['2020-02-15':'2020-04-30']
    
    # Apply a clean, minimal aesthetic
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot realized returns in the background
    ax.plot(crisis_df.index, crisis_df['realized'], 
            label='Realized Returns', color='#2c3e50', alpha=0.3, linewidth=2)
    
    # Plot FHS VaR (solid, bold red for emphasis)
    ax.plot(crisis_df.index, -crisis_df['fhs_exc'], 
            label='FHS VaR (99%)', color='#c0392b', linewidth=2.5)
            
    # Plot HS VaR (dashed blue)
    ax.plot(crisis_df.index, -crisis_df['hs_exc'], 
            label='Plain HS VaR (99%)', color='#2980b9', linewidth=2, linestyle='--')
            
    # Plot Normal VaR (dotted grey)
    ax.plot(crisis_df.index, -crisis_df['normal_exc'], 
            label='Normal VaR (99%)', color='#7f8c8d', linewidth=1.5, linestyle=':')
    
    # Title and axis labels formatting
    ax.set_title('VaR Reactivity: The Feb-Apr 2020 Market Crash', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
    
    # Date formatting for a clean X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Legend styling
    ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=11)
    
    # Grid and spine cleanup for a modern look
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save and display
    plt.tight_layout()
    plt.savefig('results/crisis_overlay_beautiful.png', dpi=300)
    print("Beautiful plot saved to results/crisis_overlay_beautiful.png")
    plt.show()

if __name__ == "__main__":
    plot_crisis_overlay()