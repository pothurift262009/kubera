import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def compute_v4_metrics(portfolio_rets: pd.Series):
    """
    Advanced Risk/Reward Metrics v4: Sharpe, Sortino, Calmar, PF, and Rolling Stability.
    """
    if portfolio_rets.empty:
        return {'Sharpe': 0, 'Total Return': 0}
        
    cum_rets = (1 + portfolio_rets).cumprod()
    total_ret = cum_rets.iloc[-1] - 1
    
    # Yearly bars: 75/day * 252 days
    bars_per_year = 75 * 252
    ann_ret = (1 + total_ret)**(bars_per_year / len(portfolio_rets)) - 1
    
    # Corrected Sharpe/Sortino
    rf = 0.05 / bars_per_year
    excess = portfolio_rets - rf
    sharpe = np.sqrt(bars_per_year) * excess.mean() / (excess.std() + 1e-10)
    
    downside = portfolio_rets[portfolio_rets < 0]
    sortino = np.sqrt(bars_per_year) * excess.mean() / (downside.std() + 1e-10)
    
    # Max Drawdown
    highwater = cum_rets.cummax()
    drawdown = (cum_rets / highwater) - 1
    max_dd = drawdown.min()
    
    # Profit Factor
    gross_wins = portfolio_rets[portfolio_rets > 0].sum()
    gross_losses = portfolio_rets[portfolio_rets < 0].sum()
    profit_factor = gross_wins / abs(gross_losses) if gross_losses != 0 else np.inf
    
    # Rolling Sharpe (30-day = 30 * 75 bars)
    window = 30 * 75
    if len(portfolio_rets) > window:
        rolling_sharpe = portfolio_rets.rolling(window).apply(lambda x: np.sqrt(bars_per_year) * x.mean() / (x.std() + 1e-10)).dropna()
        final_roll_sharpe = rolling_sharpe.iloc[-1]
    else:
        final_roll_sharpe = sharpe
    
    return {
        'Total Return': total_ret,
        'Ann. Return': ann_ret,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max Drawdown': max_dd,
        'Profit Factor': profit_factor,
        'Rolling 30D Sharpe': final_roll_sharpe,
        'Win Rate': (portfolio_rets > 0).mean()
    }

def run_backtest_elite_v4(df: pd.DataFrame, probs: np.array,
                         cost=0.0002, slippage=0.0001, 
                         long_threshold=0.52, short_threshold=0.55, confidence_gap_threshold=0.12,
                         vol_filter_quantile=0.2, spread_filter_threshold=0.0015):
    """
    Elite Production Backtester V4: Confidence-Based Sizing, Clustering, and Session Filters.
    Strictly follows ISSUE 1, 2, 3, 4, 5 requirements.
    """
    logger.info("Starting Elite Backtest (V4: High-Precision Sizing & Stability)...")
    
    bt_df = df.copy()
    bt_df['prob_short'] = probs[:, 0]
    bt_df['prob_long'] = probs[:, 2]
    
    # Sorting for Confidence Gap
    sorted_probs = np.sort(probs, axis=1)
    bt_df['confidence_gap'] = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    # 1. Base Signal & Filter Logic
    c_gap_mask = bt_df['confidence_gap'] > confidence_gap_threshold
    bt_df['raw_signal'] = 0
    bt_df.loc[(bt_df['prob_long'] > long_threshold) & c_gap_mask, 'raw_signal'] = 1
    bt_df.loc[(bt_df['prob_short'] > short_threshold) & c_gap_mask, 'raw_signal'] = -1
    
    # ISSUE 4: Session Filter (09:15-09:30 and 15:15-15:30)
    # Using minute-based filtering on IST index
    bt_df['time_only'] = bt_df['datetime'].dt.time
    start_cut = pd.to_datetime('09:30').time()
    end_cut = pd.to_datetime('15:15').time()
    bt_df['session_filter'] = ((bt_df['time_only'] >= start_cut) & (bt_df['time_only'] <= end_cut)).astype(int)
    
    # General Filters (OFI + Vol + Spread)
    bt_df['vol_filter'] = (bt_df['atr_pct'] > bt_df['atr_pct'].quantile(vol_filter_quantile)).astype(int)
    bt_df['spread_filter'] = (bt_df['spread_pct'] < spread_filter_threshold).astype(int)
    
    bt_df['ofi_filter'] = 0
    bt_df.loc[(bt_df['raw_signal'] == 1) & (bt_df['ofi'] > 0), 'ofi_filter'] = 1
    bt_df.loc[(bt_df['raw_signal'] == -1) & (bt_df['ofi'] < 0), 'ofi_filter'] = 1
    
    # Combined target signal
    bt_df['target_signal_filtered'] = bt_df['raw_signal'] * bt_df['session_filter'] * bt_df['vol_filter'] * bt_df['spread_filter'] * bt_df['ofi_filter']
    
    # ISSUE 3: Trade Clustering Filter (Min 3 bars between entries)
    bt_df = bt_df.sort_values(['symbol', 'datetime'])
    final_signals = []
    
    for _, group in bt_df.groupby('symbol'):
        s = group['target_signal_filtered'].values.copy()
        last_trade_idx = -999
        for i in range(len(s)):
            if s[i] != 0:
                if (i - last_trade_idx) < 3: # ISSUE 3: clustering
                    s[i] = 0
                else:
                    last_trade_idx = i
        final_signals.extend(s)
    
    bt_df['cluster_filtered_signal'] = final_signals
    
    # ISSUE 2: Confidence-Based Position Sizing
    # size = min(1.0, (confidence_gap - 0.12) * 3 + 0.5)
    scale = (bt_df['confidence_gap'] - 0.12) * 3 + 0.5
    bt_df['pos_size'] = np.clip(scale, 0, 1.0)
    
    bt_df['target_pos'] = bt_df['cluster_filtered_signal'] * bt_df['pos_size']
    
    # Smoothing & Execution (Persistence)
    bt_df['persistent_pos'] = bt_df.groupby('symbol')['target_pos'].transform(lambda x: x.replace(0, np.nan).ffill().fillna(0))
    bt_df['actual_pos'] = bt_df.groupby('symbol')['persistent_pos'].shift(1).fillna(0)
    
    # Returns & Costs
    bt_df['bar_ret'] = bt_df.groupby('symbol')['close'].pct_change().fillna(0)
    bt_df['strat_ret_gross'] = bt_df['actual_pos'] * bt_df['bar_ret']
    
    # ISSUE 1: Reduced Costs
    bt_df['turnover'] = bt_df.groupby('symbol')['actual_pos'].diff().abs().fillna(0)
    bt_df['execution_cost'] = bt_df['turnover'] * (cost + slippage)
    bt_df['net_ret'] = bt_df['strat_ret_gross'] - bt_df['execution_cost']
    
    # Global Metrics
    portfolio_rets = bt_df.groupby('datetime')['net_ret'].mean()
    metrics = compute_v4_metrics(portfolio_rets)
    
    trades = bt_df[bt_df['turnover'] > 0]
    num_trades = len(trades)
    
    logger.info(f"== V4 BACKTEST METRICS (N_TRADES={num_trades}) ==")
    for k, v in metrics.items():
        if 'Sharpe' in k or 'Return' in k or 'Drawdown' in k or 'Factor' in k:
            if 'Sharpe' in k or 'Factor' in k: logger.info(f"{k}: {v:.2f}")
            else: logger.info(f"{k}: {v:.2%}")
            
    os.makedirs('logs', exist_ok=True)
    trades.to_csv('logs/trade_log_v4.csv', index=False)
    
    return bt_df, portfolio_rets, metrics
