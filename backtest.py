import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def compute_advanced_metrics(portfolio_rets: pd.Series):
    """
    Sortino, Calmar, Max Drawdown Duration.
    """
    # 1. Total & Annualized Return
    cum_rets = (1 + portfolio_rets).cumprod()
    total_ret = cum_rets.iloc[-1] - 1
    
    # Approx 75 5-min bars per day, 252 days per year
    bars_per_year = 75 * 252
    ann_ret = (1 + total_ret)**(bars_per_year / len(portfolio_rets)) - 1
    
    # 2. Sharpe & Sortino (Annualized)
    rf = 0.05 / bars_per_year # zero approx
    excess = portfolio_rets - rf
    sharpe = np.sqrt(bars_per_year) * excess.mean() / (excess.std() + 1e-10)
    
    downside = portfolio_rets[portfolio_rets < 0]
    sortino = np.sqrt(bars_per_year) * excess.mean() / (downside.std() + 1e-10)
    
    # 3. Max Drawdown & Duration
    highwater = cum_rets.cummax()
    drawdown = (cum_rets / highwater) - 1
    max_dd = drawdown.min()
    
    drawdown_idx = drawdown[drawdown < 0].index
    if not drawdown_idx.empty:
        # Max duration in bars
        is_dd = (drawdown < 0).astype(int)
        dd_runs = is_dd.groupby((is_dd != is_dd.shift()).cumsum()).cumsum()
        max_dd_duration = dd_runs.max()
    else:
        max_dd_duration = 0
        
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    return {
        'Total Return': total_ret,
        'Ann. Return': ann_ret,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max Drawdown': max_dd,
        'Max DD Duration': max_dd_duration,
        'Calmar': calmar
    }

def run_backtest_elite(df: pd.DataFrame, preds: np.array, probs: np.array,
                      cost=0.0005, slippage=0.0002, prob_threshold=0.55):
    """
    Elite backtester with trade logging and turnover tracking.
    """
    logger.info("Starting Elite Backtest...")
    
    bt_df = df.copy()
    bt_df['pred'] = preds
    bt_df['pred_prob'] = np.max(probs, axis=1)
    
    # 1. Target Position: 0=Short, 2=Long, 1=Flat -> -1, 1, 0
    pos_map = {0: -1, 1: 0, 2: 1}
    bt_df['target_pos'] = bt_df['pred'].map(pos_map)
    bt_df.loc[bt_df['pred_prob'] < prob_threshold, 'target_pos'] = 0
    
    # 2. Execution Delay (Next Bar)
    bt_df['actual_pos'] = bt_df.groupby('symbol')['target_pos'].shift(1).fillna(0)
    
    # 3. Returns and Costs
    bt_df['bar_ret'] = bt_df.groupby('symbol')['close'].pct_change()
    bt_df['strat_ret'] = bt_df['actual_pos'] * bt_df['bar_ret']
    
    # Turnover-based costs
    bt_df['turnover'] = bt_df.groupby('symbol')['actual_pos'].diff().abs().fillna(0)
    bt_df['costs'] = bt_df['turnover'] * (cost + slippage)
    bt_df['net_ret'] = bt_df['strat_ret'] - bt_df['costs']
    
    # 4. Global Portfolio Metrics
    portfolio_rets = bt_df.groupby('datetime')['net_ret'].mean()
    metrics = compute_advanced_metrics(portfolio_rets)
    
    # 5. Trade Logging
    # We identify "trades" as when position changes from or to a non-zero state.
    # For simplicity, we log everytime position != prev_position and (pos != 0 or prev_pos != 0)
    bt_df['pos_changed'] = bt_df.groupby('symbol')['actual_pos'].diff().abs().gt(0).astype(int)
    trades = bt_df[bt_df['pos_changed'] == 1].copy()
    
    os.makedirs('logs', exist_ok=True)
    trades[['datetime', 'symbol', 'close', 'actual_pos', 'net_ret']].to_csv('logs/trade_log_elite.csv', index=False)
    logger.info(f"Detailed trade log saved to logs/trade_log_elite.csv ({len(trades)} events)")
    
    # Print Metrics
    logger.info("== ELITE BACKTEST METRICS ==")
    for k, v in metrics.items():
        if 'Return' in k or 'Drawdown' in k:
            logger.info(f"{k}: {v:.2%}")
        else:
            logger.info(f"{k}: {v:.2f}")
            
    return bt_df, portfolio_rets, metrics
