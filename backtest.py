import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def compute_alpha_metrics(portfolio_rets: pd.Series):
    if portfolio_rets.empty: return {'Sharpe': -9, 'Profit Factor': 0}
    cum_rets = (1 + portfolio_rets).cumprod()
    total_ret = cum_rets.iloc[-1] - 1
    bars_per_year = 75 * 252
    
    ann_ret = (1 + total_ret)**(bars_per_year / len(portfolio_rets)) - 1
    sharpe = np.sqrt(bars_per_year) * portfolio_rets.mean() / (portfolio_rets.std() + 1e-10)
    
    downside = portfolio_rets[portfolio_rets < 0]
    sortino = np.sqrt(bars_per_year) * portfolio_rets.mean() / (downside.std() + 1e-10)
    
    highwater = cum_rets.cummax()
    max_dd = ((cum_rets / highwater) - 1).min()
    
    gross_wins = portfolio_rets[portfolio_rets > 0].sum()
    gross_losses = portfolio_rets[portfolio_rets < 0].sum()
    profit_factor = gross_wins / abs(gross_losses) if gross_losses != 0 else np.inf
    
    return {
        'Total Return': total_ret,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Profit Factor': profit_factor,
        'Win Rate': (portfolio_rets > 0).mean()
    }

def size_fn_v12(p, threshold=0.5):
    """
    Fixed Position Sizing (Fix 3): Quadratic scaling for high-conviction signals.
    Amplifies signals significantly as they approach probability = 1.0.
    """
    x = (p - threshold) / (1.0 - threshold + 1e-10)
    x = np.clip(x, 0, 1)
    return x ** 2

def perform_bucket_analysis(df: pd.DataFrame, score_col: str, returns_col: str, n_buckets: int = 5):
    temp_df = df.dropna(subset=[score_col, returns_col]).copy()
    if temp_df.empty: return pd.Series()
    
    temp_df['bucket'] = pd.qcut(temp_df[score_col], n_buckets, labels=False, duplicates='drop')
    bucket_returns = temp_df.groupby('bucket')[returns_col].mean()
    
    logger.info(f"== BUCKET ANALYSIS VALIDATION ({score_col}) ==")
    logger.info(f"Bucket Mean Returns:\n{bucket_returns}")
    return bucket_returns

def run_backtest_high_conviction_v12(df: pd.DataFrame, probs: np.array,
                                    cost=0.0002, slippage=0.0001, 
                                    top_n=3,
                                    min_hold_bars=3, 
                                    spread_filter_threshold=0.001):
    """
    High-Conviction Alpha Backtester V12: Capital Allocation Engine.
    Concentrates only on Top Bucket (4) and Top N signals per timestamp.
    """
    logger.info(f"Starting High-Conviction Alpha Backtest (V12: Top-Bucket & Top-{top_n} Selection)...")
    
    bt_df = df.copy()
    bt_df['p_s'], bt_df['p_f'], bt_df['p_l'] = probs[:, 0], probs[:, 1], probs[:, 2]
    
    # 1. RANKING ENGINE
    bt_df['score_l'] = bt_df['p_l'] - bt_df['p_s']
    
    # 2. BUCKET & TOP-N SELECTION
    bt_df['bucket_l'] = pd.qcut(bt_df['score_l'], 5, labels=False, duplicates='drop')
    bt_df['rank_ts'] = bt_df.groupby('datetime')['score_l'].rank(ascending=False)
    
    # MANDATORY FILTERS (Fix 1/2)
    c_bucket_4 = (bt_df['bucket_l'] == 4)
    c_top_n = (bt_df['rank_ts'] <= top_n)
    
    # 3. EXPECTANCY & TIGHT SPREAD (Fix 4/5)
    bt_df['expectancy_l'] = (bt_df['p_l'] * 2.2 * bt_df['atr_pct']) - (bt_df['p_s'] * 1.0 * bt_df['atr_pct'])
    c_exp_l = (bt_df['expectancy_l'] > 0)
    c_spread = (bt_df['spread_pct'] < spread_filter_threshold)
    
    # 4. TRIGGER CONSTRUCTION
    bt_df['trigger'] = 0
    bt_df.loc[c_bucket_4 & c_top_n & c_exp_l & c_spread, 'trigger'] = 1
    
    # BACKTEST LOOP
    bt_df = bt_df.sort_values(['symbol', 'datetime'])
    final_positions = []
    
    for _, grp in bt_df.groupby('symbol'):
        triggers = grp['trigger'].values
        p_l = grp['p_l'].values
        
        pos, hold_count, entry_size = 0.0, 0, 0.0
        symbol_pos = np.zeros(len(triggers))
        
        for i in range(len(triggers)):
            t = triggers[i]
            
            if pos == 1:
                # SIMPLE EXIT: Signal Weakening
                if p_l[i] < 0.45:
                    if hold_count >= min_hold_bars:
                        pos = 0
                        entry_size = 0
                hold_count += 1
            else:
                # ENTRY & QUADRATIC SIZING (Fix 3)
                if t == 1:
                    pos = 1
                    hold_count = 1
                    entry_size = size_fn_v12(p_l[i], threshold=0.5)
            
            symbol_pos[i] = pos * entry_size
            
        final_positions.extend(symbol_pos)
        
    bt_df['persistent_pos'] = final_positions
    bt_df['actual_pos'] = bt_df.groupby('symbol')['persistent_pos'].shift(1).fillna(0)
    
    # RETURNS
    bt_df['net_ret'] = (bt_df['actual_pos'] * bt_df.groupby('symbol')['close'].pct_change().fillna(0)) - \
                      (bt_df.groupby('symbol')['actual_pos'].diff().abs().fillna(0) * (cost+slippage))
    
    # METRICS
    p_rets = bt_df.groupby('datetime')['net_ret'].mean()
    metrics = compute_alpha_metrics(p_rets)
    
    bt_df['pos_sign'] = np.sign(bt_df['actual_pos'])
    num_trades = len(bt_df[bt_df.groupby('symbol')['pos_sign'].diff().fillna(0) != 0])
    avg_pos_size = bt_df[bt_df['actual_pos'] != 0]['actual_pos'].abs().mean()
    
    logger.info(f"== V12 HIGH-CONVICTION METRICS (N_TRADES={num_trades}) ==")
    logger.info(f"Average Position Size: {avg_pos_size:.4f}")
    for k, v in metrics.items():
        if 'Sharpe' in k or 'Factor' in k: logger.info(f"{k}: {v:.2f}")
        else: logger.info(f"{k}: {v:.2%}")
    
    return bt_df, p_rets, metrics
