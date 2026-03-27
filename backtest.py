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

def run_backtest_capital_allocation_v13(df: pd.DataFrame, probs: np.array,
                                        cost=0.0002, slippage=0.0001, 
                                        top_n=2,
                                        cooldown_bars=3,
                                        min_hold_bars=2, 
                                        spread_filter_threshold=0.001):
    """
    Capital Allocation Backtester V13: Portfolio Normalization & Conviction Scaling.
    Optimized for high returns and capital efficiency (>100% exposure).
    """
    logger.info(f"Starting V13 Capital Allocation Backtest (Top-{top_n} Selection)...")
    
    bt_df = df.copy()
    bt_df['p_s'], bt_df['p_f'], bt_df['p_l'] = probs[:, 0], probs[:, 1], probs[:, 2]
    
    # 1. ALPHA RANKING & BUCKETS
    bt_df['score_l'] = bt_df['p_l'] - bt_df['p_s']
    bt_df['bucket_l'] = pd.qcut(bt_df['score_l'], 5, labels=False, duplicates='drop')
    bt_df['rank_ts'] = bt_df.groupby('datetime')['score_l'].rank(ascending=False)
    
    # 2. FILTERS (Fix 3: Top-N reduced to 2 for count control)
    c_bucket_4 = (bt_df['bucket_l'] == 4)
    c_top_n = (bt_df['rank_ts'] <= top_n)
    bt_df['expectancy_l'] = (bt_df['p_l'] * 2.2 * bt_df['atr_pct']) - (bt_df['p_s'] * 1.0 * bt_df['atr_pct'])
    c_exp_l = (bt_df['expectancy_l'] > 0)
    c_spread = (bt_df['spread_pct'] < spread_filter_threshold)
    
    # 3. TRIGGER (Signal Identification)
    bt_df['signal'] = (c_bucket_4 & c_top_n & c_exp_l & c_spread).astype(int)
    
    # 4. CROSS-SECTIONAL BACKTEST LOOP
    bt_df = bt_df.sort_values(['datetime', 'symbol'])
    symbols = bt_df['symbol'].unique()
    datetimes = bt_df['datetime'].unique()
    
    # State tracking
    current_pos = {s: 0.0 for s in symbols}
    entry_size = {s: 0.0 for s in symbols}
    hold_count = {s: 0 for s in symbols}
    last_exit_idx = {s: -10 for s in symbols}
    
    final_pos_list = []
    
    # Pre-pivot for speed
    pivot_signal = bt_df.pivot(index='datetime', columns='symbol', values='signal').fillna(0)
    pivot_pl = bt_df.pivot(index='datetime', columns='symbol', values='p_l').fillna(0)
    pivot_score = bt_df.pivot(index='datetime', columns='symbol', values='score_l').fillna(0)
    
    logger.info("Processing cross-sectional allocation...")
    
    for t_idx, dt in enumerate(datetimes):
        # 1. Identify Candidate Entries & Active Holds
        candidates = []
        active_weights = {}
        
        for s in symbols:
            sig = pivot_signal.at[dt, s]
            p_l = pivot_pl.at[dt, s]
            score = pivot_score.at[dt, s]
            
            # EXIT CHECK
            if current_pos[s] > 0:
                # Minimum hold enforcement (Fix 5)
                if hold_count[s] >= min_hold_bars:
                    # Weakening exit or Reversal
                    if p_l < 0.45 or sig == -1: # sig won't be -1 in long-only
                        current_pos[s] = 0.0
                        last_exit_idx[s] = t_idx
                        hold_count[s] = 0
                    else:
                        hold_count[s] += 1
                else:
                    hold_count[s] += 1
            
            # ENTRY CHECK (Fix 4: Cooldown)
            if current_pos[s] == 0:
                if sig == 1 and (t_idx - last_exit_idx[s]) > cooldown_bars:
                    # VALID ENTRY CANDIDATE
                    # Fix 2: Strong Scaling
                    x = np.clip((p_l - 0.5) / 0.5, 0, 1)
                    raw_size = max(0.2, x ** 2)
                    # Fix 6: Conviction Scaling
                    conviction = max(0.01, score)
                    active_weights[s] = raw_size * conviction
                else:
                    pass
            else:
                # PERSIST EXISTING HOLD
                # We still re-weight holds cross-sectionally
                x = np.clip((p_l - 0.5) / 0.5, 0, 1)
                raw_size = max(0.2, x ** 2)
                conviction = max(0.01, score)
                active_weights[s] = raw_size * conviction

        # 2. PORTFOLIO NORMALIZATION (Fix 1)
        if active_weights:
            total_weight = sum(active_weights.values())
            for s, w in active_weights.items():
                normalized_size = w / total_weight
                current_pos[s] = normalized_size
        
        # 3. Store results
        for s in symbols:
            final_pos_list.append({'datetime': dt, 'symbol': s, 'persistent_pos': current_pos[s]})

    # Re-merge results
    pos_df = pd.DataFrame(final_pos_list)
    bt_df = bt_df.merge(pos_df, on=['datetime', 'symbol'])
    
    bt_df['actual_pos'] = bt_df.groupby('symbol')['persistent_pos'].shift(1).fillna(0)
    
    # RETURNS & AUDIT
    bt_df['net_ret'] = (bt_df['actual_pos'] * bt_df.groupby('symbol')['close'].pct_change().fillna(0)) - \
                      (bt_df.groupby('symbol')['actual_pos'].diff().abs().fillna(0) * (cost+slippage))
    
    # METRICS
    p_rets = bt_df.groupby('datetime')['net_ret'].sum() # Sum because it's a portfolio at 100% total weight
    metrics = compute_alpha_metrics(p_rets)
    
    bt_df['pos_sign'] = np.sign(bt_df['actual_pos'])
    num_trades = len(bt_df[bt_df.groupby('symbol')['pos_sign'].diff().fillna(0) != 0])
    avg_pos_size = bt_df[bt_df['actual_pos'] != 0]['actual_pos'].mean()
    utilization = bt_df.groupby('datetime')['actual_pos'].sum().mean()
    
    logger.info(f"== V13 CAPITAL ALLOCATION METRICS (N_TRADES={num_trades}) ==")
    logger.info(f"Average Position Size: {avg_pos_size:.4f}")
    logger.info(f"Capital Utilization: {utilization:.2%}")
    for k, v in metrics.items():
        if 'Sharpe' in k or 'Factor' in k: logger.info(f"{k}: {v:.2f}")
        else: logger.info(f"{k}: {v:.2%}")
        
    return bt_df, p_rets, metrics
