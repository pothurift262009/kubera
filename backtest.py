import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def compute_v9_metrics(portfolio_rets: pd.Series):
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

def run_backtest_elite_v9(df: pd.DataFrame, probs: np.array,
                         cost=0.0002, slippage=0.0001, 
                         long_threshold=0.58, short_threshold=0.60, 
                         conf_gap_threshold=0.10, strength_percentile=0.20,
                         min_hold_bars=3, 
                         vol_filter_quantile=0.4, spread_filter_threshold=0.0012):
    """
    Elite Production Backtester V9: Scoring System & High-Precision Sizing.
    Follows Senior Quant Engineer spec for PF > 1.2 and Sharpe > 1.2.
    Fixes expectancy bugs, brittle filters, and premature exits.
    """
    logger.info("Starting Elite Backtest (V9: Scoring Engine & Sizing Overhaul)...")
    
    bt_df = df.copy()
    # Classes: 0=Short, 1=Flat, 2=Long
    bt_df['p_s'], bt_df['p_f'], bt_df['p_l'] = probs[:, 0], probs[:, 1], probs[:, 2]
    
    # 1. FIX EXPECTANCY CALCULATION (ISSUE 1)
    # E[R] = P(long)*TP - P(short)*SL (Corrected for 3-class model)
    # Using V8 labeling specs: TP=2.2, SL=1.0
    bt_df['expectancy_l'] = (bt_df['p_l'] * 2.2 * bt_df['atr_pct']) - (bt_df['p_s'] * 1.0 * bt_df['atr_pct'])
    bt_df['expectancy_s'] = (bt_df['p_s'] * 2.2 * bt_df['atr_pct']) - (bt_df['p_l'] * 1.0 * bt_df['atr_pct'])
    
    # 2. SCORING SYSTEM (ISSUE 3)
    bt_df['strength'] = np.max(probs, axis=1)
    sorted_p = np.sort(probs, axis=1)
    bt_df['conf_gap'] = sorted_p[:, -1] - sorted_p[:, -2]
    strength_floor = bt_df['strength'].quantile(strength_percentile)
    vol_floor = bt_df['atr_pct'].quantile(vol_filter_quantile)
    
    # Pre-compute Scoring Components
    bt_df['c_regime'] = ((bt_df['adx_14'] > 25) | (bt_df['atr_pct'] > vol_floor)).astype(int)
    bt_df['c_spread'] = (bt_df['spread_pct'] < spread_filter_threshold).astype(int)
    bt_df['c_gap'] = (bt_df['conf_gap'] > conf_gap_threshold).astype(int)
    bt_df['c_strength'] = (bt_df['strength'] > strength_floor).astype(int)
    
    # Long Scoring
    l_micro = ((bt_df['ofi'] > 0) & (bt_df['micro_mid_diff'] > 0)).astype(int)
    bt_df['score_l'] = (bt_df['p_l'] > long_threshold).astype(int) + \
                       bt_df['c_gap'] + bt_df['c_strength'] + \
                       (bt_df['expectancy_l'] > 0).astype(int) + \
                       bt_df['c_regime'] + l_micro + bt_df['c_spread']
                       
    # Short Scoring
    s_micro = ((bt_df['ofi'] < 0) & (bt_df['micro_mid_diff'] < 0)).astype(int)
    bt_df['score_s'] = (bt_df['p_s'] > short_threshold).astype(int) + \
                       bt_df['c_gap'] + bt_df['c_strength'] + \
                       (bt_df['expectancy_s'] > 0).astype(int) + \
                       bt_df['c_regime'] + s_micro + bt_df['c_spread']
    
    bt_df['trigger'] = 0
    # Take trade if Score >= 3 (Optimal for trade frequency vs quality balance)
    bt_df.loc[bt_df['score_l'] >= 3, 'trigger'] = 1
    bt_df.loc[bt_df['score_s'] >= 3, 'trigger'] = -1
    
    # 3. POSITION PERSISTENCE & IMPROVED EXIT (ISSUE 4)
    # Buffer Exit: entry_threshold + 0.05 to avoid premature profit taking
    l_exit_thresh = long_threshold + 0.05
    s_exit_thresh = short_threshold + 0.05
    max_hold_limit = 6 # Corresponds to LABEL_CONFIG max_bars
    
    bt_df = bt_df.sort_values(['symbol', 'datetime'])
    final_positions = []
    
    for _, grp in bt_df.groupby('symbol'):
        triggers = grp['trigger'].values
        p_l = grp['p_l'].values
        p_s = grp['p_s'].values
        
        pos = 0.0
        hold_count = 0
        entry_size = 0.0
        
        symbol_pos = np.zeros(len(triggers))
        
        for i in range(len(triggers)):
            t = triggers[i]
            
            if pos == 1:
                # EXIT/REVERSAL with BUFFER and MAX_HOLD (ISSUE 4)
                if t == -1 or p_s[i] > s_exit_thresh or hold_count >= max_hold_limit:
                    if hold_count >= min_hold_bars:
                        pos = -1 if t == -1 else 0
                        entry_size = max(0, p_s[i] - short_threshold) / (1.0 - short_threshold + 1e-10) if pos == -1 else 0
                        hold_count = 1 if t == -1 else 0
                else: hold_count += 1
            elif pos == -1:
                # EXIT/REVERSAL with BUFFER
                if t == 1 or p_l[i] > l_exit_thresh or hold_count >= max_hold_limit:
                    if hold_count >= min_hold_bars:
                        pos = 1 if t == 1 else 0
                        entry_size = max(0, p_l[i] - long_threshold) / (1.0 - long_threshold + 1e-10) if pos == 1 else 0
                        hold_count = 1 if t == 1 else 0
                else: hold_count += 1
            else:
                # 4. SCALED POSITION SIZING (ISSUE 2)
                if t != 0:
                    pos = t
                    hold_count = 1
                    if pos == 1:
                        entry_size = max(0, p_l[i] - long_threshold) / (1.0 - long_threshold + 1e-10)
                    else:
                        entry_size = max(0, p_s[i] - short_threshold) / (1.0 - short_threshold + 1e-10)
            
            symbol_pos[i] = pos * entry_size if pos != 0 else 0
            
        final_positions.extend(symbol_pos)
        
    bt_df['persistent_pos'] = final_positions
    bt_df['actual_pos'] = bt_df.groupby('symbol')['persistent_pos'].shift(1).fillna(0)
    
    # 5. RETURNS & AUDIT
    bt_df['net_ret'] = (bt_df['actual_pos'] * bt_df.groupby('symbol')['close'].pct_change().fillna(0)) - \
                      (bt_df.groupby('symbol')['actual_pos'].diff().abs().fillna(0) * (cost+slippage))
    
    # 6. METRICS
    p_rets = bt_df.groupby('datetime')['net_ret'].mean()
    metrics = compute_v9_metrics(p_rets)
    
    bt_df['pos_sign'] = np.sign(bt_df['actual_pos'])
    num_trades = len(bt_df[bt_df.groupby('symbol')['pos_sign'].diff().fillna(0) != 0])
    
    logger.info(f"== V9 ELITE BACKTEST METRICS (N_TRADES={num_trades}) ==")
    for k, v in metrics.items():
        if 'Sharpe' in k or 'Factor' in k: logger.info(f"{k}: {v:.2f}")
        else: logger.info(f"{k}: {v:.2%}")
        
    os.makedirs('logs', exist_ok=True)
    bt_df[bt_df.groupby('symbol')['pos_sign'].diff().fillna(0) != 0].to_csv('logs/trade_log_v9.csv', index=False)
    
    return bt_df, p_rets, metrics
