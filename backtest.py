import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def compute_v7_metrics(portfolio_rets: pd.Series):
    if portfolio_rets.empty: return {'Sharpe': -9}
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
    profit_factor = gross_wins / abs(gross_losses) if gross_losses != 0 else np.nan
    
    return {
        'Total Return': total_ret,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max Drawdown': max_dd,
        'Profit Factor': profit_factor,
        'Win Rate': (portfolio_rets > 0).mean()
    }

def run_backtest_elite_v7(df: pd.DataFrame, probs: np.array,
                         cost=0.0002, slippage=0.0001, 
                         long_threshold=0.60, short_threshold=0.62, 
                         conf_gap_threshold=0.12, strength_percentile=0.25,
                         min_hold_bars=3, 
                         vol_filter_quantile=0.4, spread_filter_threshold=0.0012):
    """
    Elite Production Backtester V7: Quality-Based Filtering & Expectancy Alignment.
    Implements Strength Percentiles, Confidence Gaps, and Microstructure Confirmation.
    Follows Production-Grade Alpha selection standards.
    """
    logger.info("Starting Elite Backtest (V7: Trade Quality & Alpha Edge Optimization)...")
    
    bt_df = df.copy()
    bt_df['p_s'], bt_df['p_f'], bt_df['p_l'] = probs[:, 0], probs[:, 1], probs[:, 2]
    
    # 1. CORE QUALITY METRICS (ISSUE 1 & 2)
    # Strength = Max Predicted Probability
    bt_df['strength'] = np.max(probs, axis=1)
    
    # Confidence Gap = Top1 - Top2
    sorted_probs = np.sort(probs, axis=1)
    bt_df['conf_gap'] = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    # Strength Threshold: Remove lowest 25% (ISSUE 1)
    strength_floor = bt_df['strength'].quantile(strength_percentile)
    
    # 2. EXPECTANCY FILTER (ISSUE 3)
    # E[R] = P(Long)*2.0*ATR - P(Short)*1.0*ATR (V5 Asymmetric Labeling)
    bt_df['expectancy_l'] = bt_df['p_l'] * (2.0 * bt_df['atr_pct']) - (1.0 - bt_df['p_l']) * (1.0 * bt_df['atr_pct'])
    bt_df['expectancy_s'] = bt_df['p_s'] * (2.0 * bt_df['atr_pct']) - (1.0 - bt_df['p_s']) * (1.0 * bt_df['atr_pct'])
    
    # 3. BASE SIGNAL & ELITE FILTERS
    # Regime: Trend Check or Volatility Check
    vol_floor = bt_df['atr_pct'].quantile(vol_filter_quantile)
    bt_df['regime_ok'] = ((bt_df['adx_14'] > 25) | (bt_df['atr_pct'] > vol_floor)).astype(int)
    
    # Microstructure Agreement (ISSUE 5)
    l_micro = (bt_df['ofi'] > 0) & (bt_df['micro_mid_diff'] > 0)
    s_micro = (bt_df['ofi'] < 0) & (bt_df['micro_mid_diff'] < 0)
    
    # Unified Trigger (ISSUE 4 & 5)
    bt_df['trigger'] = 0
    # Long Criteria
    l_mask = (bt_df['p_l'] > long_threshold) & (bt_df['conf_gap'] > conf_gap_threshold) & \
             (bt_df['strength'] > strength_floor) & (bt_df['expectancy_l'] > 0) & \
             bt_df['regime_ok'] & l_micro & (bt_df['spread_pct'] < spread_filter_threshold)
    bt_df.loc[l_mask, 'trigger'] = 1
    
    # Short Criteria
    s_mask = (bt_df['p_s'] > short_threshold) & (bt_df['conf_gap'] > conf_gap_threshold) & \
             (bt_df['strength'] > strength_floor) & (bt_df['expectancy_s'] > 0) & \
             bt_df['regime_ok'] & s_micro & (bt_df['spread_pct'] < spread_filter_threshold)
    bt_df.loc[s_mask, 'trigger'] = -1
    
    # 4. POSITION PERSISTENCE & MIN-HOLD (MAINTAIN V6 LOGIC)
    bt_df = bt_df.sort_values(['symbol', 'datetime'])
    final_positions = []
    
    for _, grp in bt_df.groupby('symbol'):
        triggers = grp['trigger'].values
        probs_l = grp['p_l'].values
        probs_s = grp['p_s'].values
        
        pos = 0.0          # Current position (-1, 0, 1)
        hold_count = 0     # Bars held
        entry_size = 0.0   # Confidence scaled
        
        symbol_pos = np.zeros(len(triggers))
        
        for i in range(len(triggers)):
            t = triggers[i]
            
            if pos == 1:
                # Contrary signal or extreme neutral probs exit
                if t == -1 or probs_s[i] > short_threshold:
                    if hold_count >= min_hold_bars:
                        pos = -1 if t == -1 else 0
                        entry_size = probs_s[i] if t == -1 else 0
                        hold_count = 1 if t == -1 else 0
                else: hold_count += 1
            elif pos == -1:
                # Contrary signal or extreme neutral probs exit
                if t == 1 or probs_l[i] > long_threshold:
                    if hold_count >= min_hold_bars:
                        pos = 1 if t == 1 else 0
                        entry_size = probs_l[i] if t == 1 else 0
                        hold_count = 1 if t == 1 else 0
                else: hold_count += 1
            else:
                if t != 0:
                    pos = t
                    entry_size = probs_l[i] if pos > 0 else probs_s[i]
                    hold_count = 1
                    
            symbol_pos[i] = pos * entry_size if pos != 0 else 0
            
        final_positions.extend(symbol_pos)
        
    bt_df['persistent_pos'] = final_positions
    bt_df['actual_pos'] = bt_df.groupby('symbol')['persistent_pos'].shift(1).fillna(0)
    
    # 5. RETURNS & COSTS
    bt_df['bar_ret'] = bt_df.groupby('symbol')['close'].pct_change().fillna(0)
    bt_df['net_ret'] = (bt_df['actual_pos'] * bt_df['bar_ret']) - \
                      (bt_df.groupby('symbol')['actual_pos'].diff().abs().fillna(0) * (cost + slippage))
    
    # 6. METRICS
    p_rets = bt_df.groupby('datetime')['net_ret'].mean()
    metrics = compute_v7_metrics(p_rets)
    
    # Count sign-changing unique entries
    bt_df['pos_sign'] = np.sign(bt_df['actual_pos'])
    num_trades = len(bt_df[bt_df.groupby('symbol')['pos_sign'].diff().fillna(0) != 0])
    
    logger.info(f"== V7 ELITE BACKTEST METRICS (N_TRADES={num_trades}) ==")
    for k, v in metrics.items():
        if 'Sharpe' in k or 'Factor' in k: logger.info(f"{k}: {v:.2f}")
        else: logger.info(f"{k}: {v:.2%}")
        
    os.makedirs('logs', exist_ok=True)
    bt_df[bt_df.groupby('symbol')['pos_sign'].diff().fillna(0) != 0].to_csv('logs/trade_log_v7.csv', index=False)
    
    return bt_df, p_rets, metrics
