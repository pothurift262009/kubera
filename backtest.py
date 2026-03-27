import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def compute_v2_risk_metrics(portfolio_rets: pd.Series):
    """
    Advanced Risk/Reward Metrics: Sharpe, Sortino, Calmar, Max Drawdown Duration.
    """
    cum_rets = (1 + portfolio_rets).cumprod()
    total_ret = cum_rets.iloc[-1] - 1
    
    # 5-min bars: 75 / day * 252 days / year
    bars_per_year = 75 * 252
    ann_ret = (1 + total_ret)**(bars_per_year / len(portfolio_rets)) - 1
    
    # Corrected Sharpe/Sortino (Annualized)
    rf = 0.05 / bars_per_year # Libor approximation
    excess = portfolio_rets - rf
    sharpe = np.sqrt(bars_per_year) * excess.mean() / (excess.std() + 1e-10)
    
    downside = portfolio_rets[portfolio_rets < 0]
    sortino = np.sqrt(bars_per_year) * excess.mean() / (downside.std() + 1e-10)
    
    # Max Drawdown & Duration
    highwater = cum_rets.cummax()
    drawdown = (cum_rets / highwater) - 1
    max_dd = drawdown.min()
    
    # Calmar Ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    # Drawdown Duration
    is_dd = (drawdown < 0).astype(int)
    dd_groups = (is_dd != is_dd.shift()).cumsum()
    dd_runs = is_dd.groupby(dd_groups).cumsum()
    max_dd_duration = int(dd_runs.max()) if not dd_runs.empty else 0
    
    # Win Rate
    win_rate = (portfolio_rets > 0).mean()
    
    return {
        'Total Return': total_ret,
        'Ann. Return': ann_ret,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max Drawdown': max_dd,
        'Max DD Duration': max_dd_duration,
        'Calmar': calmar,
        'Win Rate': win_rate
    }

def run_backtest_elite_v2(df: pd.DataFrame, probs: np.array,
                         cost=0.0007, slippage=0.0003, prob_threshold=0.60,
                         vol_filter_quantile=0.1, spread_filter_threshold=0.002):
    """
    Elite Production Backtester: High-Impact Trading Filtering & Confidence-Based Entry.
    Includes Execution Delay, Persistence, and Vol-Regime Filters.
    """
    logger.info(f"Starting Elite Backtest (V2) with Prob-Threshold={prob_threshold}...")
    
    bt_df = df.copy()
    bt_df['prob_short'] = probs[:, 0]
    bt_df['prob_flat'] = probs[:, 1]
    bt_df['prob_long'] = probs[:, 2]
    
    # 1. Trading Logic: Confidence-Based Entry
    bt_df['raw_signal'] = 0
    bt_df.loc[bt_df['prob_long'] > prob_threshold, 'raw_signal'] = 1
    bt_df.loc[bt_df['prob_short'] > prob_threshold, 'raw_signal'] = -1
    
    # 2. Filters: Volatility and Spread
    # Avoid wide spreads: avoid liquidity trap
    bt_df['spread_filter'] = (bt_df['spread_pct'] < spread_filter_threshold).astype(int)
    
    # Avoid extremely low volatility sessions (noise)
    vol_floor = bt_df['atr_pct'].quantile(vol_filter_quantile)
    bt_df['vol_filter'] = (bt_df['atr_pct'] > vol_floor).astype(int)
    
    # Combined Decision
    bt_df['target_pos'] = bt_df['raw_signal'] * bt_df['spread_filter'] * bt_df['vol_filter']
    
    # 3. Position Persistence & Smoothing (Avoid jumping every bar)
    # HOLD positions if signal becomes flat (confidence drops) but DO NOT exit immediately 
    # unless confidence in the opposite direction is strong. 
    # For now, we apply persistent signals to reduce turnover.
    bt_df['final_signal'] = bt_df.groupby('symbol')['target_pos'].transform(lambda x: x.replace(0, np.nan).ffill().fillna(0))
    
    # 4. EXECUTION DELAY: Decision at T is executed at T+1
    bt_df['actual_pos'] = bt_df.groupby('symbol')['final_signal'].shift(1).fillna(0)
    
    # 5. Returns & Multi-Dimensional Costs
    bt_df['bar_ret'] = bt_df.groupby('symbol')['close'].pct_change().fillna(0)
    bt_df['strat_ret_gross'] = bt_df['actual_pos'] * bt_df['bar_ret']
    
    # Dynamic Turnover Cost (Execution Cost)
    bt_df['turnover'] = bt_df.groupby('symbol')['actual_pos'].diff().abs().fillna(0)
    bt_df['execution_cost'] = bt_df['turnover'] * (cost + slippage)
    
    bt_df['net_ret'] = bt_df['strat_ret_gross'] - bt_df['execution_cost']
    
    # 6. Global Portfolio Performance
    portfolio_rets = bt_df.groupby('datetime')['net_ret'].mean()
    metrics = compute_v2_risk_metrics(portfolio_rets)
    
    # 7. Audit & Reporting
    trades = bt_df[bt_df['turnover'] > 0].copy()
    num_trades = len(trades)
    avg_trade_ret = trades['net_ret'].mean() if num_trades > 0 else 0
    
    logger.info(f"== V2 BACKTEST METRICS (N_TRADES={num_trades}) ==")
    for k, v in metrics.items():
        if 'Sharpe' in k or 'Sortino' in k or 'Calmar' in k:
            logger.info(f"{k}: {v:.2f}")
        else:
            logger.info(f"{k}: {v:.2%}")
    logger.info(f"Average Return Per Trade: {avg_trade_ret:.4%}")
    
    os.makedirs('logs', exist_ok=True)
    trades[['datetime', 'symbol', 'close', 'actual_pos', 'net_ret', 'prob_short', 'prob_long']].to_csv('logs/trade_log_v2.csv', index=False)
    
    return bt_df, portfolio_rets, metrics
