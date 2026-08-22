"""
Kubera Configuration — Single source of truth for all parameters.
"""

# ── Data ─────────────────────────────────────────────────────────
DATA_FILE = "nifty50_1min_5years.parquet"
RESAMPLE_FREQ = "5min"

# ── Labeling ─────────────────────────────────────────────────────
PT_PCT = 3.0
SL_PCT = 1.2
HORIZON_BARS = 24     # 12 bars × 5min = 60 minutes

# ── Model ────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
PURGE_BARS = 24       # Must equal HORIZON_BARS to prevent label leakage

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 42,
    'eval_metric': 'aucpr',
    'scale_pos_weight': 3.25,
}

# ── Costs ────────────────────────────────────────────────────────
# Zerodha MIS round-trip: brokerage(~0.04%) + STT(~0.025%) + exchange(~0.005%) + slippage
TRANSACTION_COST_PCT = 0.12
SLIPPAGE_PCT = 0.03
TOTAL_COST_PCT = TRANSACTION_COST_PCT + SLIPPAGE_PCT  # 0.15%

# ── Trading ──────────────────────────────────────────────────────
MAX_POSITIONS = 3
POSITION_SIZE_INR = 50000
MAX_HOLD_MINS = 60
NIFTY_FILTER = -0.5   # Skip BUY if Nifty day return < this

LONG_THRESHOLD = 0.10
SHORT_THRESHOLD = 0.10
LUNCH_LONG_THRESHOLD = 0.10

# ── Time Gates ───────────────────────────────────────────────────
LUNCH_START = "11:30"
LUNCH_END = "13:30"
NO_NEW_ENTRY_HOUR = 15.25
NO_NEW_ENTRY_MIN = 30

# ── Risk Management (Live Trading) ───────────────────────────────
MAX_DAILY_LOSS_PCT = -2.0     # Circuit breaker: stop trading if daily PnL hits this
MAX_SECTOR_POSITIONS = 1      # Max simultaneous positions per sector

# ── Symbols ──────────────────────────────────────────────────────
SYMBOLS = {
    'ADANIENT':   6401,
    'ADANIPORTS': 3861249,
    'AXISBANK':   1510401,
    'BAJAJFINSV': 4268801,
    'BAJFINANCE': 81153,
    'HDFCBANK':   341249,
    'HINDALCO':   348929,
    'ICICIBANK':  1270529,
    'INDUSINDBK': 1346049,
    'INFY':       408065,
    'JSWSTEEL':   3001089,
    'KOTAKBANK':  492033,
    'LT':         2939649,
    'M&M':        519937,
    'MARUTI':     2815745,
    'RELIANCE':   738561,
    'SBIN':       779521,
    'TATASTEEL':  895745,
    'TECHM':      3465729,
    'TITAN':      897537,
    'WIPRO':      969473,
    'APOLLOHOSP': 40193,
    'ASIANPAINT': 60417,
    'BAJAJ-AUTO': 4267265,
    'BEL':        98049,
    'BHARTIARTL': 2714625,
    'BPCL':       134657,
    'BRITANNIA':  140033,
    'CIPLA':      177665,
    'COALINDIA':  5215745,
    'DIVISLAB':   2800641,
    'DRREDDY':    225537,
    'EICHERMOT':  232961,
    'GRASIM':     315393,
    'HCLTECH':    1850625,
    'HDFCLIFE':   119553,
    'HEROMOTOCO': 345089,
    'HINDUNILVR': 356865,
    'ITC':        424961,
    'NESTLEIND':  4598529,
    'NTPC':       2977281,
    'ONGC':       633601,
    'POWERGRID':  3834113,
    'SBILIFE':    5582849,
    'SHRIRAMFIN': 1102337,
    'SUNPHARMA':  857857,
    'TATACONSUM': 878593,
    'TATAMOTORS': 884737,
    'TCS':        2953217,
    'ULTRACEMCO': 2952193,
}

# ── Sector Mapping (for correlation/diversification control) ─────
SYMBOL_SECTOR = {
    # Banking & Financial Services
    'AXISBANK':   'BANKING',
    'HDFCBANK':   'BANKING',
    'ICICIBANK':  'BANKING',
    'INDUSINDBK': 'BANKING',
    'KOTAKBANK':  'BANKING',
    'SBIN':       'BANKING',
    # NBFC / Financial Services
    'BAJFINANCE': 'NBFC',
    'BAJAJFINSV': 'NBFC',
    'SHRIRAMFIN': 'NBFC',
    'HDFCLIFE':   'NBFC',
    'SBILIFE':    'NBFC',
    # IT
    'INFY':       'IT',
    'TCS':        'IT',
    'HCLTECH':    'IT',
    'TECHM':      'IT',
    'WIPRO':      'IT',
    # Pharma
    'CIPLA':      'PHARMA',
    'DRREDDY':    'PHARMA',
    'DIVISLAB':   'PHARMA',
    'SUNPHARMA':  'PHARMA',
    'APOLLOHOSP': 'PHARMA',
    # Auto
    'MARUTI':     'AUTO',
    'TATAMOTORS': 'AUTO',
    'M&M':        'AUTO',
    'BAJAJ-AUTO': 'AUTO',
    'EICHERMOT':  'AUTO',
    'HEROMOTOCO': 'AUTO',
    # Metals & Mining
    'HINDALCO':   'METALS',
    'JSWSTEEL':   'METALS',
    'TATASTEEL':  'METALS',
    'COALINDIA':  'METALS',
    # Oil & Gas / Energy
    'RELIANCE':   'ENERGY',
    'BPCL':       'ENERGY',
    'ONGC':       'ENERGY',
    'NTPC':       'ENERGY',
    'POWERGRID':  'ENERGY',
    # Infrastructure / Conglomerate
    'ADANIENT':   'INFRA',
    'ADANIPORTS': 'INFRA',
    'LT':         'INFRA',
    'GRASIM':     'INFRA',
    'ULTRACEMCO': 'INFRA',
    'BEL':        'INFRA',
    # FMCG / Consumer
    'HINDUNILVR': 'FMCG',
    'ITC':        'FMCG',
    'NESTLEIND':  'FMCG',
    'BRITANNIA':  'FMCG',
    'TATACONSUM': 'FMCG',
    'TITAN':      'FMCG',
    'ASIANPAINT': 'FMCG',
    # Telecom
    'BHARTIARTL': 'TELECOM',
}