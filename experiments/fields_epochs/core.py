"""Only LR and repeated exposure change from the frozen fields-SFT recipe."""
import math
from experiments.fields_sft.train import recipe as parent_recipe

RATES={'A':1e-4,'B':5e-5}

def schedule(ids):
    if len(ids)!=600 or len(set(ids))!=600:raise ValueError('Exactly600 unique original examples required')
    return list(ids)*3

def recipe(arm,fingerprint,data_sha,cap):
    if arm not in RATES:raise ValueError('Unknown arm')
    return dict(parent_recipe('F',fingerprint,data_sha,cap),arm=arm,learning_rate=RATES[arm],max_microsteps=1800,checkpoints=list(range(200,1801,200)))

def check_lr(actual,expected):
    if not math.isclose(float(actual),expected,rel_tol=1e-6,abs_tol=0):raise ValueError('Restored optimizer learning rate differs from recipe')

def require_grid(cells):
    expected={f'{a}-{e}-{s}' for a in RATES for e in [1,2,3] for s in ['dev','seen']}
    if set(cells)!=expected or any(c['n']!=120 for c in cells.values()):raise ValueError('Incomplete epoch/split grid')
