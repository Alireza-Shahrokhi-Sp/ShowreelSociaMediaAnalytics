"""Inject Section J: Target distribution over time."""
import json, pathlib, uuid

NB = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

def md_cell(src, cid=None):
    return {"cell_type": "markdown", "id": cid or uuid.uuid4().hex[:8],
            "metadata": {}, "source": src}

def code_cell(src, cid=None):
    return {"cell_type": "code", "id": cid or uuid.uuid4().hex[:8],
            "metadata": {}, "execution_count": None, "outputs": [], "source": src}

MD_J = """\
## J. Target Distribution Over Time

Where are comments directed -- at the creator, their content/work, appearance, other users,
or products -- and how has that mix shifted across the 10-year archive?\
"""

CODE_J_LOAD = """\
# J.0  Load comment-level data with target + timestamp.
import pandas as pd, numpy as np, matplotlib.pyplot as plt, pathlib

BASE = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab')
OUT_J = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/outputs')

tdf = pd.read_parquet(
    BASE / 'Ali/outputs/stage2_sentiment/sentiment_instagram.parquet',
    columns=['timestamp','target','sentiment','toxicity'])

tdf['timestamp'] = pd.to_datetime(tdf['timestamp'].astype(str))
tdf['month']  = tdf['timestamp'].dt.to_period('M').astype(str)
tdf['year']   = tdf['timestamp'].dt.year
tdf['is_negative'] = tdf['sentiment'] == 'negative'
tdf['is_toxic']    = tdf['toxicity'].isin(['high','medium'])

TGT_LABEL = {
    'content_work': 'Content / work',
    'creator':      'Creator',
    'other_user':   'Other user',
    'appearance':   'Appearance',
    'product':      'Product',
    'none':         'None / unclear',
    'off_topic':    'Off-topic',
}
TGT_ORDER = ['Content / work', 'Creator', 'Other user',
             'Appearance', 'Product', 'None / unclear', 'Off-topic']
TGT_COLORS = ['#3498db','#e67e22','#e74c3c','#9b59b6','#2ecc71','#95a5a6','#f39c12']

tdf['target_lbl'] = tdf['target'].map(TGT_LABEL)

print(f'Comments: {len(tdf):,}  |  months: {tdf["month"].nunique()}')
print()
print(tdf['target_lbl'].value_counts().to_string())\
"""

CODE_J_L1 = """\
# J.L1. Stacked area: target share (%) per month -- absolute composition.
n_per_month = tdf.groupby('month').size()
stable_j = n_per_month[n_per_month >= 100].index  # only months with real volume

mt = (tdf[tdf['month'].isin(stable_j)]
        .groupby(['month','target_lbl']).size()
        .reset_index(name='n'))
mt['pct'] = mt.groupby('month')['n'].transform(lambda x: x / x.sum() * 100)
piv = (mt.pivot(index='month', columns='target_lbl', values='pct')
         .reindex(columns=TGT_ORDER, fill_value=0))
piv.index = pd.to_datetime(piv.index.astype(str), format='%Y-%m')
piv = piv.sort_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.stackplot(piv.index,
             [piv[t].values for t in TGT_ORDER],
             labels=TGT_ORDER,
             colors=TGT_COLORS, alpha=0.85)
ax.set_ylabel('% of comments')
ax.set_ylim(0, 100)
ax.set_title('Target distribution over time  (% of comments per month)', fontsize=13)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.tick_params(axis='x', rotation=35)
plt.tight_layout()
plt.savefig(OUT_J / 'J_L1_target_share_stacked.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_J / 'J_L1_target_share_stacked.svg', bbox_inches='tight')
plt.show()

print('Annual target mix (%):')
ann = (tdf.groupby(['year','target_lbl']).size()
          .reset_index(name='n'))
ann['pct'] = ann.groupby('year')['n'].transform(lambda x: x/x.sum()*100)
print(ann.pivot(index='year', columns='target_lbl', values='pct')
         .reindex(columns=TGT_ORDER).round(1).to_string())\
"""

CODE_J_L2 = """\
# J.L2. Line chart: negative rate per target over time (12-month rolling).
neg_by_tgt = (tdf[tdf['month'].isin(stable_j)]
               .groupby(['month','target_lbl'])['is_negative']
               .mean().mul(100).reset_index(name='neg_rate'))

fig, ax = plt.subplots(figsize=(14, 6))
for color, tgt in zip(TGT_COLORS, TGT_ORDER):
    sub = neg_by_tgt[neg_by_tgt['target_lbl'] == tgt].copy()
    sub = sub.sort_values('month')
    sub['month_dt'] = pd.to_datetime(sub['month'].astype(str), format='%Y-%m')
    sub = sub.set_index('month_dt').sort_index()
    roll = sub['neg_rate'].rolling(12, center=True, min_periods=4).mean()
    ax.plot(sub.index, roll.values, linewidth=2.2, label=tgt, color=color)

ax.set_ylabel('Negative rate (%, 12-mo rolling)')
ax.set_xlabel('Month')
ax.set_title('Negative rate per target over time  (12-month rolling average)', fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.tick_params(axis='x', rotation=35)
ax.grid(True, linewidth=0.3, alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_J / 'J_L2_target_negrate_over_time.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_J / 'J_L2_target_negrate_over_time.svg', bbox_inches='tight')
plt.show()\
"""

CODE_J_L3 = """\
# J.L3. Heatmap: neg rate per target x year (compact, year as rows).
neg_hm = (tdf.groupby(['year','target_lbl'])['is_negative']
              .mean().mul(100).unstack('target_lbl'))
neg_hm = neg_hm.reindex(columns=TGT_ORDER)
# drop years with very few posts
yr_counts = tdf.groupby('year').size()
neg_hm = neg_hm.loc[yr_counts[yr_counts >= 500].index]

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(neg_hm.values, cmap='YlOrRd', aspect='auto', vmin=0)
ax.set_xticks(range(len(neg_hm.columns)))
ax.set_xticklabels(neg_hm.columns, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(len(neg_hm.index)))
ax.set_yticklabels(neg_hm.index.astype(str), fontsize=10)
for i in range(len(neg_hm.index)):
    for j in range(len(neg_hm.columns)):
        v = neg_hm.values[i, j]
        if not (v != v):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=8.5, color='white' if v > 18 else 'black')
plt.colorbar(im, ax=ax, label='% negative')
ax.set_title('Negative rate per target x year', fontsize=13, pad=10)
plt.tight_layout()
plt.savefig(OUT_J / 'J_L3_target_negrate_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_J / 'J_L3_target_negrate_heatmap.svg', bbox_inches='tight')
plt.show()\
"""

CODE_J_L4 = """\
# J.L4. Small-multiples: volume trend per target (comment count, 12-mo rolling).
vol_by_tgt = (tdf[tdf['month'].isin(stable_j)]
               .groupby(['month','target_lbl']).size()
               .reset_index(name='n'))

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
axes = axes.flatten()

for ax, tgt, color in zip(axes, TGT_ORDER, TGT_COLORS):
    sub = vol_by_tgt[vol_by_tgt['target_lbl'] == tgt].copy()
    sub['month_dt'] = pd.to_datetime(sub['month'].astype(str), format='%Y-%m')
    sub = sub.set_index('month_dt').sort_index()
    ax.bar(sub.index, sub['n'].values, width=20, color=color, alpha=0.35)
    roll = sub['n'].rolling(12, center=True, min_periods=4).mean()
    ax.plot(sub.index, roll.values, color=color, linewidth=2)
    ax.set_title(tgt, fontsize=10)
    ax.tick_params(axis='x', rotation=40, labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.4)

axes[-1].set_visible(False)  # 7 targets, 8 subplots
plt.suptitle('Comment volume per target over time  (bars=monthly, line=12-mo rolling)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_J / 'J_L4_target_volume_small_multiples.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_J / 'J_L4_target_volume_small_multiples.svg', bbox_inches='tight')
plt.show()\
"""

new_cells = [
    md_cell(MD_J,           'j_header'),
    code_cell(CODE_J_LOAD,  'j_load'),
    code_cell(CODE_J_L1,    'j_l1'),
    code_cell(CODE_J_L2,    'j_l2'),
    code_cell(CODE_J_L3,    'j_l3'),
    code_cell(CODE_J_L4,    'j_l4'),
]

nb['cells'].extend(new_cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Done: {len(nb["cells"])} cells. New: {[c["id"] for c in new_cells]}')
