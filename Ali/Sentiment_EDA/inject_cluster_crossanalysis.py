"""Inject Section H2 cluster cross-analysis cells into persona_sentiment_rfm.ipynb."""
import json, pathlib, uuid

NB = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb')

nb = json.loads(NB.read_text(encoding='utf-8'))


def md_cell(src, cid=None):
    return {"cell_type": "markdown", "id": cid or uuid.uuid4().hex[:8],
            "metadata": {}, "source": src}


def code_cell(src, cid=None):
    return {"cell_type": "code", "id": cid or uuid.uuid4().hex[:8],
            "metadata": {}, "execution_count": None, "outputs": [], "source": src}


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

MD_H2 = """\
## H2. Cluster Cross-Analysis: Sentiment · Emotion · Intent · Target

Full 360-degree view of each lifecycle cluster across every analytical dimension.
All charts below use the four window-level clusters (Brand advocates, Established regulars,
Passive regulars, Occasional visitors) as the common axis and rotate through sentiment,
emotion, intent, and target lenses studied individually in Sections A-G.\
"""

# H2-1: cluster x emotion heatmap
CODE_H2_1 = """\
# H2-1. Cluster x Emotion heatmap (row-normalised % within cluster).
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

_wl = df.dropna(subset=['wl_cluster']).copy()

top_em = _wl['emotion'].value_counts().head(10).index.tolist()
_wl_em = _wl[_wl['emotion'].isin(top_em)]

emo_heat = (
    pd.crosstab(_wl_em['wl_cluster'], _wl_em['emotion'], normalize='index') * 100
)

CLUSTER_ORDER_H2 = ['Brand advocates', 'Established regulars', 'Passive regulars', 'Occasional visitors']
emo_heat = emo_heat.reindex([c for c in CLUSTER_ORDER_H2 if c in emo_heat.index])

# sort emotions by mean value descending
emo_heat = emo_heat[emo_heat.mean().sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(13, 3.8))
im = ax.imshow(emo_heat.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(emo_heat.columns)))
ax.set_xticklabels(emo_heat.columns, rotation=35, ha='right', fontsize=10)
ax.set_yticks(range(len(emo_heat.index)))
ax.set_yticklabels(emo_heat.index, fontsize=10)

for i in range(len(emo_heat.index)):
    for j in range(len(emo_heat.columns)):
        v = emo_heat.values[i, j]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                fontsize=8, color='white' if v > 35 else 'black')

plt.colorbar(im, ax=ax, label='% of cluster comments')
ax.set_title('Emotion fingerprint per lifecycle cluster  (row-normalised)', fontsize=13, pad=10)
plt.tight_layout()
plt.savefig(OUT / 'H2_1_cluster_emotion_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'H2_1_cluster_emotion_heatmap.svg', bbox_inches='tight')
plt.show()
print(emo_heat.round(1).to_string())\
"""

# H2-2: cluster x intent heatmap
CODE_H2_2 = """\
# H2-2. Cluster x Intent heatmap (row-normalised % within cluster).
_wl2 = df.dropna(subset=['wl_cluster']).copy()
_wl2['intent_clean'] = _wl2['intent'].fillna('unknown')

int_heat = (
    pd.crosstab(_wl2['wl_cluster'], _wl2['intent_clean'], normalize='index') * 100
)
int_heat = int_heat.reindex([c for c in CLUSTER_ORDER_H2 if c in int_heat.index])  # type: ignore
# sort intents by mean desc
int_heat = int_heat[int_heat.mean().sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(11, 3.8))
im = ax.imshow(int_heat.values, cmap='PuBu', aspect='auto')
ax.set_xticks(range(len(int_heat.columns)))
ax.set_xticklabels(int_heat.columns, rotation=35, ha='right', fontsize=10)
ax.set_yticks(range(len(int_heat.index)))
ax.set_yticklabels(int_heat.index, fontsize=10)

for i in range(len(int_heat.index)):
    for j in range(len(int_heat.columns)):
        v = int_heat.values[i, j]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                fontsize=9, color='white' if v > 40 else 'black')

plt.colorbar(im, ax=ax, label='% of cluster comments')
ax.set_title('Intent fingerprint per lifecycle cluster  (row-normalised)', fontsize=13, pad=10)
plt.tight_layout()
plt.savefig(OUT / 'H2_2_cluster_intent_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'H2_2_cluster_intent_heatmap.svg', bbox_inches='tight')
plt.show()
print(int_heat.round(1).to_string())\
"""

# H2-3: cluster x target - side-by-side neg heatmap + stacked bar
CODE_H2_3 = """\
# H2-3. Cluster x Target: negative-rate heatmap (left) and stacked target-mix bar (right).
_wl3 = df.dropna(subset=['wl_cluster']).copy()
_wl3['target_lbl'] = _wl3['target'].fillna('none').map(TGT_LABEL).fillna('None / unclear')

# negative rate per cluster x target
neg3 = (_wl3.groupby(['wl_cluster', 'target_lbl'])['is_negative']
             .mean().mul(100).unstack('target_lbl'))
neg3 = neg3.reindex([c for c in CLUSTER_ORDER_H2 if c in neg3.index])

# volume mix per cluster x target (row-normalised)
mix3 = (pd.crosstab(_wl3['wl_cluster'], _wl3['target_lbl'], normalize='index') * 100)
mix3 = mix3.reindex([c for c in CLUSTER_ORDER_H2 if c in mix3.index])  # type: ignore

# align columns: sort by overall neg rate
col_order3 = neg3.mean().sort_values(ascending=False).index.tolist()
neg3 = neg3[col_order3]
mix3 = mix3[[c for c in col_order3 if c in mix3.columns]]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))

# left: neg-rate heatmap
im3 = axes[0].imshow(neg3.values, cmap='Reds', aspect='auto', vmin=0)
axes[0].set_xticks(range(len(neg3.columns)))
axes[0].set_xticklabels(neg3.columns, rotation=35, ha='right', fontsize=9)
axes[0].set_yticks(range(len(neg3.index)))
axes[0].set_yticklabels(neg3.index, fontsize=10)
for i in range(len(neg3.index)):
    for j in range(len(neg3.columns)):
        v = neg3.values[i, j]
        if not (v != v):  # skip NaN
            axes[0].text(j, i, f'{v:.1f}', ha='center', va='center',
                         fontsize=8, color='white' if v > 55 else 'black')
plt.colorbar(im3, ax=axes[0], label='% negative')
axes[0].set_title('Negative rate: cluster x target', fontsize=11)

# right: stacked bar of target volume mix
TCOLORS = plt.cm.tab10.colors
bottom3 = [0] * len(mix3.index)
for k, col in enumerate(mix3.columns):
    vals = mix3[col].values
    axes[1].barh(range(len(mix3.index)), vals, left=bottom3,
                 label=col, color=TCOLORS[k % len(TCOLORS)])
    bottom3 = [b + v for b, v in zip(bottom3, vals)]
axes[1].set_yticks(range(len(mix3.index)))
axes[1].set_yticklabels(mix3.index, fontsize=10)
axes[1].set_xlabel('% of cluster comments')
axes[1].set_title('Target volume mix per cluster  (row-normalised)', fontsize=11)
axes[1].legend(loc='lower right', fontsize=8, ncol=2)

plt.suptitle('Cluster x Target: where attention lands and how hostile it gets', fontsize=13)
plt.tight_layout()
plt.savefig(OUT / 'H2_3_cluster_target_crossanalysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'H2_3_cluster_target_crossanalysis.svg', bbox_inches='tight')
plt.show()\
"""

# H2-4: cluster x sentiment avg score over time
CODE_H2_4 = """\
# H2-4. Avg sentiment score over time, one line per cluster (lifecycle trajectory).
_wl4 = df.dropna(subset=['wl_cluster', 'sentiment_score']).copy()
_wl4['month_str'] = _wl4['month'].astype(str)

ts4 = (_wl4.groupby(['month_str', 'wl_cluster'])['sentiment_score']
            .mean().reset_index())
ts4['month_dt'] = pd.to_datetime(ts4['month_str'], format='%Y-%m')
ts4 = ts4.sort_values('month_dt')

CLUSTER_COLORS = {
    'Brand advocates':     '#2ecc71',
    'Established regulars': '#e74c3c',
    'Passive regulars':    '#3498db',
    'Occasional visitors':    '#95a5a6',
}

fig, ax = plt.subplots(figsize=(13, 5))
for cluster, grp in ts4.groupby('wl_cluster'):
    ax.plot(grp['month_dt'].values, grp['sentiment_score'].values,
            marker='o', markersize=4, linewidth=2,
            label=cluster, color=CLUSTER_COLORS.get(cluster))

ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
ax.set_xlabel('Month')
ax.set_ylabel('Avg sentiment score')
ax.set_title('Avg sentiment score trajectory per lifecycle cluster', fontsize=13)
ax.legend(fontsize=10)
ax.tick_params(axis='x', rotation=35)
plt.tight_layout()
plt.savefig(OUT / 'H2_4_cluster_sentiment_trajectory.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'H2_4_cluster_sentiment_trajectory.svg', bbox_inches='tight')
plt.show()

# print monthly averages table
piv4 = ts4.pivot(index='month_str', columns='wl_cluster', values='sentiment_score')
print(piv4.round(3).tail(12).to_string())\
"""

# H2-5: Cluster sentiment fingerprint - radar / bar comparison
CODE_H2_5 = """\
# H2-5. Cluster sentiment fingerprint: grouped bar across 4 KPIs.
# (Radar charts need careful polar setup -- grouped bar is more readable at this scale.)
_wl5 = df.dropna(subset=['wl_cluster']).copy()

kpis5 = (_wl5.groupby('wl_cluster').agg(
    neg_rate    = ('is_negative',      lambda x: x.mean() * 100),
    tox_rate    = ('is_toxic',         lambda x: x.mean() * 100),
    pos_rate    = ('sentiment',        lambda x: (x == 'positive').mean() * 100),
    avg_score   = ('sentiment_score',  'mean'),
    n_comments  = ('comment_id',       'count'),
).reset_index())

kpis5 = kpis5.set_index('wl_cluster')
kpis5 = kpis5.reindex([c for c in CLUSTER_ORDER_H2 if c in kpis5.index])

# scale avg_score to 0-100 range for visual parity: original is roughly -1..+1
kpis5['avg_score_scaled'] = (kpis5['avg_score'] + 1) / 2 * 100

metrics = ['neg_rate', 'tox_rate', 'pos_rate', 'avg_score_scaled']
labels  = ['Neg rate (%)', 'Tox rate (%)', 'Pos rate (%)', 'Avg score (scaled 0-100)']
n_met = len(metrics)
n_cl  = len(kpis5)

x = np.arange(n_met)
width = 0.18
offsets = np.linspace(-(n_cl-1)/2, (n_cl-1)/2, n_cl) * width

fig, ax = plt.subplots(figsize=(11, 5))
for i, (cluster, row) in enumerate(kpis5.iterrows()):
    vals = [row[m] for m in metrics]
    bars = ax.bar(x + offsets[i], vals, width=width * 0.92,
                  label=f'{cluster}  (n={int(row["n_comments"]):,})',
                  color=list(CLUSTER_COLORS.values())[i])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Value')
ax.set_title('Cluster sentiment fingerprint: 4-KPI comparison', fontsize=13)
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(OUT / 'H2_5_cluster_sentiment_fingerprint.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'H2_5_cluster_sentiment_fingerprint.svg', bbox_inches='tight')
plt.show()
print(kpis5[metrics + ['n_comments']].round(2).to_string())\
"""

# H2-6: Cluster x Intent x Emotion bubble chart
CODE_H2_6 = """\
# H2-6. Cluster x Intent x Emotion: bubble chart (size=comment count, colour=neg rate).
# One subplot per cluster; axes are intent (y) x emotion (x).
_wl6 = df.dropna(subset=['wl_cluster']).copy()
_wl6['intent_clean'] = _wl6['intent'].fillna('unknown')

top_em6 = _wl6['emotion'].value_counts().head(8).index.tolist()
top_in6  = _wl6['intent_clean'].value_counts().head(6).index.tolist()

_wl6 = _wl6[_wl6['emotion'].isin(top_em6) & _wl6['intent_clean'].isin(top_in6)]

agg6 = (_wl6.groupby(['wl_cluster', 'intent_clean', 'emotion'])
             .agg(n=('comment_id','count'), neg=('is_negative','mean'))
             .reset_index())

clusters6 = [c for c in CLUSTER_ORDER_H2 if c in agg6['wl_cluster'].unique()]
n_cl6 = max(len(clusters6), 1)
fig, axes = plt.subplots(1, n_cl6, figsize=(4.5*n_cl6, 5), sharey=True)
if n_cl6 == 1:
    axes = [axes]

import matplotlib.cm as cm
norm6 = plt.Normalize(vmin=0, vmax=agg6['neg'].max())
cmap6 = cm.RdYlGn_r

max_n6 = agg6['n'].max()
size_scale = 1200 / max_n6

for idx, (ax, cluster) in enumerate(zip(axes, clusters6)):
    sub6 = agg6[agg6['wl_cluster'] == cluster]
    xi = {e: i for i, e in enumerate(top_em6)}
    yi = {it: i for i, it in enumerate(top_in6)}

    xs = sub6['emotion'].map(xi)
    ys = sub6['intent_clean'].map(yi)
    sizes = sub6['n'] * size_scale
    colors = cmap6(norm6(sub6['neg'].values))

    sc = ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.8, edgecolors='grey', linewidths=0.4)

    # annotate count
    for _, row6 in sub6.iterrows():
        if row6['n'] >= 20:
            ax.text(xi[row6['emotion']], yi[row6['intent_clean']],
                    str(int(row6['n'])), ha='center', va='center', fontsize=6.5,
                    color='white' if row6['neg'] > 0.45 else 'black')

    ax.set_xticks(range(len(top_em6)))
    ax.set_xticklabels(top_em6, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(top_in6)))
    ax.set_yticklabels(top_in6 if idx == 0 else [], fontsize=8)
    ax.set_title(cluster, fontsize=10, pad=6)
    ax.set_xlim(-0.6, len(top_em6)-0.4)
    ax.set_ylim(-0.6, len(top_in6)-0.4)
    ax.grid(True, linewidth=0.3, alpha=0.5)

plt.colorbar(cm.ScalarMappable(norm=norm6, cmap=cmap6),
             ax=axes[-1], label='Neg rate', shrink=0.7)
plt.suptitle('Intent x Emotion bubble chart per cluster  (size=comment count, colour=neg rate)',
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'H2_6_cluster_intent_emotion_bubble.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'H2_6_cluster_intent_emotion_bubble.svg', bbox_inches='tight')
plt.show()\
"""

# ---------------------------------------------------------------------------
# Build new cells
# ---------------------------------------------------------------------------
new_cells = [
    md_cell(MD_H2,    'h2_header'),
    code_cell(CODE_H2_1, 'h2_1_emo'),
    code_cell(CODE_H2_2, 'h2_2_int'),
    code_cell(CODE_H2_3, 'h2_3_tgt'),
    code_cell(CODE_H2_4, 'h2_4_traj'),
    code_cell(CODE_H2_5, 'h2_5_fp'),
    code_cell(CODE_H2_6, 'h2_6_bub'),
]

nb['cells'].extend(new_cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f"Done: {len(nb['cells'])} cells total. New cells: {[c['id'] for c in new_cells]}")
