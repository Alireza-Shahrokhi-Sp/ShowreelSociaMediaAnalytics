"""Inject Section I: Room State / Consensus / Alignment longitudinal analysis."""
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
MD_I = """\
## I. Room-Level Longitudinal Analysis: State · Consensus · Alignment

Each post has a room-level assessment from the LLM: a `room_state` (united / mixed / divided),
numeric `consensus` and `alignment` scores, `polarization`, and `controversy`. This section
tracks how those post-level room dynamics evolve month by month across the 10-year archive.\
"""

CODE_I_LOAD = """\
# I.0 Load room_vibe and join post timestamps from comment data.
import pandas as pd, numpy as np, matplotlib.pyplot as plt, pathlib

BASE = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab')
OUT_ROOM = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/outputs')

vibe = pd.read_parquet(BASE / 'Ali/outputs/stage2_sentiment/room_vibe_instagram.parquet')

# join month from comment timestamps
comments_ts = pd.read_parquet(
    BASE / 'Ali/outputs/stage2_sentiment/sentiment_instagram.parquet',
    columns=['media_id','timestamp'])
comments_ts['timestamp'] = pd.to_datetime(comments_ts['timestamp'].astype(str))
comments_ts['month'] = comments_ts['timestamp'].dt.to_period('M').astype(str)
post_month = comments_ts.groupby('media_id')['month'].first().reset_index()

rv = vibe.merge(post_month, on='media_id', how='left')
rv['month_dt'] = pd.to_datetime(rv['month'].astype(str), format='%Y-%m')

STATE_ORDER  = ['united', 'mixed', 'divided']
STATE_COLORS = {'united': '#2ecc71', 'mixed': '#f39c12', 'divided': '#e74c3c'}

print(f'Posts with room data: {len(rv):,}')
print(f'Date range: {rv["month"].min()} to {rv["month"].max()}')
print()
print('Room state distribution:')
print(rv['room_state'].value_counts())
print()
print('Consensus (numeric):')
print(rv['consensus'].describe().round(3))\
"""

CODE_I_L1 = """\
# I.L1. Room state distribution over time -- stacked area (% per month).
monthly_state = (rv.groupby(['month', 'room_state'])
                   .size().reset_index(name='n'))
monthly_state['pct'] = (monthly_state.groupby('month')['n']
                                     .transform(lambda x: x / x.sum() * 100))
piv_state = (monthly_state.pivot(index='month', columns='room_state', values='pct')
                           .fillna(0)
                           .reindex(columns=STATE_ORDER, fill_value=0))

piv_state.index = pd.to_datetime(piv_state.index.astype(str), format='%Y-%m')
piv_state = piv_state.sort_index()

# only keep months with >= 3 posts for stability
n_per_month = rv.groupby('month').size()
stable = n_per_month[n_per_month >= 3].index
piv_state = piv_state[piv_state.index.strftime('%Y-%m').isin(stable)]

fig, ax = plt.subplots(figsize=(14, 5))
ax.stackplot(piv_state.index,
             [piv_state[s].values for s in STATE_ORDER],
             labels=STATE_ORDER,
             colors=[STATE_COLORS[s] for s in STATE_ORDER],
             alpha=0.82)

ax.set_ylabel('% of posts')
ax.set_xlabel('Month')
ax.set_ylim(0, 100)
ax.set_title('Room state evolution over time  (% of posts per month)', fontsize=13)
ax.legend(loc='upper left', fontsize=10)
ax.tick_params(axis='x', rotation=35)
plt.tight_layout()
plt.savefig(OUT_ROOM / 'I_L1_room_state_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_ROOM / 'I_L1_room_state_evolution.svg', bbox_inches='tight')
plt.show()

# print annual averages
rv['year'] = rv['month_dt'].dt.year
print(pd.crosstab(rv['year'], rv['room_state'], normalize='index').mul(100).round(1).to_string())\
"""

CODE_I_L2 = """\
# I.L2. Consensus & alignment trajectories (monthly medians + rolling 6-month smooth).
metrics_l2 = {
    'consensus':     ('Consensus (numeric)',   '#3498db'),
    'llm_consensus': ('LLM consensus',         '#1a5276'),
    'llm_alignment': ('LLM alignment',         '#e67e22'),
    'polarization':  ('Polarization',          '#c0392b'),
}

monthly_med = (rv.groupby('month')[list(metrics_l2.keys())]
                 .median().reset_index())
monthly_med.index = pd.to_datetime(monthly_med['month'].astype(str), format='%Y-%m')
monthly_med = monthly_med[monthly_med.index.strftime('%Y-%m').isin(stable)].sort_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
axes = axes.flatten()

for ax, (col, (label, color)) in zip(axes, metrics_l2.items()):
    vals = monthly_med[col]
    ax.plot(monthly_med.index, vals.values, color=color, linewidth=1.2, alpha=0.5, label='monthly median')
    roll = vals.rolling(6, center=True, min_periods=3)
    ax.plot(monthly_med.index, roll.mean().values, color=color, linewidth=2.5, label='6-mo rolling mean')
    ax.set_title(label, fontsize=11)
    ax.set_ylabel('score')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=35)
    ax.grid(True, linewidth=0.3, alpha=0.5)

plt.suptitle('Post-level room dynamics: monthly medians + 6-month rolling average', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_ROOM / 'I_L2_consensus_alignment_trajectory.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_ROOM / 'I_L2_consensus_alignment_trajectory.svg', bbox_inches='tight')
plt.show()

print('Annual medians:')
print(rv.groupby('year')[list(metrics_l2.keys())].median().round(3).to_string())\
"""

CODE_I_L3 = """\
# I.L3. Controversy & polarization over time -- line + rolling + room_state overlay.
monthly_cp = (rv.groupby('month')[['controversy','polarization','vibe_score']]
                .median().reset_index())
monthly_cp.index = pd.to_datetime(monthly_cp['month'].astype(str), format='%Y-%m')
monthly_cp = monthly_cp[monthly_cp.index.strftime('%Y-%m').isin(stable)].sort_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, col, color, title in [
    (ax1, 'controversy',  '#e74c3c', 'Controversy (median per month)'),
    (ax2, 'polarization', '#8e44ad', 'Polarization (median per month)'),
]:
    vals = monthly_cp[col]
    ax.bar(monthly_cp.index, vals.values, width=20, color=color, alpha=0.35, label='monthly median')
    roll = vals.rolling(6, center=True, min_periods=3)
    ax.plot(monthly_cp.index, roll.mean().values, color=color, linewidth=2.5, label='6-mo rolling')
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('score')
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.4)

# shade high-controversy periods (rolling > 0.18)
roll_c = monthly_cp['controversy'].rolling(6, center=True, min_periods=3).mean()
for ax in (ax1, ax2):
    ax.fill_between(monthly_cp.index,
                    ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
                    ax.get_ylim()[1],
                    where=(roll_c > 0.18).values,
                    alpha=0.08, color='red', label='high controversy period')

ax2.tick_params(axis='x', rotation=35)
plt.suptitle('Controversy & polarization trajectory', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_ROOM / 'I_L3_controversy_polarization.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_ROOM / 'I_L3_controversy_polarization.svg', bbox_inches='tight')
plt.show()\
"""

CODE_I_L4 = """\
# I.L4. Room state x sentiment KPIs -- profile table + grouped bar.
kpi_cols = ['vibe_score','consensus','llm_consensus','llm_alignment',
            'polarization','controversy','sarcasm_rate','toxicity_rate']

state_profile = rv.groupby('room_state')[kpi_cols].median()
state_profile = state_profile.reindex(STATE_ORDER)
state_n = rv['room_state'].value_counts().reindex(STATE_ORDER)

print('Room state sentiment profile (median):')
print(state_profile.round(3).to_string())
print()
print('n posts per state:', state_n.to_dict())

# grouped bar: 4 most diagnostic KPIs
plot_cols = ['vibe_score','consensus','polarization','controversy']
plot_labels = ['Vibe score','Consensus','Polarization','Controversy']
n_kpi = len(plot_cols)
x = np.arange(n_kpi)
width = 0.24

fig, ax = plt.subplots(figsize=(10, 5))
for i, state in enumerate(STATE_ORDER):
    vals = [state_profile.loc[state, c] for c in plot_cols]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width=width * 0.92,
                  label=f'{state}  (n={state_n[state]})',
                  color=STATE_COLORS[state])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(plot_labels, fontsize=11)
ax.set_ylabel('Median value')
ax.set_title('Room state KPI profile', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_ROOM / 'I_L4_room_state_kpi_profile.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_ROOM / 'I_L4_room_state_kpi_profile.svg', bbox_inches='tight')
plt.show()\
"""

CODE_I_L5 = """\
# I.L5. LLM vibe label evolution -- stacked area by month.
# Collapse rare vibes into 'other' for readability.
MAIN_VIBES = ['appreciative','amused','supportive','celebratory']
rv['vibe_label'] = rv['llm_vibe'].where(rv['llm_vibe'].isin(MAIN_VIBES), other='divided/other')
VIBE_ORDER  = MAIN_VIBES + ['divided/other']
VIBE_COLORS = ['#2ecc71','#f1c40f','#3498db','#e67e22','#e74c3c']

monthly_vibe = (rv.groupby(['month','vibe_label'])
                  .size().reset_index(name='n'))
monthly_vibe['pct'] = (monthly_vibe.groupby('month')['n']
                                   .transform(lambda x: x / x.sum() * 100))
piv_vibe = (monthly_vibe.pivot(index='month', columns='vibe_label', values='pct')
                        .reindex(columns=VIBE_ORDER, fill_value=0))
piv_vibe.index = pd.to_datetime(piv_vibe.index.astype(str), format='%Y-%m')
piv_vibe = piv_vibe[piv_vibe.index.strftime('%Y-%m').isin(stable)].sort_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.stackplot(piv_vibe.index,
             [piv_vibe[v].values for v in VIBE_ORDER],
             labels=VIBE_ORDER,
             colors=VIBE_COLORS, alpha=0.82)
ax.set_ylabel('% of posts')
ax.set_ylim(0, 100)
ax.set_title('LLM vibe label evolution over time', fontsize=13)
ax.legend(loc='upper left', fontsize=9, ncol=3)
ax.tick_params(axis='x', rotation=35)
plt.tight_layout()
plt.savefig(OUT_ROOM / 'I_L5_vibe_label_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_ROOM / 'I_L5_vibe_label_evolution.svg', bbox_inches='tight')
plt.show()

print('Annual vibe mix (%):')
rv['vibe_label_print'] = rv['vibe_label']
print(pd.crosstab(rv['year'], rv['vibe_label_print'], normalize='index').mul(100).round(1).to_string())\
"""

CODE_I_L6 = """\
# I.L6. Consensus vs alignment scatter -- coloured by room_state, sized by controversy.
fig, ax = plt.subplots(figsize=(9, 7))

for state in STATE_ORDER:
    sub = rv[rv['room_state'] == state]
    sc = ax.scatter(sub['llm_alignment'], sub['llm_consensus'],
                    c=STATE_COLORS[state],
                    s=sub['controversy'] * 400 + 15,
                    alpha=0.45, label=f'{state}  (n={len(sub)})',
                    edgecolors='white', linewidths=0.3)

# quadrant lines
ax.axvline(0.8, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
ax.axhline(0.8, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)

# label quadrants
for (x_pos, y_pos, txt) in [
    (0.1, 0.95, 'Aligned\\n& united'),
    (0.1, 0.5,  'Low\\nconsensus'),
    (0.87, 0.5, 'Aligned but\\ndivided'),
    (0.87, 0.95,'Core zone'),
]:
    ax.text(x_pos, y_pos, txt, transform=ax.transAxes, fontsize=8,
            color='grey', ha='center', va='center')

ax.set_xlabel('LLM alignment score', fontsize=11)
ax.set_ylabel('LLM consensus score', fontsize=11)
ax.set_title('Consensus vs Alignment  (size = controversy)', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_ROOM / 'I_L6_consensus_alignment_scatter.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_ROOM / 'I_L6_consensus_alignment_scatter.svg', bbox_inches='tight')
plt.show()

# correlation table
print('Correlations (post-level):')
corr_cols = ['vibe_score','consensus','llm_consensus','llm_alignment',
             'polarization','controversy','toxicity_rate']
print(rv[corr_cols].corr().round(3).to_string())\
"""

# ---------------------------------------------------------------------------
new_cells = [
    md_cell(MD_I,         'i_header'),
    code_cell(CODE_I_LOAD,'i_load'),
    code_cell(CODE_I_L1,  'i_l1'),
    code_cell(CODE_I_L2,  'i_l2'),
    code_cell(CODE_I_L3,  'i_l3'),
    code_cell(CODE_I_L4,  'i_l4'),
    code_cell(CODE_I_L5,  'i_l5'),
    code_cell(CODE_I_L6,  'i_l6'),
]

nb['cells'].extend(new_cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Done: {len(nb["cells"])} cells total. New: {[c["id"] for c in new_cells]}')
