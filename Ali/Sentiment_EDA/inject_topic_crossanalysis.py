"""Inject Section K: Post topic x negative/toxic cross-analysis."""
import json, pathlib, uuid

NB = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

def md_cell(src, cid=None):
    return {"cell_type": "markdown", "id": cid or uuid.uuid4().hex[:8],
            "metadata": {}, "source": src}

def code_cell(src, cid=None):
    return {"cell_type": "code", "id": cid or uuid.uuid4().hex[:8],
            "metadata": {}, "execution_count": None, "outputs": [], "source": src}

MD_K = """\
## K. Post Topic x Negative & Toxic Cross-Analysis

517 posts (30% of comments, n=147k) have LLM-assigned topic labels spanning 15 categories.
This section crosses topic with negativity, toxicity, target, and emotion to answer:
which content types attract the most hostile comment sections?\
"""

CODE_K_LOAD = """\
# K.0  Load and join post topics to comment-level sentiment data.
import pandas as pd, numpy as np, matplotlib.pyplot as plt, pathlib

BASE = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab')
OUT_K = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/outputs')

topics = pd.read_csv(BASE / 'Ali/topics/camihawke_instagram_post_topics_confidence.csv')
topics['post_id'] = topics['post_id'].astype(str)

kdf = pd.read_parquet(
    BASE / 'Ali/outputs/stage2_sentiment/sentiment_instagram.parquet',
    columns=['media_id','timestamp','sentiment','sentiment_score',
             'toxicity','emotion','intent','target'])
kdf['media_id'] = kdf['media_id'].astype(str)
kdf['timestamp'] = pd.to_datetime(kdf['timestamp'].astype(str))
kdf['month'] = kdf['timestamp'].dt.to_period('M').astype(str)
kdf['year']  = kdf['timestamp'].dt.year

kdf = kdf.merge(topics, left_on='media_id', right_on='post_id', how='inner')

kdf['is_negative'] = kdf['sentiment'] == 'negative'
kdf['is_toxic']    = kdf['toxicity'].isin(['mild', 'severe'])

TGT_LABEL = {
    'content_work': 'Content / work', 'creator': 'Creator',
    'other_user': 'Other user',       'appearance': 'Appearance',
    'product': 'Product',             'none': 'None / unclear',
    'off_topic': 'Off-topic',
}
kdf['target_lbl'] = kdf['target'].map(TGT_LABEL)

# collapse small topics for readability
MAIN_TOPICS = ['Relatable Comedy','Private Life','Beauty & Fashion',
               'Theatre','Community','Culture','Travel','Food','Reflections']
kdf['topic_grp'] = kdf['topic'].where(kdf['topic'].isin(MAIN_TOPICS), other='Other')

# sort topics by negative rate for consistent ordering throughout
topic_neg = kdf.groupby('topic_grp')['is_negative'].mean().sort_values(ascending=False)
TOPIC_ORDER = topic_neg.index.tolist()

print(f'Comments with topic: {len(kdf):,}')
print(f'Topics (grouped): {kdf["topic_grp"].nunique()}')
print()
print('Comments and neg/tox rate per topic:')
summary = kdf.groupby('topic_grp').agg(
    n=('is_negative','count'),
    neg_rate=('is_negative','mean'),
    tox_rate=('is_toxic','mean'),
).reindex(TOPIC_ORDER)
summary['neg_rate'] = summary['neg_rate'].mul(100)
summary['tox_rate'] = summary['tox_rate'].mul(100)
print(summary.round(2).to_string())\
"""

CODE_K_K1 = """\
# K.1  Neg rate + tox rate per topic -- horizontal bar chart, sorted by neg rate.
agg_k1 = kdf.groupby('topic_grp').agg(
    n         = ('is_negative', 'count'),
    neg_rate  = ('is_negative', 'mean'),
    tox_rate  = ('is_toxic',    'mean'),
).reindex(TOPIC_ORDER).mul({'n':1, 'neg_rate':100, 'tox_rate':100})

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

for ax, col, color, title in [
    (axes[0], 'neg_rate', '#e74c3c', 'Negative rate (%)'),
    (axes[1], 'tox_rate', '#8e44ad', 'Toxic rate (%)'),
]:
    bars = ax.barh(range(len(TOPIC_ORDER)), agg_k1[col].values,
                   color=color, alpha=0.78)
    for bar, v, n in zip(bars, agg_k1[col].values, agg_k1['n'].values):
        ax.text(bar.get_width() + agg_k1[col].max() * 0.02,
                bar.get_y() + bar.get_height()/2,
                f'{v:.1f}%  (n={int(n):,})', va='center', fontsize=8.5)
    ax.set_yticks(range(len(TOPIC_ORDER)))
    ax.set_yticklabels(TOPIC_ORDER, fontsize=10)
    ax.set_xlabel(title, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, agg_k1[col].max() * 1.55)
    ax.grid(axis='x', linewidth=0.3, alpha=0.5)

plt.suptitle('Which post topics attract the most negative & toxic comments?', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_K / 'K1_topic_neg_tox_rate.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_K / 'K1_topic_neg_tox_rate.svg', bbox_inches='tight')
plt.show()\
"""

CODE_K_K2 = """\
# K.2  Topic x target heatmap: neg rate for each topic x target combination.
# Shows WHERE negativity lands depending on what the post is about.
k2 = kdf.dropna(subset=['target_lbl'])
TGT_ORDER_K = ['Content / work','Creator','Other user','Appearance','Product','None / unclear','Off-topic']

neg_heat2 = (k2.groupby(['topic_grp','target_lbl'])['is_negative']
               .mean().mul(100).unstack('target_lbl'))
neg_heat2 = neg_heat2.reindex(index=TOPIC_ORDER,
                               columns=[c for c in TGT_ORDER_K if c in neg_heat2.columns])

cnt_heat2 = (k2.groupby(['topic_grp','target_lbl']).size()
               .unstack('target_lbl').reindex(index=TOPIC_ORDER,
                columns=[c for c in TGT_ORDER_K if c in neg_heat2.columns]).fillna(0))

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(neg_heat2.values, cmap='YlOrRd', aspect='auto', vmin=0)
ax.set_xticks(range(len(neg_heat2.columns)))
ax.set_xticklabels(neg_heat2.columns, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(len(neg_heat2.index)))
ax.set_yticklabels(neg_heat2.index, fontsize=10)

for i in range(len(neg_heat2.index)):
    for j in range(len(neg_heat2.columns)):
        v = neg_heat2.values[i, j]
        n = cnt_heat2.values[i, j]
        if not (v != v) and n >= 30:
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=8, color='white' if v > 18 else 'black')
        elif not (v != v) and n < 30:
            ax.text(j, i, '-', ha='center', va='center', fontsize=8, color='grey')

plt.colorbar(im, ax=ax, label='% negative  (grey=n<30)')
ax.set_title('Negative rate: topic x target  (where hostility lands per content type)',
             fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(OUT_K / 'K2_topic_target_negrate_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_K / 'K2_topic_target_negrate_heatmap.svg', bbox_inches='tight')
plt.show()\
"""

CODE_K_K3 = """\
# K.3  Topic x emotion: dominant emotion fingerprint per topic (row-normalised heatmap).
top_em_k = kdf['emotion'].value_counts().head(8).index.tolist()
k3 = kdf[kdf['emotion'].isin(top_em_k)]

em_heat = (pd.crosstab(k3['topic_grp'], k3['emotion'], normalize='index') * 100)
em_heat = em_heat.reindex(TOPIC_ORDER)
# sort emotions by overall frequency
em_heat = em_heat[kdf['emotion'].value_counts().head(8).index.tolist()]

fig, ax = plt.subplots(figsize=(13, 5.5))
im = ax.imshow(em_heat.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(em_heat.columns)))
ax.set_xticklabels(em_heat.columns, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(len(em_heat.index)))
ax.set_yticklabels(em_heat.index, fontsize=10)

for i in range(len(em_heat.index)):
    for j in range(len(em_heat.columns)):
        v = em_heat.values[i, j]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                fontsize=8, color='white' if v > 45 else 'black')

plt.colorbar(im, ax=ax, label='% of topic comments')
ax.set_title('Emotion fingerprint per topic  (row-normalised)', fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(OUT_K / 'K3_topic_emotion_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_K / 'K3_topic_emotion_heatmap.svg', bbox_inches='tight')
plt.show()\
"""

CODE_K_K4 = """\
# K.4  Topic neg rate over time -- line chart per topic (annual resolution for legibility).
# Only topics with >= 500 comments total.
large_topics = kdf.groupby('topic_grp').size()
large_topics = large_topics[large_topics >= 500].index.tolist()
k4 = kdf[kdf['topic_grp'].isin(large_topics)].copy()

ann_neg = (k4.groupby(['year','topic_grp'])['is_negative']
             .mean().mul(100).reset_index(name='neg_rate'))

# colour palette
import matplotlib.cm as cm
cmap_k4 = cm.tab10
colors_k4 = {t: cmap_k4(i/len(large_topics)) for i, t in enumerate(large_topics)}

fig, ax = plt.subplots(figsize=(13, 6))
for topic in [t for t in TOPIC_ORDER if t in large_topics]:
    sub = ann_neg[ann_neg['topic_grp'] == topic].sort_values('year')
    ax.plot(sub['year'].values, sub['neg_rate'].values,
            marker='o', markersize=5, linewidth=2,
            label=topic, color=colors_k4[topic])

ax.set_xlabel('Year')
ax.set_ylabel('Negative rate (%)')
ax.set_title('Negative rate per topic over time  (annual)', fontsize=13)
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.grid(True, linewidth=0.3, alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_K / 'K4_topic_negrate_over_time.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_K / 'K4_topic_negrate_over_time.svg', bbox_inches='tight')
plt.show()

print('Annual neg rate per topic (%):')
piv_k4 = ann_neg[ann_neg['topic_grp'].isin(large_topics)].pivot(
    index='year', columns='topic_grp', values='neg_rate')
piv_k4 = piv_k4[[t for t in TOPIC_ORDER if t in piv_k4.columns]]
print(piv_k4.round(1).to_string())\
"""

CODE_K_K5 = """\
# K.5  Bubble chart: topic x sentiment summary
# x=pos_rate, y=neg_rate, size=n_comments, colour=tox_rate
agg_k5 = kdf.groupby('topic_grp').agg(
    n        = ('is_negative', 'count'),
    neg_rate = ('is_negative', 'mean'),
    tox_rate = ('is_toxic',    'mean'),
    pos_rate = ('sentiment',   lambda x: (x == 'positive').mean()),
    avg_score= ('sentiment_score', 'mean'),
).reindex(TOPIC_ORDER)

fig, ax = plt.subplots(figsize=(10, 7))
import matplotlib.cm as cm
norm_k5 = plt.Normalize(vmin=agg_k5['tox_rate'].min(), vmax=agg_k5['tox_rate'].max())
cmap_k5 = cm.YlOrRd

for topic, row in agg_k5.iterrows():
    color = cmap_k5(norm_k5(row['tox_rate']))
    size  = row['n'] / agg_k5['n'].max() * 1800 + 80
    ax.scatter(row['pos_rate']*100, row['neg_rate']*100,
               s=size, c=[color], alpha=0.85,
               edgecolors='grey', linewidths=0.5)
    ax.annotate(topic,
                (row['pos_rate']*100, row['neg_rate']*100),
                fontsize=8.5, ha='center', va='bottom',
                xytext=(0, 6), textcoords='offset points')

ax.set_xlabel('Positive rate (%)', fontsize=11)
ax.set_ylabel('Negative rate (%)', fontsize=11)
ax.set_title('Topic sentiment landscape  (size=volume, colour=tox rate)', fontsize=13)
plt.colorbar(cm.ScalarMappable(norm=norm_k5, cmap=cmap_k5),
             ax=ax, label='Toxic rate')
ax.grid(True, linewidth=0.3, alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_K / 'K5_topic_sentiment_bubble.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_K / 'K5_topic_sentiment_bubble.svg', bbox_inches='tight')
plt.show()

print('Topic sentiment summary:')
print(agg_k5.assign(
    neg_pct=agg_k5['neg_rate']*100,
    tox_pct=agg_k5['tox_rate']*100,
    pos_pct=agg_k5['pos_rate']*100,
)[['n','neg_pct','tox_pct','pos_pct','avg_score']].round(2).to_string())\
"""

new_cells = [
    md_cell(MD_K,           'k_header'),
    code_cell(CODE_K_LOAD,  'k_load'),
    code_cell(CODE_K_K1,    'k_k1'),
    code_cell(CODE_K_K2,    'k_k2'),
    code_cell(CODE_K_K3,    'k_k3'),
    code_cell(CODE_K_K4,    'k_k4'),
    code_cell(CODE_K_K5,    'k_k5'),
]

nb['cells'].extend(new_cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Done: {len(nb["cells"])} cells. New: {[c["id"] for c in new_cells]}')
