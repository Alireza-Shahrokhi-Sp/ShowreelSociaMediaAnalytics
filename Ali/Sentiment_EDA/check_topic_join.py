import pandas as pd, pathlib
BASE = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab')
t = pd.read_csv(BASE / 'Ali/topics/camihawke_instagram_post_topics_confidence.csv')
print('all topics:')
print(t['topic'].value_counts().to_string())
print()

df = pd.read_parquet(BASE / 'Ali/outputs/stage2_sentiment/sentiment_instagram.parquet',
                     columns=['media_id','sentiment','toxicity','timestamp'])
t['post_id'] = t['post_id'].astype(str)
df['media_id'] = df['media_id'].astype(str)
merged = df.merge(t, left_on='media_id', right_on='post_id', how='left')
n_hit = merged['topic'].notna().sum()
hit = n_hit / len(merged)
print(f'Join hit rate: {hit:.1%}  ({n_hit:,} / {len(merged):,} comments)')
print()
print('Comments per topic:')
print(merged['topic'].value_counts().to_string())
