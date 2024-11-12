import seaborn as sns
import pandas as pd

identity_path = r"D:\Celeba\identity_CelebA.txt"
with open(identity_path) as f:
    fp_ids = [tuple(x.split(' ')) for x in f.read().split('\n')]
fp_ids = [(i[0], int(i[1])) for i in fp_ids]
df = pd.DataFrame(fp_ids, columns=['filepath', 'person_id'])
df.to_csv('CelebaTriplets.csv')
len(df['person_id'].unique())
sns.histplot(df['person_id'])
low_count_ids = []
for p in df['person_id'].unique():
    if len(df[df['person_id']==p])<2:
        low_count_ids.append(p)

neg_anchor =  df[df['person_id']==2880]['filepath'].sample(1).item()
anchors = []
for p in df['person_id'].unique():
    if p not in low_count_ids:
        base_anchor, pos_anchor = df[df['person_id']==p]['filepath'].sample(2)
        neg_anchor = df[df['person_id']!=p]['filepath'].sample(1).item()
