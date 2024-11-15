import os, sys
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

identity_path = r"D:\Celeba\identity_CelebA.txt"
with open(identity_path) as f:
    fp_ids = [tuple(x.split(' ')) for x in f.read().split('\n')]
fp_ids = [(i[0], int(i[1])) for i in fp_ids]
df = pd.DataFrame(fp_ids, columns=['filepath', 'person_id'])
df.to_csv('CelebaTriplets.csv')
len(df['person_id'].unique())
sns.histplot(df['person_id'])

#plt.show()
#print(df)

low_count_ids = []
for person_id in df['person_id'].unique():
    if len(df[df['person_id']==person_id])<2:
        low_count_ids.append(person_id)
neg_anchor =  df[df['person_id']==2880]['filepath'].sample(1).item()
anchors = []
for person_id in df['person_id'].unique():
    if person_id not in low_count_ids:
        base_anchor, pos_anchor = df[df['person_id']==person_id]['filepath'].sample(2)
        neg_anchor = df[df['person_id']!=person_id]['filepath'].sample(1).item()
        anchors.append({'person_id': person_id,
                     "base_anchor": base_anchor,
                     "positive_anchor": pos_anchor,
                     "negative_anchors": neg_anchor})

#print(anchors)

