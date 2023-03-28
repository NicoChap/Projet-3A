import pandas as pd

path = './Data/ValidationData.csv'
df = pd.read_csv(path)

num_chitchat = len(df.query('(Label == 0)'))
num_QA = len(df.query('(Label == 1)'))

print('On a', num_QA, 'donnée labelisées 1 et', num_chitchat, 'donnée labélisées 0')

qa_df = df.query('(Label == 1)')
cc_df = df.query('(Label == 0)')
if len(qa_df) - len(cc_df) > 0:
    samples = qa_df.sample(len(cc_df))
    df = pd.concat(samples, cc_df)
else:
    samples = cc_df.sample(len(qa_df))
    df = pd.concat([qa_df, samples])

print("Post balancing !!")
num_chitchat = len(df.query('(Label == 0)'))
num_QA = len(df.query('(Label == 1)'))

print('On a', num_QA, 'donnée labelisées 1 et', num_chitchat, 'donnée labélisées 0')

for i in range(len(df)):
    print(df['Input'][i])