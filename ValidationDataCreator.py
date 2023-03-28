'''Ce fichier a pour but de créer un dataset de validation pour tester notre modèle BERT
Il est important de noter que si vous souhaitez réutiliser cet algorithme pour vos
propres données, il faudra adapter l'algorithme à 1) la structure de vos données;
2) aux chemins d'accès sur votre machine (ligne 35 par exemple)'''

import pandas as pd
print('======== Reading JSONs ========')
qa_path = "Data/RawQA.json"
raw_qa = pd.read_json(qa_path)

cc_path = "Data/RawChitchat.json"
raw_cc = pd.read_json(cc_path)

new_raw_data = {'Input': [], 'Label':[]}

print('======== Adding QA data ========')
for data in raw_qa['data']:
    for paragraph in data['paragraphs']:
        for qa in paragraph['qas']:
            new_raw_data['Input'].append(qa['question'])
            new_raw_data['Label'].append(1)

print('======== Adding ChitChat data ========')
for data in raw_cc:
    for message_list in raw_cc[data]['messages']:
        for i in range(len(message_list)):
            message = message_list[i]['text']
            new_raw_data['Input'].append(message)
            new_raw_data['Label'].append(0)

print('======== Saving data ========')
new_df = pd.DataFrame(new_raw_data)

print(new_df.head())
new_df.to_csv(r'C:\Users\chapl\OneDrive\Documents\GitHub\Projet3A\Projet-3A\Data\ValidationData.csv',
              index = False)