import statistics
import matplotlib.pyplot as plt

dict_results = {}

with open("data.txt", "r") as f:
    lignes = f.readlines()


for ligne in lignes:
    liste_results = []
    index = 0
    while ligne[index] != '[' :
        index += 1
    
    if index > 0 :
        str_before = ligne[:index]
        input_choice = 'InputHybrid'
        count = 0
    else :
        count += 1
        if count == 2 :
            input_choice = 'InputQ&A'
        else :
            input_choice = 'InputChitchat'

    
    str_before_list = str_before.split(' , ')
    type_of_answer_wanted = str_before_list[0]

    if type_of_answer_wanted == 'type_of_answer_wanted=chichat' :
        type_of_answer_wanted = 'type_of_answer_wanted=chitchat'

    keep_context = str_before_list[1]

    liste_str = ligne[index:].strip().split(",")
    liste_results = [float(x[8:-1]) for x in liste_str[:-1]]
    liste_results.append(float(liste_str[-1][8:-2]))

    if keep_context not in dict_results.keys() :
       dict_results[keep_context] = {}

    if type_of_answer_wanted not in dict_results[keep_context].keys() :
        dict_results[keep_context][type_of_answer_wanted] = {}

    dict_results[keep_context][type_of_answer_wanted][input_choice] = statistics.mean(liste_results)
    

import matplotlib.pyplot as plt
import numpy as np


# Données
group_names = ['Hybride', 'Chitchat pur', 'Q&A pur']
bar_colors = ['#FDB813', '#00788C', '#EE4035'] # couleur des barres
legend_labels = ['Inputs Hybride', 'Inputs Chitchat', 'Inputs Q&A'] # légende pour chaque couleur

keep_context = "keep_context=False"

data = [[],[],[]]
for key in dict_results[keep_context].keys() :
    index = 0
    for k in dict_results[keep_context][key].keys() :
        data[index].append(dict_results[keep_context][key][k])
        index += 1

if keep_context == "keep_context=True" :
    for i in range(len(data)) :   
        d = data[i].pop()
        d1 = data[i].pop()
        data[i].insert(0,d1)
        data[i].insert(0,d)

if keep_context == "keep_context=False" :
    for i in range(len(data)) :   
        d = data[i].pop()
        data[i].insert(0,d)



# Création du graphique
fig, ax = plt.subplots()
index = np.arange(len(group_names))
bar_width = 0.25

# Groupe 1
rects1 = ax.bar(index, data[0], bar_width, color=bar_colors[0], label=legend_labels[0])
rects2 = ax.bar(index + bar_width, data[1], bar_width, color=bar_colors[1], label=legend_labels[1])
rects3 = ax.bar(index + 2*bar_width, data[2], bar_width, color=bar_colors[2], label=legend_labels[2])


# Configuration des axes et du graphique
ax.set_xlabel('Modèles utilisés')
ax.set_ylabel('Perpléxité moyenne')
ax.set_title('Perpléxité moyenne des modèles obtenus sur 200 nouvelles phrases')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(group_names)
ax.legend()

plt.show()

