from Discussion import discussion
import pandas as pd

#Ici, on prendra un jeu de test, et on calculera systématiquement la perplexité de la réponse donnée pour obtenir des résultats 
#quantifiables. On prendra notamment dans la fonction discussion les paramètres : 
#input = phrases du jeu de test; type_of_answer_needed : fixé (Chitchat, Q&A, None); keep_context : fixé (True, False)

path_test_dataset = './Data/Testing.csv'

test_dataset = pd.read_csv(path_test_dataset)

print(len(test_dataset['Input']))

test_dataset.loc[2001] = ['q',2]
print('OK')

print(test_dataset['Input'][2001])

print('-------------------------------------')


perplexity_list = discussion(type_of_answer_wanted=None, input_user=test_dataset,keep_context=False, keep_perplexity= True, show_answer = False)
QA_list = []
Chitchat_list = []

    

