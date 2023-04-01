from Discussion import discussion
import pandas as pd

#Ici, on prendra un jeu de test, et on calculera systématiquement la perplexité de la réponse donnée pour obtenir des résultats 
#quantifiables. On prendra notamment dans la fonction discussion les paramètres : 
#input = phrases du jeu de test; type_of_answer_needed : fixé (Chitchat, Q&A, None); keep_context : fixé (True, False)

path_test_dataset = './Data/Testing.csv'

test_dataset = pd.read_csv(path_test_dataset)
test_dataset = pd.concat([test_dataset.head(100),test_dataset.tail(100)])
test_dataset = test_dataset.reset_index(drop=True)
print(test_dataset)


l_type_of_answer_wanted = ['Q&A']
l_keep_context = [True, False]

for t in l_type_of_answer_wanted :
    for c in l_keep_context :
        
        print('-------------------------------------')

        type_of_answer_wanted = t
        keep_context = c
        keep_perplexity = True

        perplexity_list = discussion(type_of_answer_wanted=type_of_answer_wanted, input_user=test_dataset,keep_context=keep_context, keep_perplexity= keep_perplexity, show_answer = False)
        QA_list = []
        Chitchat_list = []

        for L in range(len(test_dataset['Label'])) :
            if test_dataset['Label'][L] == 0 :
                Chitchat_list.append(perplexity_list[L])
            else : 
                QA_list.append(perplexity_list[L])


        with open('data.txt', 'a') as f:
            f.write('type_of_answer_wanted=' + str(type_of_answer_wanted) + " , keep_context=" + str(keep_context) + " , [")
            for i in range(len(perplexity_list)):
                if i < len(perplexity_list) -1:
                    f.write(str(perplexity_list[i]) + ',')
                else :
                    f.write(str(perplexity_list[i]) + '] \n[')
            
            for i in range(len(Chitchat_list)):
                if i < len(Chitchat_list) -1:
                    f.write(str(Chitchat_list[i]) + ',')
                else :
                    f.write(str(Chitchat_list[i]) + '] \n[')
            
            for i in range(len(QA_list)):
                if i < len(QA_list) -1:
                    f.write(str(QA_list[i]) + ',')
                else :
                    f.write(str(QA_list[i]) + '] \n')
    

