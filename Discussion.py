#import Perplexite
import Generateur
from transformers import GPT2Tokenizer,GPT2LMHeadModel

def discussion(type_of_answer_wanted=None,input_user=None,keep_context=True,keep_perplexity=False,show_answer=True) :
    list_perplexity = []
    indice = 0
    msg = " "
    while 1!=0 :
        
        if input_user is None :
            sentence = input()
        else :
            if indice < len(input_user) :
                sentence = input_user['Input'][indice]
                indice+=1
                print(indice)
            else :
                sentence = 'q'

        if sentence == "Quit" or sentence == "quit" or sentence == 'q' :
            break
        
        if type_of_answer_wanted == None :
            if input_user is None :
                type_of_answer_needed = "chitchat"
            else :
                type_of_answer_needed = ['chitchat','Q&A'][input_user['Label'][indice-1]]

        else :
            type_of_answer_needed = type_of_answer_wanted

        #print("Message envoyé dans le modèle : ", msg)
        if len(msg) > 2048 :
            msg = msg[-2048:]
        
        answer = Generateur.generator(sentence,type_of_answer_needed,msg)

        if keep_context :
            msg += sentence
        else :
            msg = sentence
        
        msg+='\n'

        if show_answer :
            print(answer)

        p = Generateur.get_perplexity(msg,answer,type_of_answer_needed)
        '''
        separator = ['.','?','!','\n']
        sentences = Perplexite.custom_split(separator,answer)
        #print("Phrases : ",sentences)

        temp_answer  = ""
        old_s = ""
        for i in range(len(sentences)) :
            iwannabreak = False
            #print("old_s : ",old_s)
            if i >= 1 :
                for s in sentences[:i-1] :
                    #print("petit_s = ",s)
                    if s == old_s and len(s) > 2:
                        iwannabreak = True

            if iwannabreak :
                #print('Break')
                break
            s = sentences[i]

            temp_answer += s
            #print("S = ",temp_answer)
            perp = Perplexite.get_perplexity(msg,temp_answer,type_of_answer_needed)
            old_s = s
            print(perp)
            if perp < p :
                answer = temp_answer
                p = perp
        '''
        msg+='\n'
        msg += answer
        #print("Réponse finale : ",answer)
        if keep_perplexity :
            list_perplexity.append(p)
    return list_perplexity