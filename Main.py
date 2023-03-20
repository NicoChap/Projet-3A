import Generateur
import math
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import Perplexite

Txt_generator = Generateur.Text_generator()



msg = ""
print('.............................................')
while 1!=0 :
    
    sentence = input()
    if sentence == "Quit" or sentence == "quit" or sentence == 'q' :
        break

    type_of_answer_needed = "Q&A"
    
    msg+='\n'
    msg += sentence
    #print("Message envoyé dans le modèle : ", msg)
    if len(msg) > 2048 :
        msg = msg[-2048:]
    answer = Txt_generator.chatbot_response(msg,type_of_answer_needed)

    print("Réponse complète : ",answer)
    p = 10e10
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

    msg+='\n'
    msg += answer
    print("Réponse finale : ",answer)
print('.............................................')
