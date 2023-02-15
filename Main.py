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

    type_of_answer_needed = "chitchat"
    
    msg += sentence
    if len(msg) > 2048 :
        msg = msg[-2048:]
    answer = Txt_generator.chatbot_response(msg,type_of_answer_needed)
    #print(answer)
    p = 10e10
    separator = ['.','?','!']
    sentences = Perplexite.custom_split(separator,answer)
    #print(sentences)

    temp_answer  = ""
    old_s = ""
    for s in sentences :
        print("S = ",s)
        if s == old_s :
            break
        temp_answer += s
        perp = Perplexite.get_perplexity(msg,temp_answer,type_of_answer_needed)
        old_s = s
        print(perp)
        if perp < p :
            answer = temp_answer
            p = perp

    msg += answer
    print(answer)
print('.............................................')
