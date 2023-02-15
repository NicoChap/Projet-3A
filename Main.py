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

    print(Perplexite.get_perplexity(msg,answer,type_of_answer_needed))

    msg += answer
    print(answer)
print('.............................................')
