"""
import openai


# utiliser votre propre clé d'API OpenAI ici
openai.api_key = "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"

class Text_generator() :
    def __init__(self) -> None:
       pass



    def chatbot_response(self,message,type_of_answer = ""):
        if type_of_answer == "chitchat" :
            eng = "text-davinci-002"
        elif type_of_answer == "Q&A": 
            eng = "text-curie-001"
        else :
            eng = "davinci"

        # utiliser GPT-3 pour obtenir une réponse à la question
        response = openai.Completion.create(
            engine= eng,
            prompt=f"{message}\n",
            max_tokens=100,
            n=1,
            temperature=0.4,
        )
        # renvoyer la réponse sous forme de chaîne de caractères
        txt = response.choices[0].text
        #print('.............................................')
        msg = txt
        '''
        msg = ""
        for c in txt :
            if c not in ['.','?','!'] :
                msg += c
            else :
                msg += c
                break'''
        return msg

"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

ModelChitchat = AutoModelForCausalLM.from_pretrained("./Models/Chitchat/Model")
TokenizerChitchat = AutoTokenizer.from_pretrained("./Models/Chitchat/Tokenizer",padding_side = 'left')

ModelQA = AutoModelForSeq2SeqLM.from_pretrained("./Models/QA/Model")
TokenizerQA = AutoTokenizer.from_pretrained("./Models/QA/Tokenizer")

def generator(message,type_of_answer="chitchat",context = " ") :
    if type_of_answer == "Q&A" :
        input_text = "question: " + message + "</s> question_context: " + context
    
        input_tokenized = TokenizerQA.encode(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=1024)

        summary_ids = ModelQA.generate(input_tokenized, 
                                        max_length=30, 
                                        min_length=5, 
                                        num_beams=2,
                                        early_stopping=True,
                                    )
        new_msg = [TokenizerQA.decode(id, clean_up_tokenization_spaces=True, skip_special_tokens=True) for id in summary_ids] 
        new_msg = str(new_msg[0])

    else : 
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = TokenizerChitchat.encode(message + TokenizerChitchat.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        chat_history_ids = TokenizerChitchat.encode(context,return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = ModelChitchat.generate(bot_input_ids, max_length=1000, pad_token_id=TokenizerChitchat.eos_token_id)

        new_msg = TokenizerChitchat.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return new_msg


def get_perplexity(context,response,type_of_answer):
    if type_of_answer == "Q&A" :
        model = ModelQA
        tokenizer = TokenizerQA
    else : 
        model = ModelChitchat
        tokenizer = TokenizerChitchat

    input_ids = tokenizer.encode(context + tokenizer.eos_token + response, return_tensors='pt')
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss.item()
    perplexity = torch.exp(torch.tensor(loss))
    return perplexity