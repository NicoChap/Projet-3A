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

# exemple d'utilisation
"""
Txt_generator = Text_generator()
type_of_answer_needed = "chitchat"

sentence = "Talk to me."

#print(chatbot_response("Bonjour, comment vas-tu?"))
#print(chatbot_response("Qu'est-ce que tu aimes faire?"))
#print(chatbot_response("Quel est ton rôle dans cette conversation?"))
print(Txt_generator.chatbot_response(sentence,type_of_answer_needed))
print('.............................................')"""