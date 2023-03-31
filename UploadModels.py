from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM

"""
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)

# télécharger le modèle complet pour une utilisation ultérieure hors ligne
model.save_pretrained(save_directory="./Models/Chitchat/Model")


tokenizer_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Télécharger le tokenizer complet pour une utilisation ultérieure hors ligne
tokenizer.save_pretrained("./Models/Chitchat/Tokenizer")

"""
model_name = "consciousAI/question-answering-generative-t5-v1-base-s-q-c"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# télécharger le modèle complet pour une utilisation ultérieure hors ligne
model.save_pretrained(save_directory="./Models/QA/Model")


tokenizer_name = "consciousAI/question-answering-generative-t5-v1-base-s-q-c"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Télécharger le tokenizer complet pour une utilisation ultérieure hors ligne
tokenizer.save_pretrained("./Models/QA/Tokenizer")