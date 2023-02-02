#Pour utiliser un réseau BERT il faut faire du pré-traitement de donnéesd'abord tokenize une phrase.
#Pour cela, il faut tokenize la phrase en plusieurs token créer un token de départ, de fin,
# de padding (pour arriver à la len max). Puis convertir le tout en ID.
from dataset import Dataset

DS = Dataset
DS.SetFilename(DS, filename='Projet-3A\Data\data.csv')
DS.SetModel(DS, 'bert-base-uncased')
max_length = DS.max_length(DS)
#On tokenize les données car le BERT n'accepte que ce format en tant qu'input et/ou pour être entraine.
sentence = "Je suis alle en Jamaïque hier soir pour faire du snowboard"
print(DS.Encode(DS,sentence, True, max_length))
