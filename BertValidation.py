import torch
from torch.utils.data import TensorDataset, DataLoader
from dataset import Dataset
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np

print("=========Loading Model=========")
device = "cpu"
path = "C:/Users/chapl/OneDrive/Desktop/FinetunedModels"
#path ='C:/Users/chapl/OneDrive/Desktop/FinetunedModels'
BERTmodel = BertForSequenceClassification.from_pretrained('bert-base-uncased')
BERTmodel.load_state_dict(torch.load(path))
tokenizer = BertTokenizer.from_pretrained('C:/Users/chapl/OneDrive/Desktop/old_model')
BERTmodel.to(device)
print("")
print("=========Model Loaded=========")

csv_path = 'C:/Users/chapl/OneDrive/Documents/GitHub/Projet3A/Projet-3A/Data/Testing.csv'
DS = Dataset(filename=csv_path, model='bert-base-uncased')

testing_dict = DS.SelectFewExamples(1)
print("Testing model on {:} values".format(len(testing_dict['input'])))

input_ids = torch.cat(testing_dict['input'], dim=0)
attention_masks = torch.cat(testing_dict["attention_mask"], dim=0)
labels = torch.tensor(testing_dict["label"])

prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_dataloader = DataLoader(prediction_data, batch_size=64)

BERTmodel.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for step, batch in enumerate(prediction_dataloader):
    if step % 10:
        print("Working on batch {:} out of {:}".format(step, len(prediction_dataloader)))
  # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        result = BERTmodel(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        return_dict=True)

    logits = result.logits

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # Store predictions and true labels
    pred_labels_i = np.argmax(logits, axis=1).flatten()
    predictions.append(pred_labels_i)
    true_labels.append(label_ids)
print('    PREDICTION DONE.')

same_result = 0
num_inputs = len(testing_dict['input'])
for i in range(len(true_labels)):
  dic_difference = {}
  difference = np.array(predictions[i]) - np.array(true_labels[i])
  same_result += np.count_nonzero(difference == 0)
print("Nombre de prédictions correctes: ", same_result)
print("Nombre total de prédicitions effectuees: ", num_inputs)

