'''Ce fichier est utilisé pour le fintuning d'un modèle BERT pré-entrainé'''

#Code grandement inspiré d'ici: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
#Pour utiliser un réseau BERT il faut faire du pré-traitement de données d'abord tokenize une phrase.
#Pour cela, il faut tokenize la phrase en plusieurs token créer un token de départ, de fin,
# de padding (pour arriver à la len max). Puis convertir le tout en ID.
from dataset import Dataset
from transformers import BertForSequenceClassification, AdamW
import torch
from torch.utils.data import random_split, DataLoader
import random
import numpy as np
import os

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

DS = Dataset(filename='Data\data.csv', model='bert-base-uncased')
#On tokenize les données.En effet, le BERT n'accepte que ce format en tant qu'input 
# et/ou pour être entraine.
encoded_data = DS.EncodeAll()
train_size = int(len(encoded_data.tensors[0])*0.9)
validation_size = len(encoded_data.tensors[0]) - train_size
#On prend aléatoirement des données du dataset et on les placent soit dans le training, soit dans la validation
train_dataset, validation_dataset = random_split(encoded_data, [train_size, validation_size])


#Puis, on créé des batchs aléatoirement à partir de chaque dataset (training et alidation):
batch_size = 32
training_dataloader = DataLoader( train_dataset, batch_size = batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size)

#Import du modèle BERT :
BERTmodel = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification. You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

#Where to perform the mathematical operation (CPU/GPU) ?
if torch.cuda.is_available():
    device_utilise = "cuda"
else:
    device_utilise = "cpu"
device = torch.device(device_utilise)
print("")
print("Le composant utilise est: " + device_utilise)
print("")

#Definir l'optimizer a utiliser:
optimizer = AdamW(BERTmodel.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5,
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
#Configuration pour le fine-tuning:
#AJOUTER TOUT LES VALIDATION TEST ETC. VOIR CETTE VIDEO:
#https://huggingface.co/docs/transformers/training onglet 'TRAIN'.

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []
epochs = 4

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    BERTmodel.train()

    number_of_batch = len(training_dataloader)
    # For each batch of training data...
    for step, batch in enumerate(training_dataloader):
        # Progress update every 40 batches.
        if step % 10 == 0 and not step == 0:    
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, number_of_batch))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        BERTmodel.zero_grad()     

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        forward_pass = BERTmodel.forward(input_ids= b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = forward_pass.loss, forward_pass.logits
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(BERTmodel.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate. Is it useful ??
        # scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(training_dataloader)            

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    BERTmodel.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        if step % 10 == 0 and not step == 0:    
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, number_of_batch))
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            forward_pass = BERTmodel(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = forward_pass.loss
            logits = forward_pass.logits
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy
        }
    )

print("")
print("Training complete!")

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = BERTmodel.module if hasattr(BERTmodel, 'module') else BERTmodel  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
DS.SaveTokenizer(output_dir)