# Preliminaries
import os
import pandas as pd
import numpy as np
import re

#Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

#Transformers
from transformers import T5ForConditionalGeneration

#Warning
import warnings
warnings.filterwarnings('ignore')

#Mymodule
import config


#helper function
def encode(batch,tokenizer,max_len):
    '''
    This function takes in a batch and a tokenizer and outputs encoded batch
    '''
    return tokenizer.encode_plus(
      batch,
      None,
      add_special_tokens = True,
      max_length = max_len,
      pad_to_max_length = True
         )

#loading model
model = T5ForConditionalGeneration.from_pretrained('t5-large')
num_added_toks = config.Tokenizer.add_tokens(['<label>','<cause>'])
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(config.Tokenizer))

#loading Model states
models_path = "/content/gdrive/My Drive/Colab Notebooks/emotions_model_T5_Large/T5_large_emotions_4.pt" # add path to your saved model here
model.load_state_dict(torch.load(models_path))

device='cuda'
model.to(device)


def predict(test_example):
    '''
    This function generates prediction for given input
    '''
  model.eval()
  with torch.no_grad():
    encoded = encode(test_example,config.Tokenizer,config.MAX_INPUT_LEN)

    ids = torch.tensor(encoded['input_ids'],dtype=torch.long).to(device).reshape(1,config.MAX_INPUT_LEN) #Modify this if you are using different batch_size
    mask = torch.tensor(encoded['attention_mask'],dtype=torch.long).to(device).reshape(1,config.MAX_INPUT_LEN) #Modify this if you are using different batch_size

    generated_ids = model.generate(
            input_ids=ids,
            attention_mask=mask,
            num_beams=1,
            max_length=config.MAX_TARGET_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )
  
    preds = [
            config.Tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]

  print(preds)


# start predicting
predict("I am sad because I didn't get the award")