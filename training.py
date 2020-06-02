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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, WarmUp, get_linear_schedule_with_warmup

#Warning
import warnings
warnings.filterwarnings('ignore')

#Mymodule
import config


# Processing Function
def process_data(path):
    '''
    This function pre-processes the data in the specified format as explained in readme to give input to T5
    '''
  processed = []
  with open(path) as f:
    for line in f:
      pattern = "<(.*?)>"
      label = re.search(pattern,line).group(1)

      start = line.find("<cause>") + len("<cause>")
      end = line.find("<\\cause>")
      cause = line[start:end]

      target = '<label>' + ' '+ label +' ' + '<cause>' + ' '+ cause + '</s>'

      line = re.sub(pattern,'',line)

      processed.append({'sentence':line,'label':label,'cause':cause,'target':target})
  
  return pd.DataFrame(processed)


#Custom Dataset

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

class EmotionDataset(Dataset):
    '''
    Custom Dataset Class
    '''
  def __init__(self,data,tokenizer):
    self.data = data
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.data)

  def __getitem__(self,item):

    input = encode(self.data.iloc[item,0],self.tokenizer,config.MAX_INPUT_LEN)
    target = encode(self.data.iloc[item,3],self.tokenizer,config.MAX_TARGET_LEN)
    target = target['input_ids']
    input_ids = input['input_ids']
    mask = input['attention_mask']

    return {'input_ids':torch.tensor(input_ids,dtype=torch.long),
                'mask': torch.tensor(mask,dtype=torch.long),
                 'target':torch.tensor(target,dtype=torch.long)}


# Loading Model and adding our new tokens
model = T5ForConditionalGeneration.from_pretrained('t5-large')
num_added_toks = config.Tokenizer.add_tokens(['<label>','<cause>'])
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(config.Tokenizer))


#Training Function
def train_fn(data_loader, model, optimizer, device, scheduler,epoch):
    '''
    This is training function for model training
    '''
  model.train()
  
  for bi, d in enumerate(data_loader):
        ids = d["input_ids"]
        mask = d["mask"]
        labels = d['target']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        labels = labels.to(device,dtype=torch.long)
          
        optimizer.zero_grad()
        outputs = model(
            input_ids =ids,
            attention_mask=mask,
            lm_labels = labels
        )

        loss = outputs[0]                        
        loss.backward()

        optimizer.step()
        if scheduler is not None:
                scheduler.step()

        if (bi+1) % 50 == 0:
           print('Epoch [{}/{}], bi[{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, config.EPOCHS, bi+1,len(data_loader), loss.item()))


def run():
  training_data = process_data("/content/gdrive/My Drive/Emotion Cause.txt") # add the path to your files here

  training_dataset = EmotionDataset(training_data,config.Tokenizer)
  emotions_dataloader = DataLoader(training_dataset,
                                batch_size=config.BATCH_SIZE,
                                shuffle=True,
                                num_workers=4)
  
  model.to(device)

  num_train_steps = int(len(emotions_dataloader) / config.BATCH_SIZE * config.EPOCHS)

  optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
  scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_train_steps)

  for epoch in range(config.EPOCHS):
        print(f"EPOCH {epoch+1} started" + '=' * 30)
        train_fn(emotions_dataloader, model, optimizer, device, scheduler,epoch=epoch)
        
        models_folder = "/content/gdrive/My Drive/Colab Notebooks/emotions_model_T5_Large" #add path where you want to save your model here
        if not os.path.exists(models_folder):
          os.mkdir(models_folder)
        torch.save(model.state_dict(), os.path.join(models_folder, f"T5_large_emotions_{epoch}.pt"))


#start Training
run()