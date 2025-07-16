#for creating a bot
import logging
from telegram import Update,KeyboardButton,ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler,filters

#for model deployment
import os
import re
import glob
import torch
import pickle
import string
import tokenizers
import unicodedata
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast


#preparing all the necessary data for deploying a model

#loading some necessary data
with open('main_tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

#model 
class NamesGRU(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,seq_len = 16,num_layers = 1,drop_prob = 0.12,bf = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.rnn = nn.GRU(input_size = embed_dim,hidden_size = hidden_size,num_layers=num_layers,dropout=drop_prob,batch_first=bf)
        self.linear = nn.Linear(num_layers*hidden_size,vocab_size) #edit linear layer accordingly

    def forward(self,x):
        #checking shapes
        y = self.embed(x)
        y = y.transpose(0,1)
        _,y = self.rnn(y)
        y = y.transpose(0,1)
        y = y.reshape(y.shape[0],-1)
        y = self.linear(y)
        return y
    

#creating the model and loading the weights for it
vocab_size = tokenizer.get_vocab_size()
embed_dim = 25
hidden_size = 75
num_layers = 3
drop_prob = 0.5

#initializing the model
main_model = NamesGRU(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    drop_prob=drop_prob,
    bf = True
)
main_model.load_state_dict(torch.load(f = '/home/luchian/all_data/uni_data/MyModels/NAME_GEN_model.pth'))

#function for model inference
def model_inf(language = 'russian',model = main_model,ft = PreTrainedTokenizerFast(tokenizer_object = tokenizer)):
    """Creates a name of the given language"""
    model.eval()
    available_languages = LANGS
    #check whether language is in available languages
    if language not in available_languages:
        raise ValueError('The language is not available')
    #staring producing a name 
    current_letter_id = ft('<'+language+'>', padding = 'do_not_pad', return_tensors='pt')['input_ids']
    name = ''
    y_pred = model(current_letter_id).softmax(dim = 1)
    y_pred = y_pred.squeeze(dim = 0)
    while len(name) < 16:
        #adding a letter
        next_pred_ind = Categorical(probs=y_pred).sample([1]).item()
        if next_pred_ind == 0:
            break
        name += ft._tokenizer.id_to_token(next_pred_ind)
        #updating prediction
        y_pred = model(ft(name[-1], padding = 'do_not_pad', return_tensors='pt')['input_ids']).softmax(dim = 1).squeeze(dim = 0)
    return name
    



#deploying the model
LANGS = ['irish',
 'chinese',
 'scottish',
 'german',
 'russian',
 'greek',
 'portuguese',
 'korean',
 'dutch',
 'french',
 'spanish',
 'polish',
 'japanese',
 'czech',
 'vietnamese',
 'arabic',
 'italian',
 'english']

BUTTONS = [[KeyboardButton(lang)] for lang in LANGS]

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi! I am a bot that generates name. Type the language you want and I will generate a name for you",
                                   reply_markup=ReplyKeyboardMarkup(BUTTONS))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_language = update.message.text
    model_prediction = model_inf(current_language)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Here is your generated name in {current_language} language: {model_prediction}',
                            reply_markup=ReplyKeyboardMarkup(BUTTONS))



if __name__ == '__main__':
    application = ApplicationBuilder().token('MY_TOKEN').build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT,handle_message))
    
    application.run_polling()

