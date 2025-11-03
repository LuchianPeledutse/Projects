import io
import os
import math
import base64
import pickle

from PIL import Image

import fastapi
from pydantic import BaseModel

import torch
import torchvision
import torchsummary
from torch.nn import utils
from torch import nn, optim, onnx, distributions
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional

#global parameters
D_MODEL = 516
N_HEADS = 6
NUM_LAYERS = 4

MODEL_WEIGHTS_PATH = os.getenv('MODEL_WEIGHTS_PATH')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')

#model data, processing tools, and functions

#transform for PIL pictures
def resnet_trans(pil_image):
    """
    Parameters
    ----------
    pil_image: image opened with pillow library 
    
    returns
    ----------
    Tensor image transformed for ResNet; with dtype float32
    """
    #apply necessary transformations for ResNet
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transforms(pil_image)

#transform for words
def make_spaces(word):
    """Makes spaces between the letter of a given word"""
    return ' '.join(list(word))

#function for positional embeddings
def pos_embeds(embedding_matrix, dev='cpu'):
    """
    Parameters
    ----------
    embedding matrix: torch.Tensor of shape (batch_size, seq_len, embedding_dim)

    returns
    -------
    embedding matrix + positional encoding
    same embedding matrix with added positional embeddings
    """
    batch_size, seq_len, d_model = embedding_matrix.size()
    position = torch.arange(seq_len, device=dev).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=dev) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=dev)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return embedding_matrix + pe.unsqueeze(0)

#model inference
def model_inference(pillow_picture, model, tokenizer, word_transform=make_spaces, picture_transform=resnet_trans, device='cpu', limit=100):
    """
    Description
    -----------
    Given an image returns generated text
    """
    #change model state to evaluation
    model.eval()
    #transform picture for forward pass
    tensor_picture = picture_transform(pillow_picture).unsqueeze(dim = 0).to(device = device)
    #creating starting sequence
    sequence = ''
    #while last token does not equal to eos or limit achieved keep generating
    iter_count = 0
    while True and iter_count < limit:
        #update count
        iter_count += 1
        sequence_tensor = torch.tensor(tokenizer(word_transform(sequence)).input_ids[:-1]).reshape(1,-1).to(device = device)
        y_pred = model(x_pic_tensor = tensor_picture, x_letter_ids = sequence_tensor).squeeze(dim = 1)[-1].softmax(dim = 0) #choose the last token probabilities
        #make distribution over vocabulary tokens
        next_sampled_token = y_pred.argmax(dim = 0).detach().cpu().item()
        #if we reached the last token we break (else continute)
        if next_sampled_token == tokenizer._tokenizer.get_vocab()['<eos>']:
            break
        else:
            #add new token
            sequence += tokenizer._tokenizer.id_to_token(next_sampled_token)
    return sequence


#main model
#Enocder-Decoder model for TextRecognition 
class TRModel(nn.Module):
    def __init__(self, vocabulary_size, d_model = D_MODEL, nhead = N_HEADS, num_layers = NUM_LAYERS, dropout = 0.5, padding_indx = 0, device = 'cpu', train=True, freeze_params=30):
        super().__init__()
        self.device = device
        #Encoder
        #load ResNet with last two linear layers removed
        self.encoder = nn.Sequential(*list(torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1, progress = False).children())[:-2])
        #freeze some layesr
        count = 0
        for parameter in self.encoder.parameters():
            if count > freeze_params:
                break
            parameter.requires_grad = False
            count += 1

        #linear head that project to embedding dimension
        self.linear_encoder_head = nn.Linear(512,d_model)

        #Decoder
        #embedding layer
        self.embed_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=d_model, padding_idx=padding_indx)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        #linear head over vocabulary
        self.linear_head = nn.Linear(d_model,vocabulary_size)
    
    def generate_mask(self,N):
        """Generates a mask for transformer decoder self-attention"""
        mask = (torch.triu(torch.ones(N,N)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x_pic_tensor, x_letter_ids):
        """
        parameters
        ----------
        x_pic: tensor of shape batch_size X channels X H x W
        x_text: tensor of shape seq_len X batch_size X d_model (embed_dim)
        """
        encoder_forward = self.encoder(x_pic_tensor)
        encoder_forward = encoder_forward.reshape(encoder_forward.shape[0], encoder_forward.shape[1], -1)
        #reshape to shape (batch_size, spatial_sequence_len, num_channels)
        encoder_forward = encoder_forward.transpose(1,2)
        #projecting number of channels to d_model shape
        encoder_forward = self.linear_encoder_head(encoder_forward)
        #adding positional embeddings
        encoder_forward = pos_embeds(encoder_forward, dev=self.device)
        #reshaping to shape (seq_len, batch_size, embed_dim (d_model))
        encoder_forward = encoder_forward.transpose(0,1)
        
        #getting vector embeddings
        letter_embeds = self.embed_layer(x_letter_ids) 
        #add positional embeddings and reshaping to seq_len first
        letter_embeds = pos_embeds(letter_embeds, dev=self.device).transpose(0,1)
        seq_len = letter_embeds.shape[0]
        #creating a mask for decoder forwarding
        tgt_mask = self.generate_mask(seq_len).to(self.device)
        #going through decoder
        decoder_forward = self.decoder(letter_embeds, memory=encoder_forward, tgt_mask=tgt_mask)
        clf_head = self.linear_head(decoder_forward)
        return clf_head
    
#laoding tokenizer
with open(TOKENIZER_PATH,'rb') as tokenizer_file:
    model_tokenizer = pickle.load(tokenizer_file)

#loading model weights
model = TRModel(vocabulary_size=model_tokenizer._tokenizer.get_vocab_size())
model.load_state_dict(torch.load(f=MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))

#request bodies
class InputData(BaseModel):
    """Request body for incoming data input"""
    image_64_base: str


#API
app = fastapi.FastAPI()


@app.post('/prediction')
async def predict(data: InputData):
    image = Image.open(io.BytesIO(base64.b64decode(data.image_64_base))).convert("RGB")
    prediction = model_inference(pillow_picture=image, model=model, tokenizer=model_tokenizer)
    return {"generated_text": prediction}


