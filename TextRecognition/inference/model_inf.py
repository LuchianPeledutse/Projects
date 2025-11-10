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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_WEIGHTS_PATH = os.getenv('MODEL_WEIGHTS_PATH')
TOKENIZER_OBJECT_PATH = os.getenv('TOKENIZER_OBJECT_PATH')
MODEL_OBJECT_PATH = os.getenv('MODEL_OBJECT_PATH')
PROCESSOR_OBJECT_PATH = os.getenv('PROCESSOR_OBJECT_PATH')

#loading model, processor, and tokenizer obejects
with open(TOKENIZER_OBJECT_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

with open(PROCESSOR_OBJECT_PATH, 'rb') as f:
    processor = pickle.load(f)

#cpu is the default device
#transform for PIL pictures
def vit_trans(pil_picture, processor=processor, device='cpu'):
    return processor(pil_picture.convert("RGB"), return_tensors="pt").pixel_values[0].unsqueeze(dim=0).to(device=device)


def model_inference(pillow_picture, model, tokenizer, processor, limit=64):
    """
    Description
    -----------
    Given an image returns generated text
    """
    model.eval()
    tens_pic = processor(pillow_picture)
    gen = model.generate(tens_pic, do_sample=True, top_k=50, top_p=0.95, max_length=limit)
    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].split(' ')
    return ''.join(decoded)
    

#loading model weights
MODEL_NAME = 'microsoft/trocr-small-handwritten'
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, pad_token_id=0, bos_token_id=2, eos_token_id=3).to(dtype=torch.float16, device='cpu')
model.decoder.get_input_embeddings().padding_idx = 0
model.decoder.resize_token_embeddings(40)

model.float()
model.load_state_dict(torch.load(f=MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()

#request bodies
class InputData(BaseModel):
    """Request body for incoming data input"""
    image_64_base: str


#API
app = fastapi.FastAPI()


@app.post('/prediction')
async def predict(data: InputData):
    image = Image.open(io.BytesIO(base64.b64decode(data.image_64_base))).convert("RGB")
    prediction = model_inference(pillow_picture=image, model=model, tokenizer=tokenizer, processor=vit_trans, limit=64)
    return {"generated_text": prediction}


