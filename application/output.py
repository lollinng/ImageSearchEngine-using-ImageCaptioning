import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
import imageio
from PIL import Image
from application.seoptimization.caption_image_beam_search import caption_image_beam_search
from application.seoptimization.cosine import Doct2vect

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")



def run_query(text):
    model = 'application/Utils/BEST_checkpoint_flicker30k_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map = 'application/Utils/WORDMAP_flicker30k_5_cap_per_img_5_min_word_freq.json'

    beam_size = 5
    path = "application/static/test"
    
    checkpoint = torch.load(model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    
    images = os.listdir(path)
    
    reference_string = text
    result = []
    for img in images:
        img_path = 'application/static/test/'+ img      
        with open(word_map, 'r') as j:
            word_map_dict = json.load(j)
        rev_word_map = {v: k for k, v in word_map_dict.items()}  # ix2word
        # Encode, decode with attention and beam search
        seq= caption_image_beam_search(encoder, decoder, img_path, word_map_dict, beam_size)
        seq_text =  " ".join([rev_word_map[ind] for ind in seq][1:-1])
        result.append([seq_text,img])

    result = pd.DataFrame(result,columns=['seq_text','img'])     
    corpus = [reference_string]+list(result['seq_text'])

    # capturing similarties between seq_text with reference_string
    result['cosine_values'] = Doct2vect(corpus)[1:]  
    result['cosine_values']= result['cosine_values'].apply(lambda x: '%.2f' % float(x*100))

    result = result.values.tolist()
    result.sort(key=lambda x: float(x[2]) , reverse = True)

    return result
 
