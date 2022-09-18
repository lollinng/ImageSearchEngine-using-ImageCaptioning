import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
import imageio
from PIL import Image
from application.seoptimization.caption_image_beam_search import caption_image_beam_search
from application.seoptimization.consine import texttocosine

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

        cosine_value = float("%.2f" % (texttocosine(reference_string,seq_text)*100))

        result.append([cosine_value,seq_text,img]) 

    result.sort(reverse = True)
    return result
 
