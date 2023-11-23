import waveglow
import text
import audio

import torch
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

def synthesis(model, text, alpha=1.0, ):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(text_cleaners):
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    data_list = list(text.text_to_sequence(test, text_cleaners) for test in tests)

    return data_list
