import math
import torch
import random

def generate_batch_indicies(batch_size,dataset_length):

    indicies = list(range(dataset_length))
    
    random.shuffle(indicies)

    for i1 in range(0,dataset_length,batch_size):
        i2 = min(i1 + batch_size,dataset_length)

        yield torch.tensor(indicies[i1:i2])

