#!/usr/bin/env python2
import numpy as np
import torch

class SmoothNoise():
    def __init__(self,shape=(1,)):
        self.shape = shape
        self.x = np.zeros(shape)
        self.mean = np.zeros(shape)
        self.return_speed = 0.01
        self.std = 0.03


    def sample(self):
        self.x += self.return_speed * (self.mean-self.x) + np.random.normal(0,self.std,self.shape)
        return torch.tensor(self.x).float()
        
if __name__ == "__main__":
    print("Hello There")
    sn = SmoothNoise((1,))
    x = []
    for i in range(1000):
        x.append(float(sn.sample()))

    from matplotlib import pyplot as plt

    plt.plot(x)
    plt.show()
