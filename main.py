import os
import cv2
from particle_filter import Tracker
import numpy as np
import matplotlib.pyplot as plt

def load_mask():
    pass
    
def main():
    directory = "../../data/"
    name = directory + 'sequences-train/bag'
    T = Tracker(name, n_particles=1000, dt=0.1, firstMask=True, alpha=0.8, lbd=20)
    T.plot = False
    T.track(10)
    plt.plot(T.scores)
    plt.show()

if __name__=="__main__":
    main()
