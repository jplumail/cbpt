import os
import cv2
from particle_filter import Tracker
import numpy as np

def create_legend(img,pt1,pt2):
    text1 = "Before resampling"
    cv2.putText(img,text1, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    text2 = "After resampling"
    cv2.putText(img,text2, pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    
def main():
    directory = "../../data/"
    name = directory + 'sequences-train/bag'
    T = Tracker(name, n_particles=100, dt=0.1, firstMask=True)
    T.track(5)

if __name__=="__main__":
    main()
