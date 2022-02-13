import os
import cv2
from particle_filter import Tracker
import numpy as np
import matplotlib.pyplot as plt


directory = "../../data/sequences-train/"
photos = ["bag", "bear", "book", "camel", "rhino", "swan"]

def test(default_params, param_test_name, param_test_list, display=True, n_test=5):
    test_scores = []
    for param in param_test_list:
        print(param)
        default_params[param_test_name] = param
        scores_param = [[] for i in range(n_test)]
        for i in range(n_test):
            T = Tracker(**default_params)
            T.plot = False
            T.track()
            scores_param[i] = T.scores
        scores_param = np.array(scores_param).mean(axis=0)
        test_scores.append(scores_param)
    if display:
        for i, param in enumerate(param_test_list):
            plt.plot(test_scores[i], label="%s: %s"%(param_test_name, param))
        plt.xlabel("frame")
        plt.ylabel("Score")
        plt.legend()
        plt.title(name.split("/")[-1])
        plt.show()
    return test_scores

name = directory + photos[0] # testing with bag
scores = test({"n_particles": 20, "data_dir": "../../data/sequences-train"}, "name", ["bag", "bear", "book", "camel", "rhino", "swan"], n_test=5)
