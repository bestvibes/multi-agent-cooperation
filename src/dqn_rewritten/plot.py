import os
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd()

def compute_moving_averages(values, window_size):
    avgs = np.array([np.mean(values[max(0, i-window_size+1):i+1]) for i in range(len(values))])
    return avgs

class PlotTrainingHistoryCartPole():
    def __init__(self, fig_save_path, filename="cartpole_training_history.png"):
        self.fig_save_path = fig_save_path
        self.filename = filename
        
    def __call__(self, loss_history, score_history, score_avgs):
        plt.figure(3, figsize=(8, 4))
        plt.clf()
        plt.title('Learning History')
        
        plt.subplot(121)
        avg_losses = compute_moving_averages(loss_history, 100)
        plt.plot(avg_losses, 'r-')

        plt.subplot(122)
        plt.plot(score_history, 'g.', markersize=3)
        plt.plot(score_avgs, 'r:', linewidth=3)
        
        plt.savefig(cwd + '\\'+ self.fig_save_path + '\\' + self.filename)