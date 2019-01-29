import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd()

class PlotLossAndReward():
    def __init__(self, pause_time, out_dir=""):
        self.pause_time = 0.001
        self.out_dir = out_dir

    def __call__(self, loss, returns):
        plt.figure(3, figsize=(8, 4))
        plt.clf()
        plt.title('Learning Curve')

        plt.subplot(131)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(loss, 'r')

        plt.subplot(132)
        plt.xlabel('Episode')
        plt.ylabel('returns')
        plt.plot(returns, 'g')

        plt.subplot(133)
        avg_return = np.array([np.mean(returns[max(0, i-99):i+1]) for i in range(len(returns))])
        plt.xlabel('Episode')
        plt.ylabel('Average Returns')
        plt.plot(avg_return, 'b')

        plt.savefig(cwd + '/'+ self.out_dir + '/plot.png')
        plt.pause(self.pause_time)
