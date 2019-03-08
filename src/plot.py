import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd()

class PlotLossAndReward():
    def __init__(self, pause_time, out_dir=""):
        self.pause_time = 0.001
        self.out_dir = out_dir

    def __call__(self, loss, returns, step_counts):
        plt.figure(3, figsize=(8, 4))
        plt.clf()
        plt.title('Learning Curve')

        # plt.subplot(221)
        # plt.xlabel('Episode')
        # plt.ylabel('Loss')
        # plt.scatter(range(len(loss)), loss, c='r')

        plt.subplot(222)
        plt.xlabel('Episode')
        plt.ylabel('Average steps to catch')
        avg_step = np.array([np.mean(step_counts[max(0, i-50):i+1]) for i in range(len(step_counts))])
        plt.plot(avg_step, c='g')

        plt.subplot(223)
        avg_return = np.array([np.mean(returns[max(0, i-50):i+1]) for i in range(len(returns))])
        plt.xlabel('Episode')
        plt.ylabel('Average Returns')
        plt.plot(avg_return, 'b')

        # plt.subplot(224)
        # avg_loss = np.array([np.mean(loss[max(0, i-50):i+1]) for i in range(len(loss))])
        # plt.xlabel('Episode')
        # plt.ylabel('Average loss')
        # plt.plot(avg_loss, 'y')

        plt.savefig(cwd + '/'+ self.out_dir + '/plot.png')
        plt.pause(self.pause_time)
