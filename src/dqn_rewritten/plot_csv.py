import numpy as np
import matplotlib.pyplot as plt

csv_path = "cartpole_attempt_history.csv"
fig_path = "cartpole_attempt_history.png"

data = []
with open(csv_path, mode='r') as csv_file:
    lines = csv_file.readlines()
    for l in lines:
        data_in_l = l[:-1].split(',')
        for d in data_in_l:
            if d != '': data.append(float(d))

mean = np.mean(data)
sd = np.std(data)

# indices = list(range(len(data)))
#plt.plot(indices, data, '.', indices, mean_line, ':')
#plt.legend(["Training steps taken","Average"])
#plt.xlabel("Attempt")
#plt.ylabel("Training Steps Before Solving")

plt.hist(data,50)
plt.title("{} attempts to solve cart-pole v0".format(len(data)))
plt.xlabel("Number of episodes before solving")
textstr = "Mean: {:0.2f}\nSD: {:0.2f}".format(mean, sd)
plt.text(1500, 6, textstr, fontsize=10,)

plt.savefig(fig_path)