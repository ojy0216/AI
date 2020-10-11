import matplotlib.pyplot as plt
import numpy as np

load = np.load('7by7-decaying.npy')
load1 = np.load('7by7.npy')

fig, ax1 = plt.subplots(figsize=(8, 5))

plt.plot(load1, label='e-greedy')
plt.plot(load, label='decaying e-greedy')

fig.tight_layout()

plt.xlim(0, 100)
plt.xlabel("Episode")
plt.ylabel("Reward return")
plt.legend()
plt.tight_layout()
plt.show()
