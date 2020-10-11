import matplotlib.pyplot as plt
import numpy as np

load = np.load('e-greedy.npy').T
load1 = np.load('decaying-e-greedy.npy').T

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward return', color=color)
ax1.plot(load[1], color=color, label='e-greedy')
ax1.plot(load1[1], color=color, linestyle='--', label='decaying e-greedy')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('epsilon', color=color)
ax2.plot(load[0], color=color)
ax2.plot(load1[0], color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()

plt.xlim(0, 100)
ax1.legend(loc='center')
plt.show()
