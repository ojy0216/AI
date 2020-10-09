import matplotlib.pyplot as plt
import numpy as np

load = np.load('e-greedy.npy').T

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward return', color=color)
ax1.plot(load[1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('epsilon', color=color)
ax2.plot(load[0], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()

plt.xlim(0, 100)
# ax1.ylim(np.min(load[1]) - 10, np.max(load[1]) + 10)
plt.show()
