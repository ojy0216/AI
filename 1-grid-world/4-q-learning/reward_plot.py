import matplotlib.pyplot as plt
import numpy as np

load = np.load('non-det_decaying_300.npy').T
load2 = np.load('non-det_ignore_decaying_300.npy')

fig, ax1 = plt.subplots(figsize=(8, 5))

plt.plot(load, label='New learning rule')
plt.plot(load2, label='Old learning rule')

plt.legend()
plt.xlim(0, 300)
plt.axhline(0, color='black', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Reward return')
fig.tight_layout()
plt.show()
