import matplotlib.pyplot as plt
import numpy as np

load = np.load('7by7.npy')

fig, ax1 = plt.subplots(figsize=(8, 5))

plt.plot(load)

fig.tight_layout()

plt.xlim(0, 100)
plt.show()
