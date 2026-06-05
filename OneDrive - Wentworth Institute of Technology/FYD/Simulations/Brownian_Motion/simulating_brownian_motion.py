import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Parameters for the normal distribution
n_steps = 1000
n_paths = 5
dt = 0.01

t = np.linspace(0, n_steps * dt,n_steps + 1)
for i in range(n_paths):
    steps = np.random.normal(loc=0, scale=np.sqrt(dt), size=n_steps)
    path = np.concatenate(([0], np.cumsum(steps)))
    plt.plot(t, path, alpha = 0.7, label =f"path{i+1}")

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Brownian Motion Simulation')
plt.legend()
plt.grid(True)
plt.show()