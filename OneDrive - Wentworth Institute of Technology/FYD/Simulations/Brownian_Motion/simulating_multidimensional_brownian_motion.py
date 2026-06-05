import numpy as np
import matplotlib.pyplot as plt

N = 1000
T = 10
dt = T/N
m = 2
n = 3

V = np.array([
    [0.3, 0.1, 0.05],
    [0.05, 0.2, 0.4]
])

assert V.shape == (m,n)
np.random.seed(42)

dW = np.random.normal(0, np.sqrt(dt), size = (n,N))
dX = V @ dW

X = np.zeros((m, N+1))
X[:,1:] = np.cumsum(dX, axis = 1)

t = np.linspace(0,T, N+1)

cov_theroitical = V @ V.T
cov_emperical = np.cov(dX)

print("Volitility Matrix V (2x3)\n")
print(V, "\n")
print("Theoritical Covariance matrix")
print(cov_theroitical, "\n")
print("Empirical Covariance matrix")
print(cov_emperical, "\n")

fig, axes = plt.subplots(1,2,figsize=(12,6), sharex = True)
colors = ["steelblue", "tomato"]

for i in range(m):
    axes[i].plot(t, X[i], color = colors[i], linewidth = 1.2)
    axes[i].axhline(0, color = "black", linewidth = 0.5, linestyle = "--")
    axes[i].set_ylabel(f"Asset {i} Value", fontsize = 12)
    axes[i].set_title(f"Asset {i} || Volitility Row {V[i]} || Total Variance = {cov_theroitical[i,i]: .4f}", fontsize  = 12)
    axes[i].grid(True, alpha = 0.3)

axes[-1].set_xlabel("Time", fontsize = 12)
plt.suptitle(f"Multidimensional Brownian Motion \n m = {m} assets, n = {n} sources of randomness \n  T = {T} N = {N} steps", 
 fontsize = 14, fontweight = "bold")
plt.tight_layout()
plt.show()


