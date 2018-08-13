import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Reproducible!
np.random.seed(42)

# A helper function to make the plots with error ellipses
def plot_error_ellipses(ax, X, S, color="k"):
    for n in range(len(X)):
        vals, vecs = np.linalg.eig(S[n])
        theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
        w, h = 2 * np.sqrt(vals)
        ell = Ellipse(xy=X[n], width=w, height=h,
                      angle=theta, color=color, lw=0.5)
        ell.set_facecolor("none")
        ax.add_artist(ell)
    ax.plot(X[:, 0], X[:, 1], ".", color=color, ms=4)

# Generate the true coordinates of the data points.
N = 10
m_true = 1.2
b_true = -0.1
X_true = np.empty((N, 2))
X_true[:, 0] = np.random.uniform(0, 10, N)
X_true[:, 1] = m_true * X_true[:, 0] + b_true
X = np.empty((N, 2))
# print (X_true)

# Generate error ellipses and add uncertainties to each point.
S = np.zeros((N, 2, 2))
for n in range(N):
    L = np.zeros((2, 2))
    L[0, 0] = np.exp(np.random.uniform(-1, 1))
    L[1, 1] = np.exp(np.random.uniform(-1, 1))
    L[1, 0] = 0.5 * np.random.randn()
    S[n] = np.dot(L, L.T)
    X[n] = np.random.multivariate_normal(X_true[n], S[n])
# print (X)
print (S)
# Plot the simulated dataset.
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
x0 = np.array([-2, 12])
ax.plot(x0, m_true*x0 + b_true, lw=1)
plot_error_ellipses(ax, X, S)
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 12)
ax.set_xlabel("x")
ax.set_ylabel("y");
plt.show()
