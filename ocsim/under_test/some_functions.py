import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# def scatterplot_density(signal,interval):
#     # x = np.random.normal(size=500)
#     # y = x * 3 + np.random.normal(size=500)
#     #
#
#     device = signal.device
#
#     x = x[::4]
#     y = y[::4]
#     xy = np.vstack([x, y])
#     z = gaussian_kde(xy)(xy)
#
#     # Sort the points by density, so that the densest points are plotted last
#     idx = z.argsort()
#     x, y, z = x[idx], y[idx], z[idx]
#
#     fig, ax = plt.subplots()
#     ax.scatter(x, y, c=z)
#     plt.show()
#     print("Hellwo ")
