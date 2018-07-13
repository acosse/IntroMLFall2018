
import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.utils import check_random_state
from sklearn import manifold, datasets
import numpy as np
Axes3D
from sklearn import manifold, datasets




## first showing the Swiss roll


n_neighbors = 10
n_samples = 4000

X1, color1 = datasets.samples_generator.make_swiss_roll(n_samples=4000)

Y1 = manifold.LocallyLinearEmbedding(n_neighbors, 2,method='standard').fit_transform(X1)


n_points = 4000

X2, color2 = datasets.samples_generator.make_s_curve(n_points, random_state=0)

Y2 = manifold.LocallyLinearEmbedding(n_neighbors, 2,method='standard').fit_transform(X2)

# Create our sphere.
random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
t = random_state.rand(n_samples) * np.pi

# Sever the poles from the sphere.
indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
colors = p[indices]
x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
    np.sin(t[indices]) * np.sin(p[indices]), \
    np.cos(t[indices])

sphere_data = np.array([x, y, z]).T

n_neighbors = 10
n_components = 2

Y3 = manifold.LocallyLinearEmbedding(n_neighbors, 2,method='standard').fit_transform(sphere_data).T


fig = plt.figure()

ax1 = fig.add_subplot(321, projection='3d')
ax1.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=color1, cmap=plt.cm.Spectral,edgecolor='k')
ax1.set_title('Original Manifold')
# To specify the number of ticks on both or any single axes
plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='z', nbins=3)


ax2 = fig.add_subplot(322)
ax2.scatter(Y1[:,0], Y1[:,1], c=color1, cmap=plt.cm.Spectral)
ax2.set_title('LLE Embedding')
plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='z', nbins=3)


ax3 = fig.add_subplot(323, projection='3d')
ax3.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=color2, cmap=plt.cm.Spectral,edgecolor='k')
ax3.set_title('Original Manifold')
plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='z', nbins=3)

ax4 = fig.add_subplot(324)
ax4.scatter(Y2[:,0], Y2[:,1], c=color2, cmap=plt.cm.Spectral)
ax4.set_title('LLE Embedding')
plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='z', nbins=3)


ax5 = fig.add_subplot(325, projection='3d')
ax5.scatter(sphere_data[:, 0], sphere_data[:, 1], sphere_data[:, 2], c=colors, cmap=plt.cm.rainbow,edgecolor='k')
ax5.set_title('Original Manifold')
plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='z', nbins=3)

ax6 = fig.add_subplot(326)
ax6.scatter(Y3[0], Y3[1],c=colors,cmap=plt.cm.rainbow)
ax6.set_title('LLE Embedding')
plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='z', nbins=3)


fig.tight_layout()
fig.savefig('/Users/acosse/Desktop/Teaching/Machine Learning/syllabus/Deep learning class/figures/LLE1')

plt.show()