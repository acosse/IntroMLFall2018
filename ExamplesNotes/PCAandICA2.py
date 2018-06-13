

# uncomment if using the original student 


# import numpy as np
# import matplotlib.pyplot as plt
# import numpy.random as rnd
# from numpy import linalg as LA
# from matplotlib import colors as mcolors

# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)



# from sklearn.decomposition import PCA, FastICA

# # generate data from student distributions

# S = np.random.standard_t(1.5, size=(20000,2))
# S[:, 0] *= 2.


# # applying PCA and ICA on the data

# from sklearn.decomposition import PCA


# pca = PCA(n_components=2)
# pca.fit(S)

# ica = FastICA(n_components=2)
# ica.fit(S)




# # scatter plot
# plt.figure()

# eigenvector = pca.components_
# norm1 = LA.norm(eigenvector[0,:])
# norm2 = LA.norm(eigenvector[1,:])


# eigenvectorICA = ica.mixing_
# compICA = eigenvectorICA
# norm1ICA = LA.norm(compICA[0,:])
# norm2ICA = LA.norm(compICA[1,:])

# print(norm1ICA)

# print(compICA)


# f=plt.figure()


# #plt.arrow(0, 0, eigenvector[0,0]/norm1, eigenvector[0,1]/norm1, color='red')    
# plt.subplot(1, 2, 1)
# plt.scatter(S[:, 0]/S.std(), S[:, 1]/S.std(), s=2, marker='o', zorder=10,color='steelblue', alpha=0.5)
# plt.quiver(0, 0, eigenvector[0,0], eigenvector[0,1], color=colors['palevioletred'],zorder=11, width=0.01, scale=6)
# plt.quiver(0, 0, eigenvector[1,0], eigenvector[1,1], color=colors['palevioletred'],zorder=11, width=0.01, scale=6)
# #plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
# plt.hlines(0, -3, 3)
# plt.vlines(0, -3, 3)
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.xlabel('x')
# plt.ylabel('y')



# plt.subplot(1, 2, 2)
# plt.scatter(S[:, 0]/S.std(), S[:, 1]/S.std(), s=2, marker='o', zorder=10,color='steelblue', alpha=0.5)
# plt.quiver(0, 0, eigenvectorICA[0,0]/norm1ICA, eigenvectorICA[0,1]/norm1ICA, color=colors['darkseagreen'],zorder=11, width=0.01, scale=6)
# plt.quiver(0, 0, eigenvectorICA[1,0]/norm2ICA, eigenvectorICA[1,1]/norm2ICA, color=colors['darkseagreen'],zorder=11, width=0.01, scale=6)
# plt.hlines(0, -3, 3)
# plt.vlines(0, -3, 3)
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.xlabel('x')
# plt.ylabel('y')


# plt.show()
# f.savefig("ICAPCA.pdf", bbox_inches='tight')



import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from numpy import linalg as LA
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)



from sklearn.decomposition import PCA, FastICA

# generate data from student distributions

S = np.random.standard_t(1.5, size=(20000,2))
S[:, 0] *= 2.

mixingMat = np.array([[0,2],[1,1]])

S = np.dot(S, mixingMat)


# applying PCA and ICA on the data

from sklearn.decomposition import PCA


pca = PCA(n_components=2)
pca.fit(S)

ica = FastICA(n_components=2)
ica.fit(S)




# scatter plot
plt.figure()

eigenvector = pca.components_
norm1 = LA.norm(eigenvector[0,:])
norm2 = LA.norm(eigenvector[1,:])


eigenvectorICA = ica.mixing_
compICA = eigenvectorICA
norm1ICA = LA.norm(compICA[0,:])
norm2ICA = LA.norm(compICA[1,:])

print(norm1ICA)

print(compICA)


f=plt.figure()


#plt.arrow(0, 0, eigenvector[0,0]/norm1, eigenvector[0,1]/norm1, color='red')    
plt.subplot(1, 2, 1)
plt.scatter(S[:, 0]/S.std(), S[:, 1]/S.std(), s=2, marker='o', zorder=10,color='steelblue', alpha=0.5)
plt.quiver(0, 0, eigenvector[0,0], eigenvector[0,1], color=colors['palevioletred'],zorder=11, width=0.01, scale=6)
plt.quiver(0, 0, eigenvector[1,0], eigenvector[1,1], color=colors['palevioletred'],zorder=11, width=0.01, scale=6)
#plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
plt.hlines(0, -3, 3)
plt.vlines(0, -3, 3)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('PCA')



plt.subplot(1, 2, 2)
plt.scatter(S[:, 0]/S.std(), S[:, 1]/S.std(), s=2, marker='o', zorder=10,color='steelblue', alpha=0.5)
plt.quiver(0, 0, eigenvectorICA[0,0]/norm1ICA, eigenvectorICA[0,1]/norm1ICA, color=colors['darkseagreen'],zorder=11, width=0.01, scale=6)
plt.quiver(0, 0, eigenvectorICA[1,0]/norm2ICA, eigenvectorICA[1,1]/norm2ICA, color=colors['darkseagreen'],zorder=11, width=0.01, scale=6)
plt.hlines(0, -3, 3)
plt.vlines(0, -3, 3)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('ICA')


plt.show()
#f.savefig("ICAPCA.pdf", bbox_inches='tight')


