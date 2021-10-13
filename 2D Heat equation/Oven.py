import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from matplotlib import cm
from skimage import color
from skimage import io 
import numba
from numba import jit

edge = np.linspace(0, 1, 100)
xv, yv = np.meshgrid(edge, edge)

img = color.rgb2gray(io.imread("../../Github/Finite Difference Method/2D Heat equation/turkey.jpeg"))
img = np.flip(img, axis=0)
plt.contour(img)

turkey_bool = img<0.9
a_turk = 1.32e-7
rawturk_temp = 273.15+25
oven_temp = 273.15 + 165

init_heat = np.zeros([100, 100]) + oven_temp
init_heat[turkey_bool] = rawturk_temp
plt.contourf(init_heat)
plt.show()
