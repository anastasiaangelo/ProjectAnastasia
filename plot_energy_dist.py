### Take input pdb, score, repack and extract one and two body energies
import csv
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

df = pd.read_csv('two_body_terms.csv')
two_energies = df['E_ij']

print(df['E_ij'].describe())

fig, ax = plt.subplots()

sns.histplot(two_energies, kde=True) 
plt.title('Filtered Two-Body Energy Distribution')
plt.xlabel('Energy')
plt.ylabel('Frequency')

# For the zoom plot
x_zoom_range = [-10, 50] 
y_zoom_range = [0, 500] 

y_max_zoomed = 500 
y_min_zoomed = 0

coordsA = "data"
coordsB = "data"
xy1 = (x_zoom_range[0], y_max_zoomed)
xy2 = (x_zoom_range[1], y_min_zoomed)

ax = plt.gca()
ax_inset = inset_axes(ax, width="50%", height="50%", loc='upper right')

sns.histplot(two_energies, kde=True, ax=ax_inset)
ax_inset.set_xlim(x_zoom_range)
ax_inset.set_ylim(y_zoom_range) 

ax.indicate_inset_zoom(ax_inset)

con1 = ConnectionPatch(xyA=xy1, xyB=xy1, coordsA=coordsA, coordsB=coordsB, axesA=ax, axesB=ax_inset, color="black")
con2 = ConnectionPatch(xyA=xy2, xyB=xy2, coordsA=coordsA, coordsB=coordsB, axesA=ax, axesB=ax_inset, color="black")
ax.add_artist(con1)
ax.add_artist(con2)

plt.show()


df1 = pd.read_csv('filtered_file_one.csv')
one_energies = df1['E_ii']

sns.histplot(one_energies, kde=True) 
plt.title('Filtered One-Body Energy Distribution')
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.show()
