### Take input pdb, score, repack and extract one and two body energies
import csv
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

df = pd.read_csv('energy_files/two_body_terms.csv')
two_energies = df['E_ij']

print(df['E_ij'].describe())

fig, ax = plt.subplots()

sns.set_theme(style="ticks")
sns.histplot(two_energies, kde=True, bins=100, color='lightsteelblue', line_kws={'color': 'steelblue',  'linewidth': 1}) 
plt.title('Filtered Two-Body Energy Distribution')
plt.xlabel('Energy')
plt.ylabel('Frequency')

# # For the zoom plot
x_zoom_range = [-15, 50] 
y_zoom_range = [0, 15000] 

y_max_zoomed = 15000 
y_min_zoomed = 0

coordsA = "data"
coordsB = "data"
xy1 = (x_zoom_range[0], y_max_zoomed)
xy2 = (x_zoom_range[1], y_min_zoomed)

ax = plt.gca()
ax_inset = inset_axes(ax, width="50%", height="50%", loc='upper right')

sns.histplot(two_energies, kde=True, bins= 1500, ax=ax_inset, color='lightsteelblue', line_kws={'color': 'steelblue', 'linewidth': 2})
ax_inset.set_xlim(x_zoom_range)
ax_inset.set_ylim(y_zoom_range) 

ax.indicate_inset_zoom(ax_inset)

con1 = ConnectionPatch(xyA=xy1, xyB=xy1, coordsA=coordsA, coordsB=coordsB, axesA=ax, axesB=ax_inset, color="black")
con2 = ConnectionPatch(xyA=xy2, xyB=xy2, coordsA=coordsA, coordsB=coordsB, axesA=ax, axesB=ax_inset, color="black")
ax.add_artist(con1)
ax.add_artist(con2)

plt.savefig('two_body_distribution.pdf', format='pdf', bbox_inches='tight')
plt.clf()


df1 = pd.read_csv('energy_files/one_body_terms.csv')
one_energies = df1['E_ii']

print(df1['E_ii'].describe())

plt.figure()
sns.histplot(one_energies, kde=True, color='lightsteelblue', line_kws={'color': 'steelblue', 'linewidth': 2})
plt.title('Filtered One-Body Energy Distribution')
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.savefig('one_body_distribution.pdf', format='pdf', bbox_inches='tight')
