### Take input pdb, score, repack and extract one and two body energies
import csv
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

df = pd.read_csv('filtered_file_two.csv')
two_energies = df['E_ij']

print(df['E_ij'].describe())


sns.histplot(two_energies, kde=True) 
plt.title('Filtered Two-Body Energy Distribution')
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.show()


df1 = pd.read_csv('filtered_file_two.csv')
one_energies = df1['E_ii']

sns.histplot(one_energies, kde=True) 
plt.title('Filtered One-Body Energy Distribution')
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.show()
