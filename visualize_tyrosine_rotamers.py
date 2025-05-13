import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.graphics as graphics
import matplotlib.pyplot as plt
import numpy as np
import os

def center_structure(atom_array):
    centroid = np.mean(atom_array.coord, axis=0)
    atom_array.coord -= centroid
    return atom_array

# CPK color mapping
cpk_colors = {
    'H': [1.00, 1.00, 1.00],
    'C': [0.50, 0.50, 0.50],
    'N': [0.00, 0.00, 1.00],
    'O': [1.00, 0.00, 0.00],
    'S': [1.00, 1.00, 0.00],
    'P': [1.00, 0.50, 0.00],
}

def get_atom_colors(atom_array):
    return np.array([cpk_colors.get(el.capitalize(), [1.0, 0.5, 0.5]) for el in atom_array.element])

# Load and center rotamers
rotamer_files = sorted([f for f in os.listdir("tyr_rotamers") if f.endswith(".pdb")])[:9]
rotamers = []
for f in rotamer_files:
    rotamer = strucio.load_structure(os.path.join("tyr_rotamers", f))
    rotamer.bonds = struc.connect_via_residue_names(rotamer)
    rotamers.append(center_structure(rotamer))

# Plot
fig, axes = plt.subplots(3, 3, subplot_kw={"projection": "3d"}, figsize=(9, 9))
axes = axes.flatten()

for rotamer, ax in zip(rotamers, axes):
    atom_colors = get_atom_colors(rotamer)
    n_bonds = rotamer.bonds.as_array().shape[0]
    bond_colors = np.full((n_bonds, 3), 0.4)  # grey RGB for bonds
    colors = np.vstack([atom_colors, bond_colors])

    graphics.plot_atoms(ax, rotamer, colors=colors, line_width=1.5)
    ax.view_init(elev=20, azim=135)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_axis_off()

fig.suptitle("Rotamers of Tyrosine", fontsize=16, weight="bold")
plt.tight_layout()
plt.show()

