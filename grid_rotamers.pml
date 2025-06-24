reinitialize
bg_color white

# Load rotamers and split
load rotamers.pdb, rotamer
split_states rotamer

# Hide all, show sticks
hide everything
show sticks, all
set stick_radius, 0.06

# Apply individual rotations
rotate x, 0, object=rotamer_0001
rotate x, 10, object=rotamer_0002
rotate x, 20, object=rotamer_0003
rotate y, 5, object=rotamer_0004
rotate y, 15, object=rotamer_0005
rotate y, 25, object=rotamer_0006
rotate z, 10, object=rotamer_0007
rotate z, 20, object=rotamer_0008
rotate z, 30, object=rotamer_0009

# Translate to 3x3 grid
translate [0,  0, 0], object=rotamer_0001
translate [8,  0, 0], object=rotamer_0002
translate [16, 0, 0], object=rotamer_0003
translate [0,  8, 0], object=rotamer_0004
translate [8,  8, 0], object=rotamer_0005
translate [16, 8, 0], object=rotamer_0006
translate [0, 16, 0], object=rotamer_0007
translate [8, 16, 0], object=rotamer_0008
translate [16,16, 0], object=rotamer_0009

# Remove any existing labels (if already applied)
label all, ""

# Zoom and visual settings
zoom all
set stick_quality, 20
set ray_shadow, off
set ambient, 0.5
set ray_opaque_background, off
set depth_cue, 0
set antialias, 2
set specular, off

# Use standard atom coloring (CPK colors)
color carbon, elem C
color nitrogen, elem N
color oxygen, elem O
color hydrogen, elem H
