reinitialize
bg_color white
set stick_radius, 0.15
set ray_shadow, off
set ambient, 0.5
set ray_opaque_background, off
set stick_quality, 20
set ray_trace_mode, 0
set antialias, 2
set depth_cue, 0
set specular, off

# Load multi-model PDB into a single object with states
load rotamers.pdb, rotamers

# Style
hide everything
show sticks, rotamers

# Color by element
color green, rotamers and elem C
color red, rotamers and elem O
color blue, rotamers and elem N
color white, rotamers and elem H

# Optional: start animation
set all_states, on
mset 1 -9  # adjust to number of states
mplay
