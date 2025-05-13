import pandas as pd
import glob
import os

# Define your source and target directories
source_dir = "/Users/aag/Downloads/ProjectAnastasia/RESULTS/qH_depths/"
target_dir = "/Users/aag/Downloads/ProjectAnastasia/RESULTS/qH_depths/"

# # Pattern match the relevant files
# nopenalty_files = glob.glob(os.path.join(source_dir, "total_depth_*rots_nopenalty.csv"))
# penalty_files = glob.glob(os.path.join(source_dir, "total_depth_*rots_penalties.csv"))

# # Load and concatenate nopenalty
# df_nopenalty = pd.concat([pd.read_csv(f) for f in nopenalty_files], ignore_index=True)
# df_nopenalty = df_nopenalty.sort_values(by="Size")
# df_nopenalty.to_csv(os.path.join(target_dir, "total_depths.csv"), index=False)

# # Load and concatenate penalties
# df_penalty = pd.concat([pd.read_csv(f) for f in penalty_files], ignore_index=True)
# df_penalty = df_penalty.sort_values(by="Size")
# df_penalty.to_csv(os.path.join(target_dir, "total_depths_penalties.csv"), index=False)

# print("✅ Files successfully merged and saved.")


# Pattern match the relevant files
nopenalty_files = glob.glob(os.path.join(source_dir, "qH_depth_*rots_nopenalty.csv"))
penalty_files = glob.glob(os.path.join(source_dir, "qH_depth_*rots_penalties.csv"))

# Load and concatenate nopenalty
df_nopenalty = pd.concat([pd.read_csv(f) for f in nopenalty_files], ignore_index=True)
df_nopenalty = df_nopenalty.sort_values(by="Size")
df_nopenalty.to_csv(os.path.join(target_dir, "qH_depths.csv"), index=False)

# Load and concatenate penalties
df_penalty = pd.concat([pd.read_csv(f) for f in penalty_files], ignore_index=True)
df_penalty = df_penalty.sort_values(by="Size")
df_penalty.to_csv(os.path.join(target_dir, "qH_depths_penalties.csv"), index=False)

print("✅ Files successfully merged and saved.")
