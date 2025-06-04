import pandas as pd
import glob
import os

# Define your source and target directories
source_dir = "/Users/aag/Downloads/ProjectAnastasia/RESULTS/qH_depths/"
target_dir = "/Users/aag/Downloads/ProjectAnastasia/RESULTS/qH_depths/"


### For total depths
# # Pattern match the relevant files
# nopenalty_files = glob.glob(os.path.join(source_dir, "total_cnots_*rots_nopenalty.csv"))
# penalty_files = glob.glob(os.path.join(source_dir, "total_cnots_*rots_penalties.csv"))

# # Load and concatenate nopenalty
# df_nopenalty = pd.concat([pd.read_csv(f) for f in nopenalty_files], ignore_index=True)
# df_nopenalty = df_nopenalty.sort_values(by="Size")
# df_nopenalty.drop_duplicates()
# df_nopenalty.to_csv(os.path.join(target_dir, "total_cnots.csv"), index=False)

# # Load and concatenate penalties
# df_penalty = pd.concat([pd.read_csv(f) for f in penalty_files], ignore_index=True)
# df_penalty = df_penalty.sort_values(by="Size")
# df_penalty.drop_duplicates()
# df_penalty.to_csv(os.path.join(target_dir, "total_cnots_penalties.csv"), index=False)

# print("✅ Files successfully merged and saved.")


### For qH depths
# # Pattern match the relevant files
# nopenalty_files = glob.glob(os.path.join(source_dir, "qH_depth_*rots_nopenalty.csv"))
# penalty_files = glob.glob(os.path.join(source_dir, "qH_depth_*rots_penalties.csv"))

# # Load and concatenate nopenalty
# df_nopenalty = pd.concat([pd.read_csv(f) for f in nopenalty_files], ignore_index=True)
# df_nopenalty = df_nopenalty.sort_values(by="Size")
# df_nopenalty.drop_duplicates()
# df_nopenalty.to_csv(os.path.join(target_dir, "qH_depths.csv"), index=False)

# # Load and concatenate penalties
# df_penalty = pd.concat([pd.read_csv(f) for f in penalty_files], ignore_index=True)
# df_penalty = df_penalty.sort_values(by="Size")
# df_penalty.drop_duplicates()
# df_penalty.to_csv(os.path.join(target_dir, "qH_depths_penalties.csv"), index=False)

# print("✅ Files successfully merged and saved.")



# ### For baseline depths
# # Pattern match the relevant files
# baseline_files = glob.glob(os.path.join(source_dir, "total_cnots_*rots_baseline.csv"))

# # Load and concatenate nopenalty
# df_baseline = pd.concat([pd.read_csv(f) for f in baseline_files], ignore_index=True)
# df_baseline = df_baseline.sort_values(by="Size")
# df_baseline.drop_duplicates()
# df_baseline.to_csv(os.path.join(target_dir, "baseline.csv"), index=False)


### For state prep depths
# Pattern match the relevant files
baseline_files = glob.glob(os.path.join(source_dir, "total_cnots_*rots_state_prep.csv"))

# Load and concatenate nopenalty
df_baseline = pd.concat([pd.read_csv(f) for f in baseline_files], ignore_index=True)
df_baseline = df_baseline.sort_values(by="Size")
df_baseline.drop_duplicates()
df_baseline.to_csv(os.path.join(target_dir, "state_prep.csv"), index=False)

