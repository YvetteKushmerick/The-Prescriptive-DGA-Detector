# Filename: 2_run_automl.py
!pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
import h2o
from h2o.automl import H2OAutoML

# Filename: 2_run_automl.py
h2o.init()

train = h2o.import_file("dga_dataset_train.csv") #train 
x = ['length', 'entropy'] # Features as x input
y = "class"               # Target as variables
train[y] = train[y].asfactor()

aml = H2OAutoML(max_models=20, max_runtime_secs=120, seed=1) #instanciating automl
aml.train(x=x, y=y, training_frame=train)

print("H2O AutoML process complete.")
print("Leaderboard (Top 10):")
print(aml.leaderboard.as_data_frame().head(10))

# (Add this to the end of 2_run_automl.py)
# Get the best performing model from the leaderboard
best_model = aml.leader

# Adding output directory
os.makedirs("./models", exist_ok=True)

# Download the MOJO artifact.
mojo_path = best_model.download_mojo(path="./models/")

# Adding try, except in the event there are errors
mojo_path = None
try:
    # Many models support MOJO; some (depending on build/type) may not.
    # get_genmodel_jar=True is handy if you plan to score with Java.
    mojo_path = best_model.download_mojo(path="./models/")
    print(f"MOJO saved to: {mojo_path}")
except Exception as e:
    print(f"Leader MOJO not available ({e}). Trying next MOJO-capable model...")
    # Walk down the leaderboard and pick the first MOJO-capable model
    lb_df = aml.leaderboard.as_data_frame()
    for mid in lb_df['model_id'].tolist():
        try:
            m = h2o.get_model(mid)
            mojo_path = m.download_mojo(path="./models/")
            print(f"MOJO saved from fallback model {mid} to: {mojo_path}")
            break
        except Exception:
            continue
    if mojo_path is None:
        print("No MOJO-capable model found. You can still use the binary model.")

# Save the binary model (always works)
model_path = h2o.save_model(model=best_model, path="./models", force=True)
print(f"Binary model saved to: {model_path}")

# After AutoML finishes
custom_name = "best_dga_model"
base, ext = os.path.splitext(model_path)          # keep .zip/.bin/etc
new_path = os.path.join("./models", custom_name + ext)
shutil.move(model_path, new_path)
print(f"Renamed model path: {new_path}")

if mojo_path:
    print(f"Production-ready MOJO at: {mojo_path}")
else:
    print("MOJO not produced; use the binary model path above for Python scoring.")

# Clean shutdown (non-interactive)
h2o.shutdown(prompt=False)