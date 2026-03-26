import json
import os
import glob

model_dir = "sherlock-finetuned"
output_path = "training_loss.json"

checkpoints = sorted(
    glob.glob(f"{model_dir}/checkpoint-*"),
    key=lambda x: int(x.split("-")[-1])
)

if not checkpoints:
    print(f"No checkpoints found in '{model_dir}'. Skipping loss extraction.")
else:
    latest = checkpoints[-1]
    checkpoint_path = os.path.join(latest, "trainer_state.json")
    print(f"Using checkpoint: {latest}")

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        log_history = data.get("log_history", [])

        with open(output_path, "w") as f:
            json.dump(log_history, f, indent=4)

        print(f"Successfully extracted {len(log_history)} log entries to {output_path}")
    else:
        print(f"trainer_state.json not found at {checkpoint_path}")
