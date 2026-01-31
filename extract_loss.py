import json
import os

checkpoint_path = "sherlock-finetuned/checkpoint-2484/trainer_state.json"
output_path = "training_loss.json"

if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    
    log_history = data.get("log_history", [])
    
    with open(output_path, "w") as f:
        json.dump(log_history, f, indent=4)
    
    print(f"Successfully extracted log_history to {output_path}")
else:
    print(f"Checkpoint file not found at {checkpoint_path}")