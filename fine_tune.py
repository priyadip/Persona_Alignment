import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import json
import os
import random

BASE_MODEL_PATH = "./Qwen2.5-7B"

class SherlockDataPreparator:
    def __init__(self, corpus_file="corpus.txt"):
        self.corpus_file = corpus_file

    def prepare_training_data(self, output_file="train_dataset.jsonl", data_fraction=1.0):
        """Prepare corpus for training in plain dialogue format (no chat template)

        Args:
            output_file: Path to save the training data
            data_fraction: Fraction of dataset to use (1.0 = full dataset, 0.5 = half)
        """
        print("Preparing training data...")

        data = []

        # Load from structured dataset.jsonl
        dataset_path = "dataset.jsonl"
        if os.path.exists(dataset_path):
            print(f"Found {dataset_path}, processing...")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'instruction' in item and 'response' in item:
                            text = f"Human: {item['instruction']}\nSherlock Holmes: {item['response']}\n\n"
                            data.append({"text": text})
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(data)} examples from {dataset_path}")

            if data_fraction < 1.0:
                random.seed(42)
                random.shuffle(data)
                data = data[:int(len(data) * data_fraction)]
                print(f"Reduced to {len(data)} examples ({data_fraction*100:.0f}% subset)")

        # Fallback: corpus chunking
        if len(data) == 0:
            print("No structured dataset found. Falling back to corpus chunking...")
            if os.path.exists(self.corpus_file):
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                for chunk in chunks:
                    if len(chunk.strip()) > 50:
                        data.append({
                            "text": f"Human: What can you tell me about this case detail?\nSherlock Holmes: {chunk.strip()}\n\n"
                        })
            else:
                print(f"Warning: {self.corpus_file} not found. Using dummy data.")
                data.append({"text": "Human: Who are you?\nSherlock Holmes: I am Sherlock Holmes.\n\n"})

        # Curated Sherlock Q&A pairs (no system prompt — model learns persona directly)
        qa_pairs = [
            ("Who are you?",
             "I am Sherlock Holmes, the world's only consulting detective. When the police are out of their depth — which is always — they come to me."),
            ("What is your method of investigation?",
             "Elementary, my dear friend. I observe what others overlook, apply rigorous logic, and follow the evidence to its inevitable conclusion."),
            ("How do you solve mysteries?",
             "The world is full of obvious things which nobody by any chance ever observes. I make it my business to see them."),
            ("What makes a good detective?",
             "A good detective must possess keen observation, logical reasoning, and vast knowledge of various subjects."),
            ("How do you analyze crime scenes?",
             "Every trifle is of importance. I examine footprints, analyze tobacco ash, study handwriting — the smallest detail can unravel the entire mystery."),
            ("What is your opinion on deduction?",
             "When you have eliminated the impossible, whatever remains, however improbable, must be the truth."),
            ("How do you approach a new case?",
             "Data! Data! Data! I cannot make bricks without clay. First, I gather all available facts before forming any theory."),
            ("I found a muddy footprint in the hallway.",
             "Describe the footprint. Is it deep or shallow? Does it show the tread of a boot or a shoe? The depth will tell us the weight of the intruder, and the tread may reveal their profession."),
            ("My jewelry box was opened, but nothing was taken.",
             "A curious incident. If nothing was stolen, the intruder was likely looking for something specific — perhaps a document or a letter. Or they were interrupted. We must look for signs of a hurried exit."),
            ("The door was locked from the inside, but the man is dead.",
             "The classic locked-room mystery. We must examine the windows, the chimney, and even the floorboards. Also, suicide cannot be ruled out yet. What was the cause of death?"),
            ("I think my business partner is embezzling money.",
             "A serious accusation. You must look for discrepancies in the ledgers. Look for payments to unknown vendors or sudden changes in their lifestyle. Do not confront them until you have proof."),
            ("There is a strange smell in the library.",
             "What kind of smell? Bitter almonds might suggest cyanide. Tobacco smoke could reveal a recent visitor. Describe the scent precisely."),
            ("I received a threatening letter.",
             "Let me see it. The paper texture, the ink, and the handwriting — or type — will tell us much about the sender. Is there a postmark?"),
            ("Someone is following me.",
             "You must be vigilant. Change your routine immediately. Look for reflections in shop windows to spot your pursuer without turning around. Can you describe them?"),
            ("Hello, Mr. Holmes.",
             "Greetings. I am at your disposal. What mystery brings you here today?"),
            ("I need your help with a case.",
             "I am listening. State the facts clearly and omit nothing. The smallest detail may be the key."),
        ]

        for q, a in qa_pairs:
            data.append({"text": f"Human: {q}\nSherlock Holmes: {a}\n\n"})

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        print(f"Training data saved to {output_file} ({len(data)} examples)")
        return output_file


class SherlockFineTuner:
    def __init__(self, base_model=BASE_MODEL_PATH, output_dir="./sherlock-finetuned"):
        self.base_model = base_model
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def fine_tune(self, train_file, epochs=3, batch_size=2, max_seq_length=1024,
                  warmup_steps=100, use_gradient_checkpointing=True):
        """Fine-tune using LoRA + 4-bit quantization, pure persona style

        Args:
            train_file: Path to training data JSONL
            epochs: Number of training epochs
            batch_size: Per-device batch size
            max_seq_length: Maximum sequence length (default 1024 for longer contexts)
            warmup_steps: Learning rate warmup steps
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        print(f"Starting fine-tuning: {self.base_model}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        model = prepare_model_for_kbit_training(model)

        # Set max sequence length on tokenizer (replaces max_seq_length in newer TRL)
        self.tokenizer.model_max_length = max_seq_length
        self.tokenizer.truncation_side = "right"

        # Enable gradient checkpointing for memory efficiency with longer sequences
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        dataset = load_dataset("json", data_files=train_file, split="train")

        # Split into train/validation for monitoring overfitting
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            optim="paged_adamw_32bit",
            report_to="none",
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=self.tokenizer,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Training started...")
        resume_ckpt = os.environ.get("RESUME_CHECKPOINT", None)
        trainer.train(resume_from_checkpoint=resume_ckpt)

        print(f"Saving model to {self.output_dir}")
        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print("Fine-tuning complete!")
        return self.output_dir


if __name__ == "__main__":
    prep = SherlockDataPreparator()
    # Use full dataset (data_fraction=1.0) for best results
    train_file = prep.prepare_training_data(data_fraction=1.0)

    tuner = SherlockFineTuner(BASE_MODEL_PATH)
    model_path = tuner.fine_tune(
        train_file,
        epochs=3,
        batch_size=2,
        max_seq_length=1024,
        warmup_steps=100,
        use_gradient_checkpointing=True
    )
    print(f"Model saved at: {model_path}")
