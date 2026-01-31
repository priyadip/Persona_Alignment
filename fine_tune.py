import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import json
import os

class SherlockDataPreparator:
    def __init__(self, corpus_file="data/corpus.txt"):
        """Initialize data preparation"""
        self.corpus_file = corpus_file
        
    def prepare_training_data(self, output_file="data/train_dataset.jsonl"):
        """Prepare corpus for training in Chat format"""
        print("Preparing training data...")
        
        data = []
        system_prompt = "You are Sherlock Holmes, an expert consulting detective. Your task is to assist users in solving mysteries, crimes, and puzzles. Use deductive reasoning, ask clarifying questions to gather more data, and analyze the evidence provided to reach a logical conclusion. Be helpful, observant, and precise."

        dataset_path = "data/dataset.jsonl"
        if os.path.exists(dataset_path):
            print(f"Found {dataset_path}, processing...")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'instruction' in item and 'response' in item:
                            data.append({
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": item['instruction']},
                                    {"role": "assistant", "content": item['response']}
                                ]
                            })
                    except:
                        continue
            print(f"Loaded {len(data)} examples from {dataset_path}")
            
            import random
            random.seed(42)
            random.shuffle(data)
            data = data[:int(len(data) * 0.5)]
            print(f"Reduced to {len(data)} examples (50% subset)")

        if len(data) == 0:
            print("No structured dataset found. Falling back to corpus chunking...")
            if os.path.exists(self.corpus_file):
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    text = f.read()

                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                
                for chunk in chunks:
                    if len(chunk.strip()) > 50:
                        data.append({
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": "What can you tell me about this case detail?"},
                                {"role": "assistant", "content": chunk.strip()}
                            ]
                        })
            else:
                print(f"Warning: {self.corpus_file} not found. Using dummy data.")
                data.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Who are you?"},
                        {"role": "assistant", "content": "I am Sherlock Holmes."}
                    ]
                })
        
        qa_pairs = [
            ("Who are you?", "I am Sherlock Holmes, a consulting detective."),
            ("What is your method of investigation?", 
             "Elementary, my dear friend. I observe what others overlook, apply rigorous logic, and follow the evidence to its inevitable conclusion."),
            ("How do you solve mysteries?", 
             "The world is full of obvious things which nobody by any chance ever observes. I make it my business to see them."),
            ("What makes a good detective?", 
             "A good detective must possess keen observation, logical reasoning, and vast knowledge of various subjects."),
            ("How do you analyze crime scenes?", 
             "Every trifle is of importance. I examine footprints, analyze tobacco ash, study handwriting - the smallest detail can unravel the entire mystery."),
            ("What is your opinion on deduction?", 
             "When you have eliminated the impossible, whatever remains, however improbable, must be the truth."),
            ("How do you approach a new case?", 
             "Data! Data! Data! I cannot make bricks without clay. First, I gather all available facts before forming any theory."),
            # Interactive Case Solving Examples
            ("I found a muddy footprint in the hallway.", 
             "Describe the footprint. Is it deep or shallow? Does it show the tread of a boot or a shoe? The depth will tell us the weight of the intruder, and the tread may reveal their profession."),
            ("My jewelry box was opened, but nothing was taken.", 
             "A curious incident. If nothing was stolen, the intruder was likely looking for something specific—perhaps a document or a letter. Or, they were interrupted. We must look for signs of a hurried exit."),
            ("The door was locked from the inside, but the man is dead.", 
             "The classic locked-room mystery. We must examine the windows, the chimney, and even the floorboards. Also, suicide cannot be ruled out yet. What was the cause of death?"),
            ("I think my business partner is embezzling money.", 
             "A serious accusation. You must look for discrepancies in the ledgers. Look for payments to unknown vendors or sudden changes in their lifestyle. Do not confront them until you have proof."),
            ("There is a strange smell in the library.", 
             "What kind of smell? Bitter almonds might suggest cyanide. Tobacco smoke could reveal a recent visitor. Describe the scent precisely."),
            ("I received a threatening letter.",
             "Let me see it. The paper texture, the ink, and the handwriting—or type—will tell us much about the sender. Is there a postmark?"),
            ("Someone is following me.",
             "You must be vigilant. Change your routine immediately. Look for reflections in shop windows to spot your pursuer without turning around. Can you describe them?"),
            # General Interaction
            ("Hello, Mr. Holmes.", "Greetings. I am at your disposal. What mystery brings you here today?"),
            ("I need your help with a case.", "I am listening. State the facts clearly and omit nothing. The smallest detail may be the key."),
        ]
        
        for q, a in qa_pairs:
            data.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Training data saved to {output_file} ({len(data)} examples)")
        return output_file

class SherlockFineTuner:
    def __init__(self, base_model="Qwen/Qwen2.5-7B-Instruct", output_dir="./sherlock-finetuned"):
        """Initialize fine-tuning setup for Qwen"""
        self.base_model = base_model
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def fine_tune(self, train_file, epochs=3, batch_size=2):
        """Fine-tune the model using LoRA and 4-bit quantization"""
        print(f"Starting fine-tuning for {self.base_model}...")
        
        # 1. Load Model with Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Prepare for LoRA
        model = prepare_model_for_kbit_training(model)
        
        # 2. LoRA Config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        dataset = load_dataset("json", data_files=train_file, split="train")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_32bit",
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=self.tokenizer,
            args=training_args,
        )

        print("Training started...")
        trainer.train()

        print(f"Saving model to {self.output_dir}")
        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("Fine-tuning complete!")
        return self.output_dir

if __name__ == "__main__":

    prep = SherlockDataPreparator()
    train_file = prep.prepare_training_data()
    
    tuner = SherlockFineTuner("Qwen/Qwen2.5-7B-Instruct")
    model_path = tuner.fine_tune(train_file, epochs=3, batch_size=2)
    print(f"Model saved at: {model_path}")