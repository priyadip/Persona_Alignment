import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import random
from tqdm import tqdm
import gc


class ModelEvaluator:
    def __init__(self, base_model_name="Qwen/Qwen2.5-7B-Instruct", adapter_path="./sherlock-finetuned"):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, use_adapter=False):
        print(f"\nLoading model (Adapter={use_adapter})...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )

        if use_adapter:
            print(f"Applying LoRA adapter from {self.adapter_path}")
            model = PeftModel.from_pretrained(model, self.adapter_path)

        model.eval()
        return model


    def calculate_conditional_perplexity(self, model, messages):

        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        full_enc = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        prompt_messages = messages[:-1] 
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        labels = full_enc.input_ids.clone()

        labels[:, :prompt_enc.input_ids.shape[1]] = -100

        with torch.no_grad():
            outputs = model(input_ids=full_enc.input_ids, labels=labels)
            loss = outputs.loss

        ppl = torch.exp(loss).item()
        return ppl

    def calculate_content_similarity(self, generated, reference):
        g = set(generated.lower().split())
        r = set(reference.lower().split())
        if not g or not r: return 0.0
        return len(g.intersection(r)) / len(g.union(r))

    def evaluate_dataset(self, dataset_path="data/dataset.jsonl", sample_size=30):
        print(f"\nLoading dataset: {dataset_path}")

        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    pass

        random.seed(42)
        random.shuffle(data)

        split_index = int(len(data) * 0.5)
        test_set = data[split_index:]
        
        print(f"Total examples: {len(data)}")
        print(f"Training set (seen): {split_index} examples")
        print(f"Test set (unseen):   {len(test_set)} examples")

        if len(test_set) > sample_size:
            data = random.sample(test_set, sample_size)
        else:
            data = test_set

        print(f"Evaluating on {len(data)} unseen samples\n")

        results = {
            "base": {"perplexity": [], "similarity": []},
            "tuned": {"perplexity": [], "similarity": []}
        }

        model = self.load_model(use_adapter=False)
        print("\nEvaluating Base Model...\n")
        results["base"] = self._run_eval_loop(model, data)
        del model ; gc.collect() ; torch.cuda.empty_cache()

        model = self.load_model(use_adapter=True)
        print("\nEvaluating Fine-Tuned Model...\n")
        results["tuned"] = self._run_eval_loop(model, data)
        del model ; gc.collect() ; torch.cuda.empty_cache()

        return results

    def _run_eval_loop(self, model, data):
        metrics = {"perplexity": [], "similarity": []}

        system_prompt = (
            "You are Sherlock Holmes, expert consulting detective. "
            "Collect facts, ask clarifying questions, analyze clues, and deduce logically."
        )

        for item in tqdm(data):
            if "messages" in item:
                prompt = next(m['content'] for m in item["messages"] if m["role"] == "user")
                reference = next(m['content'] for m in item["messages"] if m["role"] == "assistant")
            else:
                prompt = item["instruction"]
                reference = item["response"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            full_messages = messages + [{"role": "assistant", "content": reference}]
            ppl = self.calculate_conditional_perplexity(model, full_messages)
            metrics["perplexity"].append(ppl)

            user_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(user_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            metrics["similarity"].append(self.calculate_content_similarity(generated, reference))

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def print_report(self, results):
        print("\n" + "="*60)
        print("FINAL EVALUATION REPORT")
        print("="*60)

        rows = [
            ("Perplexity (lower=better)", "perplexity", False),
            ("Content Similarity", "similarity", True),
        ]

        for title, key, higher_better in rows:
            base = results["base"][key]
            tuned = results["tuned"][key]

            if base != 0:
                diff = ((tuned - base) / base) * 100
            else:
                diff = 0

            if not higher_better:
                diff = -diff

            print(f"{title:<30} | Base: {base:.4f} | Tuned: {tuned:.4f} | Change: {diff:+.2f}%")

        print("="*60)

        with open("evaluation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved to 'evaluation_report.json'")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_dataset(sample_size=30)
    evaluator.print_report(results)