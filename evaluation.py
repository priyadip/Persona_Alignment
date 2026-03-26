import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import random
from tqdm import tqdm
import gc
from collections import Counter

BASE_MODEL_PATH = "./Qwen2.5-7B"


class ModelEvaluator:
    def __init__(self, base_model_name=BASE_MODEL_PATH, adapter_path="./sherlock-finetuned"):
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
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16
        )

        if use_adapter:
            print(f"Applying LoRA adapter from {self.adapter_path}")
            model = PeftModel.from_pretrained(model, self.adapter_path)

        model.eval()
        return model

    def calculate_conditional_perplexity(self, model, prompt, response):
        """Perplexity on the response tokens only, given the prompt (plain text)"""
        full_text = f"Human: {prompt}\nSherlock Holmes: {response}\n\n"
        prompt_text = f"Human: {prompt}\nSherlock Holmes:"

        full_enc = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        labels = full_enc.input_ids.clone()
        labels[:, :prompt_enc.input_ids.shape[1]] = -100

        with torch.no_grad():
            outputs = model(input_ids=full_enc.input_ids, labels=labels)
            loss = outputs.loss

        return torch.exp(loss).item()

    def calculate_content_similarity(self, generated, reference):
        """Jaccard similarity between generated and reference text"""
        g = set(generated.lower().split())
        r = set(reference.lower().split())
        if not g or not r:
            return 0.0
        return len(g.intersection(r)) / len(g.union(r))

    def calculate_rouge_l(self, generated, reference):
        """ROUGE-L score based on longest common subsequence"""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()

        if not gen_tokens or not ref_tokens:
            return 0.0

        lcs = lcs_length(gen_tokens, ref_tokens)
        precision = lcs / len(gen_tokens) if gen_tokens else 0
        recall = lcs / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def calculate_victorian_term_frequency(self, text):
        """Count frequency of Victorian/Holmesian terms"""
        victorian_terms = {
            "deduce", "deduction", "elementary", "observe", "observation",
            "trifle", "trifles", "indeed", "precisely", "singular",
            "remarkable", "curious", "affair", "crime", "mystery",
            "evidence", "clue", "suspect", "witness", "footprint",
            "tobacco", "ash", "magnifying", "watson", "inspector",
            "scotland", "yard", "baker", "street", "consulting",
            "detective", "case", "examine", "investigation", "theory"
        }
        words = text.lower().split()
        count = sum(1 for w in words if w in victorian_terms)
        return count / len(words) if words else 0.0

    def evaluate_dataset(self, dataset_path="dataset.jsonl", sample_size=30):
        print(f"\nLoading dataset: {dataset_path}")

        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        random.seed(42)
        random.shuffle(data)

        split_index = int(len(data) * 0.5)
        test_set = data[split_index:]

        print(f"Total examples: {len(data)}")
        print(f"Training set (seen): {split_index} examples")
        print(f"Test set (unseen):   {len(test_set)} examples")

        data = random.sample(test_set, sample_size) if len(test_set) > sample_size else test_set
        print(f"Evaluating on {len(data)} unseen samples\n")

        model = self.load_model(use_adapter=False)
        print("\nEvaluating Base Model...\n")
        base_results = self._run_eval_loop(model, data)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        model = self.load_model(use_adapter=True)
        print("\nEvaluating Fine-Tuned Model...\n")
        tuned_results = self._run_eval_loop(model, data)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return {"base": base_results, "tuned": tuned_results}

    def _run_eval_loop(self, model, data):
        metrics = {
            "perplexity": [],
            "similarity": [],
            "rouge_l": [],
            "victorian_freq": []
        }

        for item in tqdm(data):
            # Support both dataset formats
            if "messages" in item:
                prompt = next(m['content'] for m in item["messages"] if m["role"] == "user")
                reference = next(m['content'] for m in item["messages"] if m["role"] == "assistant")
            else:
                prompt = item["instruction"]
                reference = item["response"]

            # Perplexity (plain text, no chat template)
            try:
                ppl = self.calculate_conditional_perplexity(model, prompt, reference)
                metrics["perplexity"].append(ppl)
            except Exception as e:
                print(f"Warning: Perplexity calculation failed: {e}")
                continue

            # Generation (plain text) - DETERMINISTIC for reproducibility
            input_text = f"Human: {prompt}\nSherlock Holmes:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,  # Deterministic for reproducible evaluation
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # Stop at next "Human:" if model keeps going
            generated = generated.split("Human:")[0].strip()

            metrics["similarity"].append(self.calculate_content_similarity(generated, reference))
            metrics["rouge_l"].append(self.calculate_rouge_l(generated, reference))
            metrics["victorian_freq"].append(self.calculate_victorian_term_frequency(generated))

        return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}

    def print_report(self, results):
        print("\n" + "="*60)
        print("FINAL EVALUATION REPORT")
        print("="*60)

        rows = [
            ("Perplexity (lower=better)", "perplexity", False),
            ("Content Similarity (Jaccard)", "similarity", True),
            ("ROUGE-L Score", "rouge_l", True),
            ("Victorian Term Frequency", "victorian_freq", True),
        ]

        for title, key, higher_better in rows:
            base = results["base"].get(key, 0)
            tuned = results["tuned"].get(key, 0)
            diff = ((tuned - base) / base) * 100 if base != 0 else 0
            if not higher_better:
                diff = -diff
            print(f"{title:<30} | Base: {base:.4f} | Tuned: {tuned:.4f} | Change: {diff:+.2f}%")

        print("="*60)

        with open("evaluation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved to evaluation_report.json")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_dataset(sample_size=30)
    evaluator.print_report(results)
