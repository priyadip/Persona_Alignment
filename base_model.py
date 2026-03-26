import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json

BASE_MODEL_PATH = "./Qwen2.5-7B"


class BaseModelTester:
    def __init__(self, model_name=BASE_MODEL_PATH):
        print(f"Loading base model: {model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt, max_length=150, temperature=0.7):
        """Generate plain text completion from base model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Return only newly generated tokens
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def test_prompts(self):
        """Test base model with Sherlock-style prompts (pure completion, no system prompt)"""
        test_prompts = [
            "Human: Who are you?\nSherlock Holmes:",
            "Human: How do you solve mysteries?\nSherlock Holmes:",
            "Human: A body was found in a locked room. What do you deduce?\nSherlock Holmes:",
            "Human: I found a muddy footprint in the hallway.\nSherlock Holmes:",
        ]

        print("\n" + "="*60)
        print("BASE MODEL RESPONSES (no fine-tuning)")
        print("="*60)

        results = []
        for prompt in test_prompts:
            response = self.generate_response(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 60)
            results.append({"prompt": prompt, "response": response})

        with open("base_model_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\nResults saved to base_model_results.json")
        return results


if __name__ == "__main__":
    tester = BaseModelTester(BASE_MODEL_PATH)
    tester.test_prompts()
