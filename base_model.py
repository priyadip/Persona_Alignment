import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class BaseModelTester:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """Initialize base model"""
        print(f"Loading base model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, prompt, max_length=100, temperature=0.7):
        """Generate text from base model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def test_prompts(self):
        """Test base model with Sherlock-style prompts"""
        test_prompts = [
            "My dear Watson, I have observed",
            "The case presents several curious features:",
            "Elementary, my dear fellow. The solution is",
            "Upon examining the evidence, I deduce that"
        ]
        
        print("\n" + "="*60)
        print("BASE MODEL RESPONSES")
        print("="*60)
        
        results = []
        for prompt in test_prompts:
            response = self.generate_response(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 60)
            results.append({"prompt": prompt, "response": response})
        
        # Save results
        with open("base_model_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results


if __name__ == "__main__":

    tester = BaseModelTester("Qwen/Qwen2.5-7B-Instruct")
    tester.test_prompts()