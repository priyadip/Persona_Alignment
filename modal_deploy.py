import modal
from pathlib import Path

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "scipy",
        "fastapi[standard]"
    )
    .add_local_dir("./sherlock-finetuned", remote_path="/root/sherlock-finetuned")
)

app = modal.App("sherlock-detective")

REMOTE_MODEL_DIR = "/root/sherlock-finetuned"

@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
)
class SherlockModel:
    @modal.enter()
    def enter(self):
        """This runs once when the container starts"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        base_model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype="auto"
        )

        print(f"Loading Sherlock adapter from {REMOTE_MODEL_DIR}...")

        self.model = PeftModel.from_pretrained(base_model, REMOTE_MODEL_DIR)
        self.model.eval()
        print("Sherlock is ready!")

    @modal.fastapi_endpoint(method="POST")
    def generate_web(self, data: dict):
        """Web endpoint for Streamlit"""
        prompt = data.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}
            
        import torch
        
        system_prompt = "You are Sherlock Holmes, an expert consulting detective. Your task is to assist users in solving mysteries, crimes, and puzzles. Use deductive reasoning, ask clarifying questions to gather more data, and analyze the evidence provided to reach a logical conclusion. Be helpful, observant, and precise."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"response": response}

    @modal.method()
    def generate(self, prompt: str):
        """This is the function you call remotely"""
        import torch
        
        system_prompt = "You are Sherlock Holmes, an expert consulting detective. Your task is to assist users in solving mysteries, crimes, and puzzles. Use deductive reasoning, ask clarifying questions to gather more data, and analyze the evidence provided to reach a logical conclusion. Be helpful, observant, and precise."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

# 4. Local entrypoint to test it
@app.local_entrypoint()
def main(prompt: str = "Watson, what do you make of this?"):
    print(f"Sending prompt to Modal: {prompt}")
    model = SherlockModel()
    response = model.generate.remote(prompt)
    print(f"\nSherlock (from Cloud): {response}")