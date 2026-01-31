import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class DetectiveChatbot:
    def __init__(self, model_path="sherlock-finetuned", base_model_name="Qwen/Qwen2.5-7B-Instruct"):
        """Initialize the chatbot with fine-tuned model"""
        print("Loading fine-tuned detective model...")
        print(f"Adapter path: {model_path}")
        print(f"Base model: {base_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
        
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print("=" * 60)
        
    def generate_response(self, user_input, history=None, max_length=200, temperature=0.7, top_p=0.9):
        """Generate response from the detective bot"""

        system_prompt = "You are Sherlock Holmes, an expert consulting detective. Your task is to assist users in solving mysteries, crimes, and puzzles. Use deductive reasoning, ask clarifying questions to gather more data, and analyze the evidence provided to reach a logical conclusion. Be helpful, observant, and precise."
        
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)
            
        messages.append({"role": "user", "content": user_input})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def chat(self):
        """Start interactive chat session"""
        print("\nDetective Chatbot Loaded!")
        print("=" * 60)
        print("You can now chat with the detective.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 60 + "\n")
        
        chat_history = []
        log_history = []
        
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\nDetective: Farewell! Until our paths cross again.")
                break
            
            if not user_input:
                continue

            try:
                response = self.generate_response(user_input, history=chat_history)
                print(f"\nDetective: {response}\n")

                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response})

                log_history.append({
                    "user": user_input,
                    "detective": response
                })
                
            except Exception as e:
                print(f"\nError generating response: {e}\n")
                continue
        
        print("\n" + "=" * 60)
        print(f"Conversation ended. Total exchanges: {len(log_history)}")
        print("=" * 60)
        
        return log_history


def quick_test(chatbot):
    """Run a quick test with predefined questions"""
    print("\n" + "=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60 + "\n")
    
    test_questions = [
        "Who are you?",
        "What year is it?",
        "A body was found in a locked room. What should I do?",
        "I found a cigarette butt at the crime scene. What can you deduce?",
        "How do you solve mysteries?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"[Test {i}] You: {question}")
        response = chatbot.generate_response(question)
        print(f"Detective: {response}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    chatbot = DetectiveChatbot()

    print("\nChoose mode:")
    print("1. Interactive Chat (chat with the detective)")
    print("2. Quick Test (run predefined test questions)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        quick_test(chatbot)
    else:
        chatbot.chat()
