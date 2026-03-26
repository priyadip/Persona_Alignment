import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_PATH = "./Qwen2.5-7B"
MAX_HISTORY_TURNS = 10  # Limit history to prevent context overflow


class DetectiveChatbot:
    def __init__(self, model_path="sherlock-finetuned", base_model_name=BASE_MODEL_PATH):
        print("Loading fine-tuned Sherlock Holmes model...")
        print(f"Adapter path: {model_path}")
        print(f"Base model:   {base_model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )

            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model.eval()

            print("Model loaded successfully!")
            print("=" * 60)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, user_input, history=None, max_length=200, temperature=0.7, top_p=0.9):
        """Generate response as Sherlock Holmes using plain dialogue format"""

        # Build conversation context from history (limit to recent turns)
        context = ""
        if history:
            # Keep only the most recent turns to prevent context overflow
            recent_history = history[-MAX_HISTORY_TURNS:]
            for turn in recent_history:
                context += f"Human: {turn['human']}\nSherlock Holmes: {turn['sherlock']}\n\n"

        prompt = f"{context}Human: {user_input}\nSherlock Holmes:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Check if input is too long and truncate if necessary
        max_input_length = 2048 - max_length  # Reserve space for generation
        if inputs.input_ids.shape[1] > max_input_length:
            print(f"Warning: Input too long ({inputs.input_ids.shape[1]} tokens), truncating...")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length
            ).to(self.model.device)

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
        # Stop if model starts a new turn
        response = response.split("Human:")[0].strip()
        return response

    def chat(self):
        """Interactive chat session"""
        print("\nSherlock Holmes Chatbot")
        print("=" * 60)
        print("Type 'quit', 'exit', or 'bye' to end.")
        print(f"Note: Conversation history is limited to {MAX_HISTORY_TURNS} turns.")
        print("=" * 60 + "\n")

        history = []

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\nSherlock Holmes: Farewell! Until our paths cross again.")
                break

            if not user_input:
                continue

            try:
                response = self.generate_response(user_input, history=history)
                print(f"\nSherlock Holmes: {response}\n")
                history.append({"human": user_input, "sherlock": response})

                # Trim history if it exceeds the limit
                if len(history) > MAX_HISTORY_TURNS:
                    history = history[-MAX_HISTORY_TURNS:]

            except Exception as e:
                print(f"\nError: {e}\n")

        print(f"\nConversation ended. Total exchanges: {len(history)}")
        return history


def quick_test(chatbot):
    """Run predefined test questions non-interactively"""
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
        print(f"Sherlock Holmes: {response}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    chatbot = DetectiveChatbot()

    print("\nChoose mode:")
    print("1. Interactive Chat")
    print("2. Quick Test (predefined questions)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        quick_test(chatbot)
    else:
        chatbot.chat()
