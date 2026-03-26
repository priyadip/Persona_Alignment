#!/bin/bash
#SBATCH --job-name=sherlock_chat
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --qos=dgxqos
#SBATCH --output=/scratch/data/m25csa023/PersonaAllignment/chat_test_%j.log

CONDA_PYTHON="/scratch/data/m25csa023/conda/envs/dlops/bin/python"

module purge
module load anaconda3/2024
source ~/.bashrc
conda activate /scratch/data/m25csa023/conda/envs/dlops

cd /scratch/data/m25csa023/PersonaAllignment

${CONDA_PYTHON} - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "./Qwen2.5-7B"
ADAPTER    = "./sherlock-finetuned"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()
print("Model loaded!\n" + "="*60)

questions = [
    "Who are you?",
    "What do you think about Dr. Watson?",
    "A man was found dead in a locked room with no windows. How would you approach this case?",
    "What is your method of deduction?",
    "What are your thoughts on Scotland Yard?",
    "Mr. Holmes, despite answering every question with confidence, my marks are unexpectedly low. What unseen error might explain this discrepancy?",
    "Holmes, my internet connection works flawlessly by day but fails me each night when I study. What hidden cause could produce such a pattern?",
    "I distinctly recall saving my assignment, yet it has vanished from my computer. Where should I begin my investigation, Holmes?",
    "Though I understand the theory well, one problem continues to defeat me. What crucial detail might I be overlooking?",
    "Holmes, our group study sessions always devolve into distraction rather than productivity. What factor undermines our intent?",
    "I prepare thoroughly, yet in the examination hall my mind goes blank. What causes this sudden failure of memory, Holmes?",
    "Despite long hours of study late into the night, I achieve little progress. Where lies the flaw in my method?",
    "Holmes, I have been accused of plagiarism, yet my work is original. How might I prove my innocence?",
    "A peculiar bug appears in my code when I work alone, yet vanishes when I attempt to demonstrate it. How can this be explained?",
    "Holmes, I intend to study, yet I find myself endlessly procrastinating. What governs this contradiction?",
]

for q in questions:
    prompt = f"Human: {q}\nSherlock Holmes:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200,
            temperature=0.7, top_p=0.9,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = response.split("Human:")[0].strip()
    print(f"You: {q}")
    print(f"Sherlock: {response}")
    print("-"*60)
EOF
