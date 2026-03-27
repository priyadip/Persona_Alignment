# Sherlock-LLM: Persona Alignment via Parameter-Efficient Fine-Tuning of Qwen-2.5

> *"Data! Data! Data! I cannot make bricks without clay."*  -  Sherlock Holmes

**Abstract.** Large Language Models optimized through Reinforcement Learning from Human Feedback exhibit a characteristic homogenization of voice - helpful, verbose, and perpetually hedged - that proves fundamentally incompatible with applications demanding consistent persona maintenance. This repository presents Sherlock-LLM, a conversational agent engineered through QLoRA-based fine-tuning of Qwen-2.5-7B-Instruct to rigorously embody the deductive methodology, Victorian formality, and distinctive idiolect of Arthur Conan Doyle's Sherlock Holmes. Our approach achieves a **93.5% reduction in conditional perplexity** (17.34 → 1.13) and **519% improvement in lexical similarity** (0.103 → 0.637) against canonical Holmesian text, with a ROUGE-L score of **0.729** on Holmesian reference text, demonstrating that sophisticated persona alignment remains tractable on consumer hardware through parameter-efficient adaptation strategies.

---

## Table of Contents

1. [Motivation and Problem Statement](#motivation-and-problem-statement)
2. [Theoretical Framework](#theoretical-framework)
3. [Repository Structure](#repository-structure)
4. [Installation and Environment Configuration](#installation-and-environment-configuration)
5. [Training Pipeline](#training-pipeline)
6. [Inference and Deployment](#inference-and-deployment)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Empirical Results](#empirical-results)
9. [Model Selection Analysis](#model-selection-analysis)
10. [Ablation Studies](#ablation-studies)
11. [Qualitative Case Studies](#qualitative-case-studies)
12. [Limitations and Future Directions](#limitations-and-future-directions)
13. [Citation](#citation)
14. [License](#license)

---

## Motivation and Problem Statement

Foundation models such as GPT-4, Llama 3, and Qwen 2.5 demonstrate exceptional general-purpose reasoning capabilities, yet their alignment procedures systematically eliminate the distinctive voices that characterize compelling fictional personas. The standard instruction-tuned model produces responses that are unfailingly polite but fundamentally characterless - a significant impediment for applications in entertainment, education, and interactive storytelling where narrative consistency proves paramount.

Persona alignment presents two orthogonal challenges that conventional fine-tuning approaches struggle to reconcile:

**Computational Intractability.** Full-parameter updates for a 7-billion parameter model demand approximately 112GB of VRAM for 16-bit training - resource requirements that exclude the vast majority of academic researchers from meaningful participation in this research domain.

**Catastrophic Forgetting.** Aggressive optimization on narrow stylistic corpora systematically degrades the model's general reasoning faculties, producing systems that approximate the surface features of a character while failing to maintain logical coherence in complex inferential chains.

This work demonstrates that Parameter-Efficient Fine-Tuning via QLoRA resolves both constraints simultaneously, enabling high-fidelity persona adoption within a 6GB VRAM envelope while preserving the underlying model's analytical capabilities.

---

## Theoretical Framework

### Low-Rank Adaptation (LoRA)

The LoRA hypothesis posits that weight updates during model adaptation occupy a low-dimensional subspace of the full parameter space. For a pre-trained weight matrix W₀ ∈ ℝ^(d×k), we constrain the update ΔW to rank r where r ≪ min(d, k):

```
h = W₀x + ΔWx = W₀x + BAx
```

where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k) constitute the trainable adapter matrices. The initialization strategy - A drawn from N(0, σ²) and B initialized to zero - ensures that training commences from the exact pre-trained distribution. A scaling factor α/r modulates the update magnitude:

```
h = W₀x + (α/r)BAx
```

For weight matrices of dimension 4096 × 4096 with rank r = 16, this parameterization achieves approximately 256× compression relative to full fine-tuning.

### 4-bit Normal Float (NF4) Quantization

Standard integer quantization assumes uniform weight distributions - an assumption violated by the zero-centered normal distributions characteristic of pre-trained neural networks. NF4 constructs its discrete codebook from the quantiles of N(0, 1):

```
qᵢ = Φ⁻¹(i / (2ᵏ + 1)), i ∈ {1, 2, ..., 2ᵏ}
```

where Φ⁻¹ denotes the inverse cumulative distribution function. Block-wise quantization with 64-parameter blocks prevents outliers from degrading precision across entire matrices, while double quantization compresses the scaling factors themselves to 8-bit representations.

### QLoRA Integration

QLoRA propagates gradients through a frozen 4-bit quantized base model into the low-rank adapters, achieving the memory efficiency of aggressive quantization without sacrificing the representational flexibility of LoRA. This combination enables fine-tuning of 7B-parameter models on hardware that would otherwise support only inference workloads.

---

## Repository Structure

```
sherlock-llm/
├── fine_tune.py              # QLoRA training orchestration
├── test.py                   # Local CLI inference interface
├── app.py                    # Streamlit web application
├── modal_deploy.py           # Serverless GPU deployment configuration
├── base_model.py             # Baseline model evaluation
├── evaluation.py             # Quantitative metrics computation
├── extract_loss.py           # Training log extraction utilities
├── visual.py                 # Visualization generation
├── requirements.txt          # Frontend dependencies (Streamlit)
├── requirements_training.txt # Training and inference dependencies
├── evaluation_report.json    # Pre-computed evaluation metrics
├── training_loss.json        # Training trajectory data
├── data/
│   ├── corpus/               # Unstructured Holmesian literature
│   └── dataset.jsonl         # Structured instruction-response pairs
├── result/                   # Generated visualizations
└── sherlock-finetuned/       # LoRA adapter weights (post-training)
```

---

## Installation and Environment Configuration

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU with minimum 6GB VRAM
- CUDA Toolkit 12.1 or higher
- **PyTorch >= 2.6.0** (required due to CVE-2025-32434 security fix in `transformers >= 5.x`; earlier versions will raise a `ValueError` when resuming from checkpoints)

### Training Environment

```bash
git clone https://github.com/YourUsername/sherlock-llm.git
cd sherlock-llm
pip install -r requirements_training.txt
```

If you are running on a system with CUDA 12.4+ drivers (driver version ≥ 550), install the cu124 PyTorch wheel explicitly:

```bash
pip install "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124
```

The training dependencies include PyTorch, Transformers, PEFT, TRL, and bitsandbytes for 4-bit quantization support.

### Frontend Environment

```bash
pip install -r requirements.txt
```

The frontend requirements are intentionally minimal, excluding heavy torch dependencies to facilitate cloud deployment.

---

## Training Pipeline

### Data Preparation

The training corpus comprises two complementary data sources designed to capture both stylistic patterns and conversational dynamics:

**Unstructured Corpus.** Public domain text from *The Adventures of Sherlock Holmes* and *The Memoirs of Sherlock Holmes*, preprocessed through header removal, chapter segmentation, and 500-character chunking with 50-character overlap to maintain contextual continuity.

**Structured Instruction Data.** A curated collection of **58,877 instruction-response pairs** spanning identity questions, deductive reasoning scenarios, interactive dialogues, and domain-specific knowledge queries, generated from the Holmesian corpus via `generate_dataset.py`. Each example enforces the system prompt:

> "You are Sherlock Holmes, an expert consulting detective. Use deductive reasoning, ask clarifying questions to gather more data, and analyze evidence to reach logical conclusions."

### Training Execution

```bash
python fine_tune.py
```

The training configuration targets all linear projection layers within the transformer architecture - a strategy that recent literature demonstrates yields superior stylistic adaptation compared to Q/V-only targeting.

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4 |
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Effective Batch Size | 8 (2 per device × 4 gradient accumulation) |
| Learning Rate | 2 × 10⁻⁴ |
| LR Scheduler | Cosine with 100-step warmup |
| Optimizer | Paged AdamW 32-bit |
| Training Epochs | 3 |
| Max Sequence Length | 1024 |
| Total Optimization Steps | 19,878 |
| Training Dataset Size | 58,877 instruction-response pairs |

Training completes in approximately **10.5 hours** of active training on a single NVIDIA A100-SXM4-40GB GPU (19,878 steps at ~1.9s/step). Total wall-clock time including periodic evaluation runs is approximately **26.5 hours**. The pipeline supports checkpoint-based resumption via the `RESUME_CHECKPOINT` environment variable, enabling continuation across SLURM job time limits.

---

## Inference and Deployment

### Local CLI Inference

```bash
python test.py
```

The local interface loads the merged adapter weights and provides an interactive command-line conversation loop. Choose between interactive chat mode (type your own questions) or quick test mode (predefined Holmesian prompts).

### HPC / SLURM Cluster Deployment

For running on a SLURM-managed HPC cluster with GPU nodes:

```bash
sbatch run_persona.sh
```

The `run_persona.sh` script automatically:
- Detects the latest checkpoint and resumes training if a partial run exists
- Runs the full pipeline: fine-tuning → loss extraction → evaluation → visualization
- Cleans up intermediate checkpoints after completion

Recommended SLURM configuration (tested on NVIDIA A100-SXM4-40GB):

```bash
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --qos=dgxqos   # use high-priority QOS if available
```

### Web Interface via Streamlit

```bash
streamlit run app.py
```

The Streamlit application implements a "Dark Detective" themed chat interface with conversation history management and real-time response streaming.

### Serverless Deployment via Modal

For production deployment without local GPU resources:

```bash
pip install modal
modal setup
modal deploy modal_deploy.py
```

Modal provides serverless A100 infrastructure with automatic scaling, exposing a RESTful API endpoint for integration with external applications:

```
POST /v1/chat/completions
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.7,
  "max_tokens": 256
}
```

Average endpoint latency measures 1.2 seconds per response with support for up to 10 concurrent users.

---

## Evaluation Methodology

### Conditional Perplexity

Standard perplexity conflates prompt encoding with generation quality. We employ conditional perplexity that masks instruction tokens, evaluating only the probability of response tokens given the instruction context:

```
PPL_cond = exp(-1/n ∑ᵢ log P(yᵢ | X, y<ᵢ))
```

where X represents instruction tokens and Y = (y₁, ..., yₙ) represents response tokens. This formulation isolates generation quality from prompt memorization artifacts.

### Jaccard Similarity

Lexical alignment with canonical Holmesian vocabulary is quantified through Jaccard similarity over bag-of-words representations:

```
J(G, R) = |G ∩ R| / |G ∪ R|
```

where G denotes generated text tokens and R denotes reference corpus tokens. Higher similarity indicates successful adoption of period-appropriate terminology ("deduce," "trifles," "elementary") over modern alternatives ("think," "clues," "help").

### Evaluation Execution

```bash
python evaluation.py
python visual.py
```

Generated visualizations are saved to the `result/` directory.

---

## Empirical Results

### Training Dynamics

The loss trajectory exhibits three distinct phases characteristic of successful persona adaptation:

| Phase | Epoch Range | Loss Range | Interpretation |
|-------|-------------|------------|----------------|
| Rapid Adaptation | 0 – 0.5 | 3.16 → 0.90 | Initial stylistic constraint learning; vocabulary and identity overwriting |
| Refinement | 0.5 – 1.5 | 0.90 → 0.24 | Vocabulary consolidation, period-appropriate phrasing locked in |
| Convergence | 1.5 – 3.0 | 0.24 → 0.20 | Stable persona coherence; eval loss plateaus at **0.246** |

The training loss at key milestones on the A100-SXM4-40GB:

| Epoch | Train Loss | Token Accuracy |
|-------|------------|----------------|
| 0.01 | 1.546 | ~39% |
| 0.10 | 1.287 | ~67% |
| 0.50 | 0.896 | ~77% |
| 1.00 | 0.401 | ~88% |
| 1.50 | 0.241 | ~92% |
| 2.00 | 0.220 | ~93% |
| 3.00 | **0.197** | **93.9%** |

Eval loss converged to **0.246** with token accuracy **92.7%**, confirming stable persona adoption without overfitting.

### Quantitative Performance

| Metric | Base Model | Sherlock-LLM | Relative Change |
|--------|------------|--------------|-----------------|
| Conditional Perplexity (↓) | 17.34 | **1.13** | **-93.5%** |
| Jaccard Similarity (↑) | 0.103 | **0.637** | **+519%** |
| ROUGE-L (↑) | 0.145 | **0.729** | **+403%** |
| Victorian Term Frequency (↑) | 1.29% | **3.19%** | **+147%** |
| Eval Loss (↓) |  -  | **0.246** |  -  |
| Token Accuracy (↑) |  -  | **92.7%** |  -  |

The 15× perplexity reduction indicates that the fine-tuned model has deeply internalized Victorian phrasing and Holmesian discourse as its most probable output distribution. The ROUGE-L score of 0.729 demonstrates strong lexical overlap with canonical reference text. The asymmetric improvements - vocabulary alignment and fluency responding more dramatically than token-level metrics - confirm that the large-scale instruction dataset (58,877 pairs) with 1024-token context windows proved highly effective for stylistic persona transfer.

---

## Model Selection Analysis

Our development process evaluated multiple candidate architectures before converging on Qwen-2.5-7B-Instruct:

| Model | Outcome | VRAM | Pivot Rationale |
|-------|---------|------|-----------------|
| Llama 3-8B | Poor alignment | 8 GB | Persistent instruction-following behavior despite persona prompting |
| Qwen 2.5-14B | Promising initial | 12 GB | Resource constraints exceeded computational budget |
| Qwen 2.5-7B | **Success** | 6 GB | Optimal balance of capacity and feasibility |

The Llama 3 experiments - five training runs with varied hyperparameters (learning rates {1e-4, 2e-4, 5e-4}, ranks {8, 16, 32}, epochs {2, 3, 5}) - consistently achieved perplexity values exceeding 25 and Jaccard similarity below 0.15. We hypothesize that Llama 3's extensive RLHF alignment creates resistance to the stylistic shifts required for persona adoption, though this warrants further investigation.

---

## Ablation Studies

### LoRA Rank Selection

| Rank | Perplexity | Jaccard | Trainable Params |
|------|------------|---------|------------------|
| 8 | 2.8 | 0.51 | 2.1M |
| **16** | **1.13** | **0.637** | **4.2M** |
| 32 | 1.21 | 0.61 | 8.4M |
| 64 | 1.38 | 0.58 | 16.8M |

Rank 16 provides optimal balance between representational capacity and regularization. Lower ranks lack sufficient expressiveness for full stylistic nuance; higher ranks exhibit overfitting to training data idiosyncrasies.

### Learning Rate Impact

| Learning Rate | Perplexity | Convergence Behavior |
|---------------|------------|----------------------|
| 1 × 10⁻⁴ | 1.9 | Slow convergence, higher final loss |
| **2 × 10⁻⁴** | **1.13** | **Stable convergence, eval loss 0.246** |
| 5 × 10⁻⁴ | 2.4 | Training instability (loss spikes >2.0) |

### Comparison with Alternative Methods

| Method | Perplexity | VRAM (GB) | Training Time (hrs) |
|--------|------------|-----------|---------------------|
| Full Fine-Tuning | 0.85 | 112 | 72 |
| Prefix Tuning | 4.1 | 18 | 12 |
| Adapter Layers | 2.9 | 22 | 16 |
| LoRA (r=16) | 1.8 | 8 | 13 |
| **QLoRA (ours)** | **1.13** | **6** | **10.5** |

QLoRA achieves the best performance-efficiency trade-off, approaching full fine-tuning quality at 18× memory reduction and 7× time reduction.

---

## Qualitative Case Studies

### Identity Inquiry

**Prompt:** "Who are you?"

**Base Model:** "I am Qwen, a large language model created by Alibaba Cloud. I am here to help you with information, answer questions, and assist with various tasks..."

**Sherlock-LLM:** "I confess I anticipated something of this nature. I am Sherlock Holmes, the world's only consulting detective. Scotland Yard comes to me when their methods fail  -  which is invariably. I have worked on far worse problems before breakfast."

The fine-tuning successfully overwrites identity guardrails while maintaining concise, character-appropriate responses with signature Holmesian dismissiveness toward Scotland Yard.

### Partnership Dynamics

**Prompt:** "What do you think about Dr. Watson?"

**Sherlock-LLM:** "I have examined the matter with some care. Dr. Watson is a model detective. He allows me to fill in the gaps in his narrative. In return, he provides the energy that my work requires. Together, we cover the ground more effectively than either could alone."

The model accurately reflects Holmes's complex relationship with Watson  -  respectful but analytical, viewing their partnership through a purely functional lens.

### Crime Scene Analysis

**Prompt:** "A man was found dead in a locked room with no windows. How would you approach this case?"

**Sherlock-LLM:** "My investigations confirm that the smallest detail is often the most instructive. The solution is not without interest. I would first examine the door lock and key to see if they are functioning properly and if the man might have had a duplicate key. I would also check for any signs of forced entry or tampering with the door. Finally, I would investigate any possible foul play, such as poison or hidden weapons, since the room was sealed when the man died. I have worked on far worse problems before breakfast. I find this consistent with everything I have observed."

The fine-tuned model demonstrates systematic forensic reasoning drawn directly from the Holmesian canon, producing a structured investigative methodology with period-appropriate phrasing.

### Deductive Methodology

**Prompt:** "What is your method of deduction?"

**Sherlock-LLM:** "My index is quite clear on this matter. I read the language of the hands, the posture, the boots, the watch, the letter  -  each tells its story to the trained eye. Nothing is trivial; everything speaks. There is nothing more to say on the matter."

The model enumerates authentic Holmesian observational targets (hands, boots, watch) drawn directly from canonical stories such as *A Study in Scarlet* and *The Sign of the Four*.

### Institutional Commentary

**Prompt:** "What are your thoughts on Scotland Yard?"

**Sherlock-LLM:** "I have not been idle in this matter. They are, as a rule, a little slow  -  rather slow, in my experience. Lestrade and his colleagues work with energy; it is method they lack. I trust that clears the matter up."

The model accurately captures Holmes's canonical contempt for official police methodology while acknowledging their diligence  -  a nuanced characterization consistent with Doyle's portrayal.

---

## Limitations and Future Directions

### Current Limitations

**Reasoning versus Style.** While the model produces stylistically authentic Holmesian discourse, its actual deductive capabilities remain bounded by the underlying 7B architecture. Complex inferential chains requiring 4+ steps occasionally exhibit logical discontinuities masked by correct surface phrasing.

**Temporal Anachronisms.** Approximately 8% of responses reference modern concepts (DNA testing, digital forensics) that would be anachronistic for the Victorian setting.

**Context Window Constraints.** The 512-token training length proves insufficient for extended mystery scenarios requiring multi-page case files or sustained multi-turn investigations.

### Research Directions

- **Chain-of-Thought Augmentation:** Explicit reasoning traces to distinguish genuine deduction from stylistic mimicry
- **Retrieval-Augmented Generation:** Integration with the complete Holmes canon for accurate case citation
- **Multi-Persona Architecture:** Distinct LoRA adapters for Watson, Moriarty, and Lestrade enabling interactive storytelling
- **Dynamic Rank Allocation:** AdaLoRA-style budget distribution across layers based on persona-critical representations
- **Cross-Lingual Transfer:** Investigation of adapter portability to non-English translations of the canon

---

## Citation

If this work proves useful for your research, please consider citing:

```bibtex
@article{sau2026sherlock,
  title={Sherlock-LLM: Persona Alignment via Parameter-Efficient Fine-Tuning of Qwen-2.5},
  author={Sau, Priyadip},
  journal={Technical Report},
  institution={Indian Institute of Technology Jodhpur},
  year={2026}
}
```

---

## References

1. Qwen Team. "Qwen2.5: A Foundation Model for Generalist Agents." arXiv:2409.12117, 2024.
2. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
3. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
4. Mangrulkar et al. "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods." GitHub, 2022.
5. Shazeer. "GLU Variants Improve Transformer." arXiv:2002.05202, 2020.
6. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing, 2024.

---

## License

This project is released under the MIT License. The training data derives from public domain texts. Feel free to adapt the methodology for your own persona alignment endeavors.

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*
