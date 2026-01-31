# 🕵️‍♂️ Sherlock Holmes AI Detective

> "Data! Data! Data! I can't make bricks without clay." — *Sherlock Holmes*

This project fine-tunes the **Qwen 2.5 7B** Large Language Model (LLM) to adopt the persona of **Sherlock Holmes**. It uses **LoRA (Low-Rank Adaptation)** and **4-bit quantization** for efficient training and inference. The final model can be run locally or deployed using a serverless architecture with **Modal**.

## 🌟 Features

*   **Persona Fine-tuning**: Trained on a curated dataset of Sherlock Holmes dialogues and mysteries.
*   **Efficient Training**: Uses QLoRA (Quantized LoRA) to fine-tune a 7B parameter model on consumer hardware.
*   **Interactive UI**: A polished Streamlit chat interface styled with a "Dark Detective" theme.
*   **Dual Interface**: Includes `test.py` for direct local chat and `app.py` for a web interface (connects to your own backend).
*   **Evaluation Suite**: Includes scripts for perplexity analysis, content similarity checks, and training loss visualization.

## 📂 Project Structure

```
├── app.py                  # Streamlit frontend application
├── test.py                 # Local CLI Chatbot (Runs model locally)
├── modal_deploy.py         # Modal backend deployment script
├── fine_tune.py            # Training script (QLoRA)
├── base_model.py           # Baseline model testing
├── evaluation.py           # Model evaluation metrics
├── extract_loss.py         # Helper to extract training logs
├── visual.py               # Visualization generation
├── requirements.txt        # Dependencies for Streamlit App (Frontend)
├── requirements_training.txt # Dependencies for Training & Local Inference
├── evaluation_report.json  # Pre-computed evaluation metrics
├── training_loss.json      # Pre-computed training loss data
├── data/                   # Training datasets
└── sherlock-finetuned/     # Adapter weights (not in repo, generated after training)
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ShubhamHaraniya/Persona-Modeling.git
cd Persona-Modeling
```

### 2. Train the Model (Local/GPU)
Since the fine-tuned adapters are not included in the repository, you must train the model first.

1.  **Install Training Dependencies**:
    ```bash
    pip install -r requirements_training.txt
    ```
2.  **Train**:
    ```bash
    python fine_tune.py
    ```
    *This generates adapters in `sherlock-finetuned/`.*

### 3. Chat Locally (CLI)
Chat with the model directly on your machine using the provided script:

```bash
python test.py
```

### 4. Run the App (Web UI)
To use the web interface, you must first deploy the backend (see "Cloud Deployment") to get your own URL.

1.  Update `API_URL` in `app.py` with your new Modal Webhook URL.
2.  Run the UI:
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```

### 5. Evaluate
```bash
python evaluation.py
python visual.py
```

## ☁️ Cloud Deployment

### Backend (Modal)
The heavy lifting (LLM inference) is handled by Modal.

1.  Install Modal: `pip install modal`
2.  Setup Modal token: `modal setup`
3.  Deploy the model:
    ```bash
    modal deploy modal_deploy.py
    ```
4.  Copy the **Webhook URL** provided by Modal and update the `API_URL` in `app.py`.

### Frontend (Optional: Streamlit Cloud)
To share the UI with the world:

1.  Push this repo to GitHub.
2.  Go to [Streamlit Cloud](https://share.streamlit.io/).
3.  Deploy the app pointing to `app.py`.
4.  *Note: `requirements.txt` is optimized for the cloud frontend and does not include heavy torch dependencies.*

## 📊 Performance & Visualization

The project includes tools to visualize training progress:
*   **Training Loss**: Track convergence over epochs.
*   **Perplexity**: Compare base model vs. fine-tuned model on held-out data.
*   **Cosine Similarity**: Measure how closely the style matches the reference corpus.

*(Generated charts are saved in the `result/` directory)*

## 🛠️ Tech Stack

*   **Model**: Qwen/Qwen2.5-7B-Instruct
*   **Fine-tuning**: PEFT (LoRA), bitsandbytes (4-bit quantization), TRL (Transformer Reinforcement Learning)
*   **Infrastructure**: Modal (Serverless GPU), Streamlit (UI)
*   **Language**: Python 3.10+

## 📜 License

This project is open-source. Feel free to use the code for your own detective endeavors!
