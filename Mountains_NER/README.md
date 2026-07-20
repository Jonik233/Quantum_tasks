# Task 1: Natural Language Processing - Mountain NER

## Overview
This repository contains a complete pipeline for a Named Entity Recognition (NER) model designed to identify mountain names within natural language text. 

The solution covers the entire Machine Learning lifecycle:
1. **Data Engineering:** Hybrid dataset curation combining open-source data with LLM-generated synthetic data.
2. **Model Training:** Custom-headed DistilBERT model fine-tuned on AWS SageMaker with MLflow metric tracking.
3. **Deployment:** High-performance, Dockerized FastAPI web service designed for low-latency inference.

## Project Structure

| File/Folder                       | Description                                                                           |
|:----------------------------------|:--------------------------------------------------------------------------------------|
| `eda/kaggle_dataset_eda.ipynb`    | EDA notebook for kaggle mountains ner dataset.                                        |
| `eda/synthetic_dataset_eda.ipynb` | EDA notebook for synthetic data that was generated using OpenAI API.                  |
| `eda/main_eda.ipynb`              | Notebook that walks through the creation and analysis of a final dataset.             |
| `data_generation/generate.py`     | Script utilizing the OpenAI API to generate and self-correct synthetic training data. |
| `src/model.py`                    | Contains the custom `DistilBERTClass` architecture.                                   |
| `src/train.py`                    | The PyTorch training loop with MLflow metric tracking.                                |
| `src/main.py`                     | The AWS SageMaker launcher script utilizing the HuggingFace Estimator.                |
| `src/app.py`                      | The FastAPI application logic utilizing lifespan event handlers.                      |
| `src/inference.py`                | Inference module for loading the fine-tuned model and extracting entities.            |
| `demo.ipynb`                      | Jupyter notebook demonstrating the model's inference capabilities via HTTP requests.  |
| `pyproject.toml` & `uv.lock`      | Main project dependencies managed via Astral's `uv`.                                  |
| `src/requirements.txt`            | Minimal runtime dependencies strictly scoped for the SageMaker training container.    |
| `Dockerfile` & `Makefile`         | Containerization instructions for the FastAPI web service.                            |

## Dependency Management
This project utilizes a dual-dependency strategy to ensure lightning-fast, bloat-free environments:
* **Local & Inference Environment:** Managed entirely by **`uv`**. The `pyproject.toml` strictly isolates heavy training libraries (like `scikit-learn` and `sagemaker`) into development groups, ensuring the final Docker inference container remains incredibly lightweight.
* **AWS SageMaker Environment:** A scoped `src/requirements.txt` is injected directly into the AWS Deep Learning Container, installing only the runtime metrics libraries (`evaluate`, `seqeval`) needed during the fine-tuning job.

## Dataset Curation
NER models require highly balanced data to correctly learn entity boundaries and the 'Outside' (O) background context. A hybrid data engineering approach was utilized:
* **Baseline Data:** An open-source Kaggle NER Mountain dataset: https://www.kaggle.com/code/geraygench/mountain-ner-eda-basseline-model?select=mountain_dataset_with_markup.csv
* **Synthetic Generation:** Since EDA revealed a severe class imbalance (most Kaggle samples contained zero mountains). `data_generation/generate.py` generated 1,000 diverse synthetic records using the OpenAI API to balance the final dataset.
* **Self-Healing Alignment:** The generation pipeline features a programmatic self-correction algorithm to guarantee that LLM-generated character spans perfectly align with the text, solving the inherent token-counting limitations of LLMs.

## Model Architecture & Training
The core model utilizes **DistilBERT** (`distilbert-base-cased`), retaining 95% of BERT's performance while being 40% smaller and 60% faster, making it ideal for scalable web APIs.

* **Custom Head:** A custom classification head (Dropout $\rightarrow$ Linear $\rightarrow$ GELU $\rightarrow$ Linear) was built on top of the contextual embeddings to project representations into the 3 target BIO classes (O, B-MOUNTAIN, I-MOUNTAIN).
* **Training Pipeline:** Fine-tuned using **AWS SageMaker** on an `ml.g6.2xlarge` instance. 
* **Metrics Tracking:** MLflow was integrated to stream step-wise loss and compute strict, chunk-level `seqeval` metrics at the end of every epoch.

### Training Results (Epoch 2)
| Metric | Score |
| :--- | :--- |
| **Validation Loss** | 0.0178 |
| **Overall F1** | 0.9356 |
| **Overall Precision** | 0.9148 |
| **Overall Recall** | 0.9574 |

You can check complete training logs in `notes/training_metrics.log`

### Training Loss over training steps
![Traiing loss](plots/train_loss_step.png)

*(See `/reports/improvements.pdf` for further analysis and potential next steps).*

## Deployment Architecture
The model is served via a **FastAPI** web service inside a **Docker** container. 
* **Lifespan Management:** The heavy PyTorch model and tokenizers are loaded into memory exactly once during the application's startup phase (`@asynccontextmanager`) and stored in the application state, ensuring sub-100ms response times for incoming requests.
* **Synchronous PyTorch Execution:** The prediction endpoint utilizes a standard `def` (rather than `async def`) to allow FastAPI to offload the CPU-blocking PyTorch tensor mathematics to an external threadpool, preventing event-loop freezing.

## Setup and Demo

### 1. Model Weights
Before building the container, download the fine-tuned model artifacts (`best_model.bin`, `config.json`, etc.) and place them inside a `model_artifacts/` directory in the project root.
The trained model artifacts can be downloaded here:
[Download model.tar.gz](https://qunatum-tasks-bucket.s3.eu-central-1.amazonaws.com/Mountains_NER/output/huggingface-pytorch-training-2026-07-20-12-29-44-225/output/model.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXSBCNESKUPROYKNH%2F20260720%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260720T214102Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9e02406a4b12db25da76f41bc94d8df9f8a1fd1472ab418f41b6917bb1bd9ba3)

### 2. Build and Run the Service
Ensure Docker is installed and running, then execute the provided Makefile command to build the image and launch the container on port 8080:

```bash
make build-container
```

### 3. Run Inference
Once the container is up and running, you can interact with the API using standard HTTP POST requests. You can view the full demonstration in demo.ipynb or run the following Python snippet:
```
import time
import requests

text = "The Andes run through several South American countries."
start = time.time()

response = requests.post("[http://0.0.0.0:8080/predict](http://0.0.0.0:8080/predict)", json={"query": text}).json()

print(f"Text: {response['original_text']}")
print("\nTokens:")
for token in response["predictions"]:
    print(f"Token {token['token']}: (label: {token['label']}, confidence: {token['confidence']})")

elapsed_time = time.time() - start
print(f"\nInference time: {elapsed_time}")
```
**Expected Output:**
```
Text: The Andes run through several South American countries.

Tokens:
Token The: (label: O, confidence: 0.9987)
Token Andes: (label: B-MOUNTAIN, confidence: 0.9667)
Token run: (label: O, confidence: 0.9993)
Token through: (label: O, confidence: 0.9993)
Token several: (label: O, confidence: 0.9993)
Token South: (label: O, confidence: 0.999)
Token American: (label: O, confidence: 0.9989)
Token countries: (label: O, confidence: 0.9991)
Token .: (label: O, confidence: 0.9993)

Inference time: 0.13576912879943848
```

### 4. Teardown
After you are done testing the endpoint, cleanly stop and remove the running container by executing:
```
make stop-container
```