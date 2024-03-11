
# Gemma 2b Instruction Fine-Tuned Model

## Introduction

This repository contains the code and model for fine-tuning the Gemma 2b Instruction model using a dataset of 20k medium articles. The fine-tuned model is designed to generate responses based on input prompts related to instructional queries in English.

### Model Details:

- **Model Name:** Gemma 2b Instruction Fine-Tuned Model
- **Framework:** Keras with JAX backend
- **Model Type:** Causal Language Model
- **Pre-trained Model:** Gemma 2b Instruction Model

## Fine-Tuning Process

The Gemma 2b Instruction Fine-Tuned Model was fine-tuned using the following process:

1. **Data Preparation:** 20k medium articles were used as the training dataset.
2. **Model Configuration:** Gemma 2b Instruction model was used as the base model.
3. **Hyperparameters Tuning:** AdamW optimizer was used with customized learning rates and weight decays.
5. **Training:** The model was trained for 2 epochs with a batch size of 2 Due to lack of GPU Clusters to handle this 2B model even with LORA config.

## Usage

### Environment Setup

Ensure you have the necessary dependencies installed:

```bash
pip install keras keras-nlp
```

### Inference

To use the fine-tuned model for inference, follow these steps:

1. **Download the Model:**
   - The fine-tuned model file `version_finetuned.keras` can be downloaded from the [Hugging Face Model Hub](https://huggingface.co/likith123/Gemma_2B_Medium_20k_FT).

2. **Inference:**

```python
from keras.models import load_model

# Load the saved model
loaded_model = load_model("version_finetuned.keras")
instruction = "How to code in python, Give me An example code"
response = ""

# Create the input prompt
prompt = f"Instruction:\n{instruction}\n\nResponse:\n{response}"
# Generate inference using the loaded model
inference_result = loaded_model.generate(prompt, max_length=1024)

# Print or use the generated response
print(inference_result)
```

## Model Card

For more information about the model, check out its model card on [Hugging Face Model Hub](https://huggingface.co/likith123/Gemma_2B_Medium_20k_FT).

## License

This project is licensed under the [MIT License](LICENSE).

