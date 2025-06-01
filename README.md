<div align="center">
<h1>TinyLlama-Linux-QA</h1>
<h3></h3>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/tiam4tt/tinyllama-linux-qa?style=for-the-badge&labelColor=11111b&color=b4befe)
![](https://img.shields.io/github/repo-size/tiam4tt/TinyLlama-Linux-QA?style=for-the-badge&labelColor=11111b&color=94e2d5)
![](https://img.shields.io/badge/model_type-seq2seq-e5c890?style=for-the-badge&labelColor=11111b)
![](https://img.shields.io/badge/task-QnA-eba0ac?style=for-the-badge&labelColor=11111b)
</div>

## Overview
TinyLlama-Linux-QA is a specialized question-answering model designed to handle Linux-related queries. It is based on the TinyLlama architecture and has been fine-tuned on a dataset of Linux questions and answers.

![preview](./assets/preview.png)

## Dataset
The model has been trained on a dataset that includes a wide range of Linux-related questions, covering topics such as system administration, command-line usage, shell scripting, and more. The dataset is designed to provide comprehensive coverage of common Linux tasks and issues.

The data is accessible within this repository, or on [kaggle](https://www.kaggle.com/datasets/tiamatt/reddit-curated-linux-qna).

## Training

Training was performed using [Unsloth](https://unsloth.ai/) for memory-efficient and faster fine-tuning. Training machine specifications:
| **Attribute** | **Details** |
| ---------- | -------------------------- |
| **Base Model** | [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| **GPU** | NVIDIA RTX 3050Ti 4GiB |
| **Memory** | 16GiB |
| **Nvidia Driver version** | 570.153.02 |
| **CUDA version** | 12.8 |
| **Python version** | 3.12.4 |
| **Kernel** | 6.14.7-arch2-1 |

The model has been uploaded to Hugging Face and is available for use under the model ID [tiam4tt/TinyLlama-1.1B-chat.v1.0-linux-qna](https://huggingface.co/tiam4tt/TinyLlama-1.1B-chat.v1.0-linux-qna)

## Usage

To use the TinyLlama-Linux-QA model, you can load it using the Hugging Face Transformers library, or with Unsloth for lighter resource consumption. Here is an example using Unsloth:

```python
from unsloth import FastLanguageModel
model_name = "tiam4tt/TinyLlama-1.1B-chat.v1.0-linux-qna"
model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=256,
            dtype=None, # Use None for automatic dtype detection
            load_in_4bit=True,
        )
        # Enable the model for inference
        FastLanguageModel.for_inference(model)
```
You can then use the model to generate answers to Linux-related questions by providing a prompt and using the `generate` method.

```python
PROMPT = """Below is a question relating to the Linux operating system, paired with a paragraph describing further context. Write a short, simple, concise, and comprehensive response to the question.
### Question
{}
### Context
{}
### Response
{}"""

question = "How do I check the disk usage of a directory in Linux?"

inputs = tokenizer(PROMPT.format(
    question,
    "",
    "" # Leave blank for generation
    ),
    return_tensors="pt").to(model.device)
output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            top_p=0.9
        )
answer = tokenizer.decode(output[0], skip_special_tokens=True)
print(answer)
```

## Run with Streamlit

To run the model with Streamlit, change directory to `app`

```bash
cd app
```
Then, install the required dependencies:
```bash
pip install -r requirements.txt
```
Finally, run the Streamlit app:
```bash
streamlit run app.py
```


## Note
*As of 01/06/2025, **Unsloth** has compatiblity issues with GPUs whose architecture is before Ampere (e.g., T4, V100, etc.). This is because the current version of `triton` 3.3.0 raises errors when computing fp16 on bf16-incompatible devices, if you encounter such case, downgrading `triton` to version **3.2.0** might be able to address the problem.*