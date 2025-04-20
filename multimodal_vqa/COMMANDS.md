# Command Reference

This document provides a comprehensive list of commands to work with the Visual Question Answering project.

## Setup Commands

### Environment Setup

```powershell
# Create a virtual environment
python -m venv multivqa

# Activate the virtual environment (Linux/macOS)
source multivqa/bin/activate

# Activate the virtual environment (Windows)
C:\Users\amuaa\multivqa\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Data Processing Commands

### Processing Raw DAQUAR Dataset

```powershell
# Process the raw DAQUAR dataset with default parameters
python src/process_data.py

# Process with custom parameters
python src/process_data.py --dataset_folder data --raw_qa_file raw/qa_pairs.txt --train_file data_train.csv --eval_file data_eval.csv --test_size 0.2
```

## Training Commands

### Basic Training

```powershell
# Train with default parameters
python scripts/train.py

# Train with custom parameters
python scripts/train.py `
  --train_csv dataset/data_train.csv `
  --test_csv dataset/data_eval.csv `
  --answer_space dataset/answer_space.txt `
  --images_dir dataset/images `
  --output_dir checkpoints `
  --text_model bert-base-uncased `
  --image_model google/vit-base-patch16-224-in21k `
  --batch_size 32 `
  --num_epochs 5 `
  --fp16

```

### Advanced Training Options

```powershell
# Train with different model architecture
python scripts/train.py \
  --text_model roberta-base \
  --image_model facebook/deit-base-patch16-224 \
  --model_name roberta_deit \
  --intermediate_dim 768

# Resume training from checkpoint
python scripts/train.py \
  --output_dir checkpoints/continued_training
```

## Testing Commands

### Basic Testing

```powershell


# Test with custom parameters
python scripts/test.py `
  --test_csv dataset/data_eval.csv `
  --answer_space dataset/answer_space.txt `
  --images_dir dataset/images `
  --checkpoint checkpoints/bert_vit/checkpoint-1560/model.safetensors `
  --num_examples 10

```

### Using Your Trained Model

```powershell
# Test with your specific checkpoint
python scripts/test.py --checkpoint /path/to/your/checkpoint-1560/model.safetensors
```

## Streamlit App Commands

### Running the Demo App

```powershell
# Navigate to app directory
cd app

# Run with specific checkpoint
CHECKPOINT_PATH=../checkpoints/bert_vit/checkpoint-1560/model.safetensors streamlit run app.py


```
```powershell
cd app
$env:CHECKPOINT_PATH = "../checkpoints/bert_vit/checkpoint-1560/model.safetensors"
streamlit run app.py
```


## Miscellaneous Commands

### Inference on a Single Image

```powershell
# Run inference on a single image (custom script)
python scripts/inference.py `
  --image_path path/to/image.jpg `
  --question "What color is the wall?" `
  --checkpoint checkpoints/bert_vit/checkpoint-1560/model.safetensors
```
python scripts/inference.py `
  --image_path path/to/image.jpg `
  --question "What color is the wall?" `
  --checkpoint checkpoints/bert_vit/checkpoint-1560/model.safetensors

### Model Performance Evaluation

```powershell
# Evaluate model performance in detail
python scripts/evaluate.py \
  --test_csv data/data_eval.csv \
  --answer_space data/answer_space.txt \
  --images_dir data/images \
  --checkpoint checkpoints/bert_vit/checkpoint-1560/model.safetensors \
  --output_file evaluation_results.json
```



