# Medical VQA: Visual Question Answering for Medical Images

This repository contains code for training and testing a Visual Question Answering (VQA) model on medical images. The model uses CLIP for image encoding and GPT-2 for language modeling, connecting them with a mapping network.

## Project Structure

```
medical-vqa/
├── model/
│   ├── __init__.py
│   ├── dataset.py      # Dataset class
│   ├── model.py        # Model architecture
│   └── utils.py        # Utility functions
├── train.py            # Training script
├── test.py             # Testing script
├── app.py              # Streamlit application
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/medical-vqa.git
cd medical-vqa
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
& "C:\Users\amuaa\vqa_env\Scripts\Activate.ps1" 
pip install -r requirements.txt
```

3. Download the required NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Data Preparation

The model expects the following data structure:
- A directory containing medical images
- CSV files for training, validation, and testing with the following columns:
  - Image filename
  - Question
  - Answer

The CSV files should be named:
- `vqa_rad_train.csv` for training
- `vqa_rad_valid.csv` for validation
- `vqa_rad_test.csv` for testing

## Training

To train the model:

```bash
python train.py --train_csv path/to/vqa_rad_train.csv \
                --val_csv path/to/vqa_rad_valid.csv \
                --test_csv path/to/vqa_rad_test.csv \
                --img_dir path/to/images \
                --batch_size 16 \
                --learning_rate 5e-3 \
                --epochs 10 \
                --device cuda \
                --save_dir checkpoints \
                
```
```
python train.py --train_csv vqa_rad_train.csv `
                --val_csv vqa_rad_valid.csv `
                --test_csv vqa_rad_test.csv `
                --img_dir img `
                --batch_size 16 `
                --learning_rate 5e-3 `
                --epochs 10 `
                --device cuda `
                --save_dir checkpoints `
                
```

### Training Options

- `--train_csv`: Path to the training CSV file (default: "vqa_rad_train.csv")
- `--val_csv`: Path to the validation CSV file (default: "vqa_rad_valid.csv")
- `--test_csv`: Path to the test CSV file (default: "vqa_rad_test.csv")
- `--img_dir`: Directory containing images (default: "img")
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 5e-3)
- `--epochs`: Number of epochs to train for (default: 10)
- `--device`: Device to use for training (cuda/cpu) (default: "cuda")
- `--save_dir`: Directory to save checkpoints (default: "checkpoints")
- `--save_every`: Save checkpoint every N epochs (default: 1)
- `--validate`: Perform validation during training
- `--resume`: Path to checkpoint to resume training from

## Testing

To evaluate the model on the test set:

```bash
python test.py --model_path checkpoints/best_model.pth \
               --test_csv path/to/vqa_rad_test.csv \
               --img_dir path/to/images \
               --device cuda \
               --save_results results/test_results.json
```

### Testing Options

- `--train_csv`: Path to the training CSV file (needed for initialization)
- `--val_csv`: Path to the validation CSV file (needed for initialization)
- `--test_csv`: Path to the test CSV file
- `--img_dir`: Directory containing images
- `--model_path`: Path to trained model checkpoint
- `--device`: Device to use for inference (cuda/cpu)
- `--save_results`: Path to save evaluation results as JSON

## Interactive Demo

To launch the Streamlit application:

```bash
streamlit run app.py
```

This will open a web interface where you can:
1. Upload a medical image
2. Enter a question about the image
3. Get the model's predicted answer

## Pre-trained Model

You can download our pre-trained model from https://huggingface.co/arushiaro/medical_vqa_arushi and place it in the `checkpoints` directory.



This project uses the following pre-trained models:
- [flaviagiammarino/pubmed-clip-vit-base-patch32](https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32) for image encoding
- [cemilcelik/distilgpt2_pubmed](https://huggingface.co/cemilcelik/distilgpt2_pubmed) for language modeling

