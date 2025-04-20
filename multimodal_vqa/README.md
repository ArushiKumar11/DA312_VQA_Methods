# Visual Question Answering with Transformers

This repository contains a multimodal transformer-based model for Visual Question Answering (VQA). The model combines BERT for text processing and ViT (Vision Transformer) for image processing to answer questions about images.

## Project Structure

```
vqa-transformer/
├── README.md                   # This file
├── requirements.txt            # Required dependencies
├── data/                       # Data directory
│   ├── README.md               # Instructions for data setup
│   ├── raw/                    # Raw DAQUAR dataset
│   │   └── qa_pairs.txt        # Raw question-answer pairs
│   ├── images/                 # Image files
│   ├── data_train.csv          # Training data
│   ├── data_eval.csv           # Evaluation data
│   └── answer_space.txt        # Answer vocabulary
├── src/                        # Source code
│   ├── __init__.py
│   ├── model.py                # Model architecture
│   ├── dataset.py              # Dataset and collator
│   ├── process_data.py         # Data processing script
│   ├── utils.py                # Utility functions
│   └── metrics.py              # Evaluation metrics (WUPS, accuracy, F1)
├── scripts/                    # Training and testing scripts
│   ├── train.py                # Training script
│   └── test.py                 # Evaluation script
├── app/                        # Streamlit demo application
│   ├── app.py                  # Streamlit app
│   └── example_images/         # Example images for demo
└── checkpoints/                # Model checkpoints
    └── bert_vit/               # BERT+ViT model
        └── checkpoint-1560/    # Best checkpoint
```

## Model Architecture

The model combines text and image transformers:
- **Text Encoder**: BERT (bert-base-uncased)
- **Image Encoder**: Vision Transformer (google/vit-base-patch16-224-in21k)
- **Fusion Layer**: Concatenation of text and image features followed by MLP
- **Classifier**: Linear layer mapping to answer vocabulary

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vqa-transformer.git
cd vqa-transformer
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset and place it in the data directory (see data/README.md for details)

## Usage

### Training

To train the model:

```bash
python scripts/train.py --train_csv data/data_train.csv --test_csv data/data_eval.csv --answer_space data/answer_space.txt --images_dir data/images --output_dir checkpoints
```

Optional arguments:
- `--text_model`: Pretrained text model name (default: 'bert-base-uncased')
- `--image_model`: Pretrained image model name (default: 'google/vit-base-patch16-224-in21k')
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 5)
- `--fp16`: Use mixed precision training (flag)

### Testing

To evaluate the model:

```bash
python scripts/test.py --test_csv data/data_eval.csv --answer_space data/answer_space.txt --images_dir data/images --checkpoint checkpoints/bert_vit/checkpoint-1560/model.safetensors
```

Optional arguments:
- `--num_examples`: Number of examples to display (default: 5)
- `--batch_size`: Batch size for evaluation (default: 32)

### Streamlit Demo

To run the interactive demo:

```bash
cd app
streamlit run app.py
```

The app will allow you to:
- Upload your own images
- Ask questions about the images
- See the model's predictions

## Performance Metrics

The model is evaluated on the following metrics:
- **WUPS Score**: Wu-Palmer Similarity for semantic similarity between predicted and true answers
- **Accuracy**: Standard classification accuracy
- **F1 Score**: Macro F1 score across all answer classes

## Trained Model Checkpoint

The checkpoint directory contains the best model checkpoint (checkpoint-1560) that was obtained after training the model on the DAQUAR dataset using Google Colab. This is not a pretrained model that you would download from elsewhere, but rather the result of running the training notebook.

You can use this checkpoint in two ways:
1. For inference - to immediately start answering visual questions without further training
2. For continued training - if you faced GPU limitations during the initial training, you can resume from this checkpoint to further improve the model

The checkpoint contains all the necessary state to either perform inference or continue the training process.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The model architecture is based on multimodal transformer approaches
- BERT implementation: https://huggingface.co/bert-base-uncased
- ViT implementation: https://huggingface.co/google/vit-base-patch16-224-in21k