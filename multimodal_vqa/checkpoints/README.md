# Model Checkpoints

This directory contains saved model checkpoints from training. The current structure is:

```
checkpoints/
└── bert_vit/
    └── checkpoint-1560/
        ├── model.safetensors  # Model weights
        ├── optimizer.pt       # Optimizer state
        └── scheduler.pt       # Scheduler state
        └── trainer_state.json # Training state
```

## Checkpoint Structure

- **bert_vit**: Directory for the BERT + ViT multimodal model
- **checkpoint-1560**: Best checkpoint from training, saved after 1560 steps

## Using Checkpoints

To use a checkpoint for inference or continued training:

1. For inference, use the `test.py` script:
   ```bash
   python scripts/test.py --checkpoint checkpoints/bert_vit/checkpoint-1560/model.safetensors
   ```

2. For the Streamlit demo:
   ```bash
   cd app
   CHECKPOINT_PATH=../checkpoints/bert_vit/checkpoint-1560/model.safetensors streamlit run app.py
   ```

3. For continuing training from a checkpoint:
   ```bash
   python scripts/train.py --output_dir checkpoints/bert_vit_continued
   ```

## Checkpoint Format

The model weights are saved in the `.safetensors` format, which is a safer and faster alternative to PyTorch's native format. You can load them using:

```python
from safetensors.torch import load_file
state_dict = load_file("checkpoints/bert_vit/checkpoint-1560/model.safetensors")
model.load_state_dict(state_dict)
```

If you don't have safetensors installed, you can use PyTorch's native format:

```python
state_dict = torch.load("checkpoints/bert_vit/checkpoint-1560/pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict)
```

## Creating New Checkpoints

When you train a model with `train.py`, new checkpoints will be saved here. The script will keep the best 3 checkpoints based on the WUPS score on the evaluation set.