import os
import torch
import nltk
from PIL import Image
from transformers import AutoTokenizer, AutoFeatureExtractor

def init_nltk():
    """Initialize NLTK resources."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def count_trainable_parameters(model):
    """Count trainable parameters in the model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def create_multimodal_preprocessors(text_model='bert-base-uncased', 
                                   image_model='google/vit-base-patch16-224-in21k'):
    """Create tokenizer and feature extractor for the model."""
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    preprocessor = AutoFeatureExtractor.from_pretrained(image_model)
    return tokenizer, preprocessor

def setup_device():
    """Set up the device for training/inference."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    return device

def get_default_checkpoint_path():
    """Get the default path to the checkpoint directory."""
    # Check common locations
    possible_paths = [
        "checkpoints/checkpoint-1560/model.safetensors",
        "checkpoints/bert_vit/checkpoint-1560/model.safetensors",
        "checkpoint/bert_vit/checkpoint-1560/model.safetensors",
        "checkpoint-1560/model.safetensors",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def process_single_example(image_path, question, tokenizer, preprocessor, model, answer_space, device):
    """Process a single example for inference."""
    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    processed_image = preprocessor(images=[image], return_tensors="pt")
    pixel_values = processed_image['pixel_values'].to(device)
    
    # Preprocess text
    encoded_text = tokenizer(
        text=question,
        padding='longest',
        max_length=24,
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    input_ids = encoded_text['input_ids'].to(device)
    token_type_ids = encoded_text['token_type_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_ids, pixel_values, attention_mask, token_type_ids)
    
    # Get prediction
    logits = output["logits"]
    pred_idx = logits.argmax(dim=-1).item()
    
    # Get top 3 predictions with probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top_probs, top_indices = torch.topk(probs, k=3)
    
    predictions = []
    for idx, (prob, pred_id) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())):
        predictions.append({
            "rank": idx + 1,
            "answer": answer_space[pred_id],
            "confidence": float(prob)
        })
    
    return {
        "answer": answer_space[pred_idx],
        "top_predictions": predictions
    }