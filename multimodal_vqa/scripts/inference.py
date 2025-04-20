#!/usr/bin/env python
"""
Inference script for running VQA model on a single image.
"""

import os
import argparse
import torch
from PIL import Image

# Add the parent directory to path so we can import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultimodalVQAModel
from src.utils import init_nltk, setup_device, create_multimodal_preprocessors, process_single_example

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single image with a VQA model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--answer_space", type=str, default="dataset/answer_space.txt", help="Path to answer vocabulary")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased", help="Pretrained text model name")
    parser.add_argument("--image_model", type=str, default="google/vit-base-patch16-224-in21k", help="Pretrained image model name")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bert_vit/checkpoint-1560/model.safetensors", 
                       help="Path to model checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist.")
        return
    
    # Validate answer space path
    if not os.path.exists(args.answer_space):
        print(f"Error: Answer space file {args.answer_space} does not exist.")
        return
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist.")
        print("Looking for alternative checkpoint paths...")
        
        # Try alternative checkpoint paths
        alt_paths = [
            "checkpoint-1560/model.safetensors",
            "checkpoints/checkpoint-1560/model.safetensors",
            "checkpoint/bert_vit/checkpoint-1560/model.safetensors",
        ]
        for path in alt_paths:
            if os.path.exists(path):
                args.checkpoint = path
                print(f"Found alternative checkpoint at: {path}")
                break
        
        if not os.path.exists(args.checkpoint):
            print("Could not find a valid checkpoint. Please provide a valid checkpoint path.")
            return
    
    # Initialize NLTK resources
    init_nltk()
    
    # Set up device
    device = setup_device()
    
    # Load answer space
    with open(args.answer_space, "r") as f:
        answer_space = f.read().splitlines()
    print(f"Loaded {len(answer_space)} possible answers from {args.answer_space}")
    
    # Create preprocessors
    print(f"Loading models: {args.text_model} and {args.image_model}")
    tokenizer, preprocessor = create_multimodal_preprocessors(args.text_model, args.image_model)
    
    # Load model from checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    model = MultimodalVQAModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_labels=len(answer_space),
        map_location=device
    )
    model.to(device)
    model.eval()
    
    # Process the example
    print("\nProcessing your question...")
    print(f"Image: {args.image_path}")
    print(f"Question: {args.question}")
    
    result = process_single_example(
        image_path=args.image_path,
        question=args.question,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        model=model,
        answer_space=answer_space,
        device=device
    )
    
    # Display the results
    print("\n" + "="*50)
    print(f"Answer: {result['answer']}")
    print("="*50)
    
    print("\nTop predictions:")
    for pred in result['top_predictions']:
        confidence_pct = pred['confidence'] * 100
        print(f"  {pred['rank']}. {pred['answer']} ({confidence_pct:.2f}%)")

if __name__ == "__main__":
    main()