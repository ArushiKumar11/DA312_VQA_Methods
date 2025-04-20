import os
import argparse
import random
import torch
import numpy as np
from PIL import Image

# Add the parent directory to path so we can import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultimodalVQAModel
from src.utils import init_nltk, setup_device, create_multimodal_preprocessors, process_single_example
from src.dataset import MultimodalCollator, load_vqa_dataset
from src.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Test a Visual Question Answering model")
    parser.add_argument("--test_csv", type=str, default="dataset/data_eval.csv", help="Path to test CSV")
    parser.add_argument("--answer_space", type=str, default="dataset/answer_space.txt", help="Path to answer vocabulary")
    parser.add_argument("--images_dir", type=str, default="dataset/images", help="Directory containing images")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased", help="Pretrained text model name")
    parser.add_argument("--image_model", type=str, default="google/vit-base-patch16-224-in21k", help="Pretrained image model name")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bert_vit/checkpoint-1560/model.safetensors", 
                       help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def display_example(image_path, question, true_answer, predicted_answer):
    """Display a single example with predictions."""
    print("*" * 50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Question: {question}")
    print(f"True Answer: {true_answer}")
    print(f"Predicted Answer: {predicted_answer}")
    print("*" * 50)
    print()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize NLTK resources
    init_nltk()
    
    # Set up device
    device = setup_device()
    
    # Load dataset
    print("Loading test dataset...")
    dataset, answer_space = load_vqa_dataset(
        train_path=args.test_csv,  # Use test CSV for both to avoid loading training data
        test_path=args.test_csv,
        answer_space_path=args.answer_space
    )
    test_dataset = dataset['test']
    
    # Create model and preprocessors
    print(f"Loading model from checkpoint: {args.checkpoint}")
    tokenizer, preprocessor = create_multimodal_preprocessors(args.text_model, args.image_model)
    
    # Create data collator
    collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        images_dir=args.images_dir
    )
    
    # Load model from checkpoint
    model = MultimodalVQAModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_labels=len(answer_space),
        map_location=device
    )
    model.to(device)
    model.eval()
    
    # Evaluate on random examples
    print(f"\nEvaluating on {args.num_examples} random examples:")
    indices = random.sample(range(len(test_dataset)), min(args.num_examples, len(test_dataset)))
    
    for idx in indices:
        example = test_dataset[idx]
        image_path = os.path.join(args.images_dir, example['image_id'] + '.png')
        
        # Process the example
        result = process_single_example(
            image_path=image_path,
            question=example['question'],
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            model=model,
            answer_space=answer_space,
            device=device
        )
        
        # Display the results
        display_example(
            image_path=image_path,
            question=example['question'],
            true_answer=example['answer'],
            predicted_answer=result['answer']
        )
        
        print("Top 3 predictions:")
        for pred in result['top_predictions']:
            print(f"  {pred['rank']}. {pred['answer']} ({pred['confidence']:.4f})")
        print()
    '''
    # Compute metrics on the whole test set
    print("\nComputing metrics on the entire test set...")
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False
    )
    
    # Evaluate
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, pixel_values, attention_mask, token_type_ids)
            
            # Collect logits and labels
            all_logits.append(outputs['logits'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate logits and labels
    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)
    '''
    # Compute metrics
    #metrics = compute_metrics((all_logits, all_labels), answer_space)
    
    #print(f"Test WUPS: {metrics['wups']:.4f}")
    #print(f"Test Accuracy: {metrics['acc']:.4f}")
    #print(f"Test F1 Score: {metrics['f1']:.4f}")
    
    #return metrics

if __name__ == "__main__":
    main()