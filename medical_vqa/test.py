import os
import argparse
import json
import torch
from tqdm import tqdm
from model import VQAModel, get_loaders, evaluate_model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get test data loader
    _, _, test_loader = get_loaders(
        csv_train=args.train_csv,
        csv_val=args.val_csv,
        csv_test=args.test_csv,
        img_dir=args.img_dir,
        batch_size=1
    )
    
    # Initialize model
    model = VQAModel().to(device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint)
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate model
    results, metrics = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Save results if specified
    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        
        output = {
            "metrics": metrics,
            "results": results
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VQA model on medical images")
    
    parser.add_argument("--train_csv", type=str, default="vqa_rad_train.csv", 
                        help="Path to training CSV file (needed for initialization)")
    parser.add_argument("--val_csv", type=str, default="vqa_rad_valid.csv", 
                        help="Path to validation CSV file (needed for initialization)")
    parser.add_argument("--test_csv", type=str, default="vqa_rad_test.csv", 
                        help="Path to test CSV file")
    parser.add_argument("--img_dir", type=str, default="img", 
                        help="Directory containing images")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use for inference (cuda/cpu)")
    parser.add_argument("--save_results", type=str, default=None, 
                        help="Path to save evaluation results as JSON")
    
    args = parser.parse_args()
    main(args)