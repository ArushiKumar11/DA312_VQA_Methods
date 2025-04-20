import os
import argparse
from copy import deepcopy
import torch
import nltk

from transformers import TrainingArguments, Trainer

# Add the parent directory to path so we can import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultimodalVQAModel
from src.dataset import MultimodalCollator, load_vqa_dataset
from src.metrics import compute_metrics
from src.utils import init_nltk, setup_device, create_multimodal_preprocessors, count_trainable_parameters

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Visual Question Answering model")
    parser.add_argument("--train_csv", type=str, default="dataset/data_train.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="dataset/data_eval.csv", help="Path to test CSV")
    parser.add_argument("--answer_space", type=str, default="dataset/answer_space.txt", help="Path to answer vocabulary")
    parser.add_argument("--images_dir", type=str, default="dataset/images", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased", help="Pretrained text model name")
    parser.add_argument("--image_model", type=str, default="google/vit-base-patch16-224-in21k", help="Pretrained image model name")
    parser.add_argument("--model_name", type=str, default="bert_vit", help="Name for the multimodal model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--intermediate_dim", type=int, default=512, help="Dimension of fusion layer")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize NLTK resources
    init_nltk()
    
    # Set up device
    device = setup_device()
    
    # Load dataset
    print("Loading dataset...")
    dataset, answer_space = load_vqa_dataset(
        train_path=args.train_csv,
        test_path=args.test_csv,
        answer_space_path=args.answer_space
    )
    
    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")
    print(f"Answer space size: {len(answer_space)}")
    
    # Create model and preprocessors
    print(f"Creating model with {args.text_model} and {args.image_model}...")
    tokenizer, preprocessor = create_multimodal_preprocessors(args.text_model, args.image_model)
    
    # Create data collator
    collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        images_dir=args.images_dir
    )
    
    # Create model
    model = MultimodalVQAModel(
        num_labels=len(answer_space),
        intermediate_dim=args.intermediate_dim,
        pretrained_text_name=args.text_model,
        pretrained_image_name=args.image_model
    ).to(device)
    
    print(f"Model has {count_trainable_parameters(model):,} trainable parameters")
    
    # Set up training arguments
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        eval_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,  # Save only the last 3 checkpoints
        metric_for_best_model='wups',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        remove_unused_columns=False,
        num_train_epochs=args.num_epochs,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=True,
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        compute_metrics=lambda eval_tuple: compute_metrics(eval_tuple, answer_space)
    )
    
    # Train model
    print("Starting training...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Evaluate model
    print("Evaluating model...")
    eval_metrics = trainer.evaluate()
    
    print(f"Training complete! Best WUPS score: {eval_metrics.get('eval_wups', 0):.4f}")
    print(f"Accuracy: {eval_metrics.get('eval_acc', 0):.4f}")
    print(f"F1 Score: {eval_metrics.get('eval_f1', 0):.4f}")
    
    return eval_metrics

if __name__ == "__main__":
    main()