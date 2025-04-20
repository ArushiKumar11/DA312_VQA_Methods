import os
import argparse
import torch
import torch.optim as optim
from torch.nn import functional as nnf
from tqdm import tqdm
from model import VQAModel, get_loaders


def train_one_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        mask = batch['mask'].to(device)
        q_len = batch['q_len'].to(device)
        
        optimizer.zero_grad()
        logits = model(images, tokens, mask, q_len)

        shift = 0
        loss = 0
        for b in range(logits.size(0)):
            condensed_tokens = tokens[b, q_len[b]+1+1:]
            condensed_logits = logits[b, shift + q_len[b]+1:-1]
            loss += nnf.cross_entropy(
                condensed_logits.reshape(-1, logits.shape[-1]), 
                condensed_tokens.flatten(), 
                ignore_index=0
            )

        loss = loss / logits.size(0)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    return running_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            tokens = batch['tokens'].to(device)
            mask = batch['mask'].to(device)
            q_len = batch['q_len'].to(device)
            
            logits = model(images, tokens, mask, q_len)
            
            shift = 0
            loss = 0
            for b in range(logits.size(0)):
                condensed_tokens = tokens[b, q_len[b]+1+1:]
                condensed_logits = logits[b, shift + q_len[b]+1:-1]
                loss += nnf.cross_entropy(
                    condensed_logits.reshape(-1, logits.shape[-1]), 
                    condensed_tokens.flatten(), 
                    ignore_index=0
                )

            loss = loss / logits.size(0)
            running_loss += loss.item()
            
    return running_loss / len(dataloader)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, _ = get_loaders(
        csv_train=args.train_csv,
        csv_val=args.val_csv,
        csv_test=args.test_csv,
        img_dir=args.img_dir,
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = VQAModel().to(device)
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")
        
        # Validate
        if args.validate:
            val_loss = validate(model, val_loader, device)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"Best model saved with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': best_val_loss if args.validate else None,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': best_val_loss if args.validate else None,
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQA model on medical images")
    
    parser.add_argument("--train_csv", type=str, default="vqa_rad_train.csv", 
                        help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, default="vqa_rad_valid.csv", 
                        help="Path to validation CSV file")
    parser.add_argument("--test_csv", type=str, default="vqa_rad_test.csv", 
                        help="Path to test CSV file")
    parser.add_argument("--img_dir", type=str, default="img", 
                        help="Directory containing images")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-3, 
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train for")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1, 
                        help="Save checkpoint every N epochs")
    parser.add_argument("--validate", action="store_true", 
                        help="Perform validation during training")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    main(args)