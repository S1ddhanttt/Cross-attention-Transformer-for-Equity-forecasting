import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

from dataset import StockRAGDataset
from model import MultimodalStockTransformer

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train():
    print(f"Using Compute Device: {DEVICE}")
    
    # 1. Instantiate Full Dataset
    dataset = StockRAGDataset()
    total_samples = len(dataset)
    
    # 2. Chronological Split (80% Train, 20% Validation)
    split_idx = int(0.8 * total_samples)
    train_indices = list(range(0, split_idx))
    val_indices = list(range(split_idx, total_samples))
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training Samples: {len(train_set)} | Validation Samples: {len(val_set)}")
    
    # 3. Model, Loss, and Optimizer Setup
    model = MultimodalStockTransformer(
        price_features=6,
        rag_dim=384,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_classes=3,
        dropout=0.2
    ).to(DEVICE)
    
    train_targets = [dataset.samples[i][3] for i in train_indices]
    class_counts = np.bincount(train_targets, minlength=3)
    class_weights = 1.0 / (class_counts + 1e-5)
    class_weights = class_weights / class_weights.sum()
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_f1 = 0.0
    
    # 4. Training Loop
    print("\nStarting Multimodal Transformer Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_preds, train_true = [], []
        
        for price_b, text_b, target_b in train_loader:
            price_b = price_b.to(DEVICE)
            text_b = text_b.to(DEVICE)
            target_b = target_b.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(price_b, text_b)
            loss = criterion(logits, target_b)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item() * price_b.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(target_b.cpu().numpy())
            
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_set)
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average="macro", zero_division=0)
        
        # 5. Validation Loop
        model.eval()
        total_val_loss = 0.0
        val_preds, val_true = [], []
        
        with torch.no_grad():
            for price_b, text_b, target_b in val_loader:
                price_b = price_b.to(DEVICE)
                text_b = text_b.to(DEVICE)
                target_b = target_b.to(DEVICE)
                
                logits, _ = model(price_b, text_b)
                loss = criterion(logits, target_b)
                
                total_val_loss += loss.item() * price_b.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(target_b.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_set)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Train F1: {train_f1:.3f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.3f}")
        
        # Save best model checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            }, f"{CHECKPOINT_DIR}/best_multimodal_transformer.pth")
            
    print(f"\n✅ Training Complete! Best Validation Macro F1: {best_val_f1:.4f}")
    print(f"Best model saved to '{CHECKPOINT_DIR}/best_multimodal_transformer.pth'")

if __name__ == "__main__":
    train()