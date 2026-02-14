"""
RIFashion: A Hybrid CNN-ViT Framework for Rotation-Invariant Apparel Classification

Complete implementation matching the research paper specifications.
Author: [Your Name]
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import time
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_json_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    Handles nested dicts, lists, numpy arrays, and numpy scalars.
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# ============================================================================
# 1. DATASET: ROTATED FASHION-MNIST WITH CONTINUOUS AUGMENTATION
# ============================================================================


class RotatedFashionMNIST(Dataset):
    """
    Fashion-MNIST dataset with continuous rotation augmentation.
    Implements the data preprocessing strategy described in Section IV-A.
    """

    def __init__(self, root='./data', train=True, download=True, rotation_prob=0.8):
        """
        Args:
            root: Dataset root directory
            train: If True, use training set (60,000 samples)
            download: If True, download dataset if not present
            rotation_prob: Probability of applying rotation (p_rot in paper)
        """
        self.dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=None
        )

        self.rotation_prob = rotation_prob
        self.train = train

        # Fashion-MNIST class names
        self.classes = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            img: Normalized tensor (1, 28, 28) in range [-1, 1]
            label: Class label (0-9)
            angle: Rotation angle in degrees [0, 360)
        """
        img, label = self.dataset[idx]

        # Apply continuous rotation augmentation
        if self.train and np.random.random() < self.rotation_prob:
            angle = float(np.random.uniform(0, 360))
            img = transforms.functional.rotate(img, angle)
        else:
            angle = 0.0

        # Convert to tensor and normalize to [-1, 1]
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, (0.5,), (0.5,))

        return img, label, torch.tensor(angle, dtype=torch.float32)


# ============================================================================
# 2. CNN FEATURE EXTRACTOR (Section III-B-A)
# ============================================================================

class CNNFeatureExtractor(nn.Module):
    """
    Three-layer CNN backbone for local feature extraction.
    Architecture: Conv(32) -> Pool -> Conv(64) -> Pool -> Conv(128) -> Pool
    Output: 3×3×128 feature map (flattened to 1152-d vector)
    """

    def __init__(self, input_channels=1, dropout_rate=0.25):
        super(CNNFeatureExtractor, self).__init__()

        # Convolutional Block 1: 28×28 -> 14×14
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional Block 2: 14×14 -> 7×7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Convolutional Block 3: 7×7 -> 3×3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Output feature dimension
        self.feature_dim = 128 * 3 * 3  # 1152

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 1, 28, 28)
        Returns:
            features: Flattened CNN features (B, 1152)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 32, 14, 14)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 64, 7, 7)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 128, 3, 3)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (B, 1152)

        return x


# ============================================================================
# 3. TRANSFORMER ENCODER BLOCK (Section III-B-C)
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Lightweight transformer encoder block with multi-head self-attention.
    Follows the architecture in Equations (5) and (6).
    """

    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network (MLP)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, seq_len, embed_dim)
        Returns:
            Output tensor (B, seq_len, embed_dim)
        """
        # Multi-Head Self-Attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x


# ============================================================================
# 4. RIFASHION: COMPLETE HYBRID ARCHITECTURE (Section III)
# ============================================================================

class RIFashion(nn.Module):
    """
    RIFashion: Hybrid CNN-Vision Transformer for Rotation-Invariant Classification

    Architecture:
    1. CNN Feature Extractor (3 conv blocks)
    2. Feature Projection to transformer embedding
    3. Lightweight Transformer Encoder (2 layers)
    4. Dual Prediction Heads (classification + rotation)

    Total Parameters: ~1.2M
    """

    def __init__(
            self,
            num_classes=10,
            embed_dim=256,
            num_transformer_layers=2,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1
    ):
        super(RIFashion, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(input_channels=1, dropout_rate=0.25)

        # 2. Feature Projection (Equation 2)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.cnn.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        # 3. Learnable Class Token (Equation 3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 4. Positional Embedding (Equation 4)
        self.pos_embed = nn.Parameter(torch.randn(1, 2, embed_dim))  # [CLS] + feature

        # 5. Transformer Encoder (Equations 5-6)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_transformer_layers)
        ])

        # 6. Classification Head (Equation 7)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 7. Rotation Prediction Head (Equation 8)
        self.rotation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1], scaled to [0, 360]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights as described in Section III-B"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_attention=False):
        """
        Forward pass through RIFashion architecture

        Args:
            x: Input images (B, 1, 28, 28)
            return_attention: If True, return attention weights for visualization

        Returns:
            class_logits: Classification predictions (B, num_classes)
            rotation_pred: Rotation angle predictions (B,) in [0, 360]
        """
        batch_size = x.size(0)

        # 1. CNN Feature Extraction
        cnn_features = self.cnn(x)  # (B, 1152)

        # 2. Project to Transformer Embedding
        features = self.feature_projection(cnn_features)  # (B, embed_dim)
        features = features.unsqueeze(1)  # (B, 1, embed_dim)

        # 3. Add Class Token (Equation 3)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, features], dim=1)  # (B, 2, embed_dim)

        # 4. Add Positional Embedding (Equation 4)
        x = x + self.pos_embed

        # 5. Apply Transformer Blocks (Equations 5-6)
        attention_weights = []
        for transformer_block in self.transformer:
            if return_attention:
                # Store attention for visualization
                attn_out, attn_weights = transformer_block.attention(x, x, x)
                attention_weights.append(attn_weights)
                x = transformer_block.norm1(x + attn_out)
                x = transformer_block.norm2(x + transformer_block.mlp(x))
            else:
                x = transformer_block(x)

        # 6. Extract Class Token Output
        cls_output = x[:, 0]  # (B, embed_dim)

        # 7. Dual Predictions
        class_logits = self.classifier(cls_output)  # (B, num_classes)
        rotation_pred = self.rotation_head(cls_output).squeeze(-1) * 360.0  # (B,) [0, 360]

        if return_attention:
            return class_logits, rotation_pred, attention_weights

        return class_logits, rotation_pred

    def get_num_params(self):
        """Calculate total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================================
# 5. DUAL-TASK LOSS FUNCTION (Section III-C)
# ============================================================================

class DualTaskLoss(nn.Module):
    """
    Combined loss for classification and rotation prediction (Equation 1).
    L_total = λ * L_cls + (1-λ) * L_rot
    """

    def __init__(self, alpha=0.7):
        """
        Args:
            alpha (λ in paper): Weight for classification loss [0, 1]
                              Default: 0.7 as specified in Section III-C
        """
        super(DualTaskLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, class_logits, rotation_pred, class_labels, rotation_angles):
        """
        Compute dual-task loss

        Args:
            class_logits: Predicted class logits (B, num_classes)
            rotation_pred: Predicted rotation angles (B,) [0, 360]
            class_labels: Ground truth class labels (B,)
            rotation_angles: Ground truth rotation angles (B,) [0, 360]

        Returns:
            total_loss: Combined loss
            cls_loss: Classification loss component
            rot_loss: Rotation loss component
        """
        # Classification Loss (Equation 9)
        cls_loss = self.ce_loss(class_logits, class_labels)

        # Rotation Prediction Loss (Equation 10)
        # Normalize to [0, 1] for stable training
        rotation_pred_norm = rotation_pred / 360.0
        rotation_angles_norm = rotation_angles / 360.0
        rot_loss = self.mse_loss(rotation_pred_norm, rotation_angles_norm)

        # Combined Loss (Equation 1)
        total_loss = self.alpha * cls_loss + (1 - self.alpha) * rot_loss

        return total_loss, cls_loss, rot_loss


# ============================================================================
# 6. TRAINER CLASS (Section IV-B)
# ============================================================================

class RIFashionTrainer:
    """
    Training and evaluation pipeline for RIFashion.
    Implements the training strategy from Section III-D.
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            device='cuda',
            learning_rate=1e-3,
            weight_decay=1e-4,
            num_epochs=30,
            alpha=0.7,
            save_dir='./results'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # Loss function (Equation 1)
        self.criterion = DualTaskLoss(alpha=alpha)

        # AdamW optimizer (Section III-D-B)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Cosine annealing scheduler (Section III-D-B)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_cls_loss': [], 'train_rot_loss': [],
            'val_loss': [], 'val_acc': [], 'val_rot_mae': [],
            'learning_rate': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_rot_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.num_epochs} [Train]',
            ncols=100
        )

        for images, labels, angles in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            angles = angles.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            class_logits, rotation_pred = self.model(images)

            # Compute loss
            loss, cls_loss, rot_loss = self.criterion(
                class_logits, rotation_pred, labels, angles
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_rot_loss += rot_loss.item()

            _, predicted = class_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_rot_loss = total_rot_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy, avg_cls_loss, avg_rot_loss

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        rotation_errors = []

        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch + 1}/{self.num_epochs} [Val]  ',
            ncols=100
        )

        with torch.no_grad():
            for images, labels, angles in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                angles = angles.to(self.device)

                # Forward pass
                class_logits, rotation_pred = self.model(images)

                # Compute loss
                loss, _, _ = self.criterion(
                    class_logits, rotation_pred, labels, angles
                )

                # Statistics
                total_loss += loss.item()
                _, predicted = class_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Rotation prediction error
                rot_error = torch.abs(rotation_pred - angles).cpu().numpy()
                rotation_errors.extend(rot_error)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        rot_mae = np.mean(rotation_errors)

        return avg_loss, accuracy, rot_mae

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("RIFashion: Rotation-Invariant Fashion Classification")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Number of epochs: {self.num_epochs}")

        total_params, trainable_params = self.model.get_num_params()
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 ** 2:.2f} MB")
        print("=" * 80 + "\n")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_acc, train_cls_loss, train_rot_loss = self.train_epoch(epoch)

            # Validation phase
            val_loss, val_acc, val_rot_mae = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Save history (ensure all values are Python native types)
            self.history['train_loss'].append(float(train_loss))
            self.history['train_acc'].append(float(train_acc))
            self.history['train_cls_loss'].append(float(train_cls_loss))
            self.history['train_rot_loss'].append(float(train_rot_loss))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(val_acc))
            self.history['val_rot_mae'].append(float(val_rot_mae))
            self.history['learning_rate'].append(float(current_lr))

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Cls: {train_cls_loss:.4f} | Rot: {train_rot_loss:.4f}")
            print(f"  Train Acc:  {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:.2f}%")
            print(f"  Rot MAE:    {val_rot_mae:.2f}°")
            print(f"  LR:         {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': float(val_acc),
                    'val_rot_mae': float(val_rot_mae),
                    'history': make_json_serializable(self.history),  # Use helper function
                    'config': {
                        'num_classes': self.model.num_classes,
                        'embed_dim': self.model.embed_dim,
                    }
                }

                torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
                print(f"  ✓ Best model saved with Val Acc: {val_acc:.2f}%")

            print()

        training_time = time.time() - start_time

        print("=" * 80)
        print(f"Training completed in {training_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")
        print("=" * 80 + "\n")

        # Save training history with proper type conversion
        try:
            history_serializable = make_json_serializable(self.history)
            with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
                json.dump(history_serializable, f, indent=4)
            print(f"Training history saved to {os.path.join(self.save_dir, 'training_history.json')}")
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")

        return self.history


# ============================================================================
# 7. EVALUATION AND VISUALIZATION
# ============================================================================

class Evaluator:
    """Comprehensive evaluation and visualization for RIFashion"""

    def __init__(self, model, test_loader, device='cuda', save_dir='./results'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.classes = test_loader.dataset.classes

    def evaluate(self):
        """Comprehensive evaluation with per-class metrics"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_rot_preds = []
        all_rot_true = []

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        print("\nEvaluating model...")

        with torch.no_grad():
            for images, labels, angles in tqdm(self.test_loader, desc='Evaluation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                angles = angles.to(self.device)

                class_logits, rotation_pred = self.model(images)
                _, predicted = class_logits.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_rot_preds.extend(rotation_pred.cpu().numpy())
                all_rot_true.extend(angles.cpu().numpy())

                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1

        # Calculate metrics (convert to Python native types immediately)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_rot_preds = np.array(all_rot_preds)
        all_rot_true = np.array(all_rot_true)

        overall_acc = float(np.mean(all_preds == all_labels) * 100)  # Convert to Python float
        rot_mae = float(np.mean(np.abs(all_rot_preds - all_rot_true)))  # Convert to Python float

        # Per-class accuracy (convert to Python native types)
        per_class_acc = {}
        for class_idx in range(len(self.classes)):
            if class_total[class_idx] > 0:
                acc = float(100. * class_correct[class_idx] / class_total[class_idx])
                per_class_acc[self.classes[class_idx]] = acc

        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        print(f"Rotation MAE: {rot_mae:.2f}°")
        print("\nPer-Class Accuracy:")
        for class_name, acc in per_class_acc.items():
            print(f"  {class_name:15s}: {acc:.2f}%")
        print("=" * 80 + "\n")

        # Create results dictionary with JSON-serializable types
        results = {
            'overall_accuracy': overall_acc,
            'rotation_mae': rot_mae,
            'per_class_accuracy': per_class_acc,
            'predictions': [int(x) for x in all_preds.tolist()],  # Ensure int type
            'labels': [int(x) for x in all_labels.tolist()]  # Ensure int type
        }

        # Save results (now guaranteed to be JSON-serializable)
        try:
            with open(os.path.join(self.save_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Evaluation results saved to {os.path.join(self.save_dir, 'evaluation_results.json')}")
        except Exception as e:
            print(f"Warning: Could not save evaluation results: {e}")

        return results

    def visualize_predictions(self, num_samples=20, save_name='predictions.png'):
        """Visualize sample predictions"""
        self.model.eval()

        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.flatten()

        # Get random samples
        dataset = self.test_loader.dataset
        indices = np.random.choice(len(dataset), num_samples, replace=False)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                img, label, angle = dataset[idx]
                img_input = img.unsqueeze(0).to(self.device)

                class_logits, rotation_pred = self.model(img_input)
                pred_class = class_logits.argmax(dim=1).item()
                pred_angle = rotation_pred.item()

                # Denormalize for display
                img_display = img.squeeze().cpu().numpy() * 0.5 + 0.5

                axes[i].imshow(img_display, cmap='gray')
                axes[i].axis('off')

                true_label = self.classes[label]
                pred_label = self.classes[pred_class]

                color = 'green' if pred_class == label else 'red'
                title = f'True: {true_label}\nPred: {pred_label}\n'
                title += f'Angle: {angle:.0f}° → {pred_angle:.0f}°'

                axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {os.path.join(self.save_dir, save_name)}")
        plt.close()

    def plot_confusion_matrix(self, save_name='confusion_matrix.png'):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                class_logits, _ = self.model(images)
                _, predicted = class_logits.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {os.path.join(self.save_dir, save_name)}")
        plt.close()


def plot_training_history(history, save_dir='./results'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Classification Accuracy', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    axes[0, 2].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Classification Loss
    axes[1, 0].plot(epochs, history['train_cls_loss'], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Classification Loss', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Rotation Loss
    axes[1, 1].plot(epochs, history['train_rot_loss'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Rotation Prediction Loss', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Rotation MAE
    axes[1, 2].plot(epochs, history['val_rot_mae'], 'r-', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('MAE (degrees)')
    axes[1, 2].set_title('Rotation Prediction Error', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"Training history saved to {os.path.join(save_dir, 'training_history.png')}")
    plt.close()


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""

    # Configuration (Section IV-B)
    CONFIG = {
        'batch_size': 128,
        'num_epochs': 30,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'alpha': 0.7,  # Loss weighting parameter λ
        'embed_dim': 256,
        'num_transformer_layers': 2,
        'num_heads': 4,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'rotation_prob_train': 0.8,
        'rotation_prob_val': 0.5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './results'
    }

    print("\n" + "=" * 80)
    print("RIFashion: A Hybrid CNN-ViT Framework for Rotation-Invariant Classification")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 80 + "\n")

    # Create datasets (Section IV-A)
    print("Loading Fashion-MNIST dataset...")
    train_dataset = RotatedFashionMNIST(
        root='./data',
        train=True,
        download=True,
        rotation_prob=CONFIG['rotation_prob_train']
    )

    val_dataset = RotatedFashionMNIST(
        root='./data',
        train=False,
        download=True,
        rotation_prob=CONFIG['rotation_prob_val']
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False,
        persistent_workers=True if CONFIG['num_workers'] > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False,
        persistent_workers=True if CONFIG['num_workers'] > 0 else False
    )

    # Create model (Section III)
    print("\nInitializing RIFashion model...")
    model = RIFashion(
        num_classes=10,
        embed_dim=CONFIG['embed_dim'],
        num_transformer_layers=CONFIG['num_transformer_layers'],
        num_heads=CONFIG['num_heads'],
        mlp_ratio=CONFIG['mlp_ratio'],
        dropout=CONFIG['dropout']
    )

    # Create trainer
    trainer = RIFashionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        num_epochs=CONFIG['num_epochs'],
        alpha=CONFIG['alpha'],
        save_dir=CONFIG['save_dir']
    )

    # Train model
    history = trainer.train()

    # Plot training history
    print("\nGenerating training visualizations...")
    plot_training_history(history, CONFIG['save_dir'])

    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(
        os.path.join(CONFIG['save_dir'], 'best_model.pth'),
        map_location=CONFIG['device']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Comprehensive evaluation
    evaluator = Evaluator(
        model=model,
        test_loader=val_loader,
        device=CONFIG['device'],
        save_dir=CONFIG['save_dir']
    )

    results = evaluator.evaluate()
    evaluator.visualize_predictions()
    evaluator.plot_confusion_matrix()

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nResults saved to: {CONFIG['save_dir']}")
    print("Generated files:")
    print("  - best_model.pth: Best model checkpoint")
    print("  - training_history.json: Training metrics")
    print("  - training_history.png: Training curves")
    print("  - evaluation_results.json: Test set performance")
    print("  - predictions.png: Sample predictions")
    print("  - confusion_matrix.png: Confusion matrix")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
