import json
from pathlib import Path

notebook_path = Path("notebooks/resnet50_training.ipynb")

cells = []

def add_markdown(content):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": content.strip().splitlines(keepends=True)
    })

def add_code(lines):
    if isinstance(lines, str):
        lines = lines.strip().splitlines(keepends=True)
    else:
        lines = [l + "\n" for l in lines]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# CELL 1 — Title and Experiment Overview
add_markdown("""# ResNet50 Training & Evaluation Notebook
### Project: AI-Based Automated Road-Surface Condition Monitoring Using Vehicle-Mounted Cameras

**Dataset**: RDD2020 / Classification Dataset (4 Damage Categories: D00 Longitudinal Crack, D10 Transverse Crack, D20 Alligator Crack, D40 Pothole)  
**Model Architecture**: ResNet50 (Pretrained on ImageNet)

---

#### Overview & Objectives:
- **Model Architecture:** ResNet50 (pre-trained on ImageNet, fine-tuned for 4 road damage classes).
- **Target Damage Classes:**
  - **D00:** Longitudinal Crack
  - **D10:** Transverse Crack
  - **D20:** Alligator Crack
  - **D40:** Pothole
- **Dataset:** Preprocessed RDD2020 image classification crop dataset located in `data/classification/`.

### Structure of the 22 Notebook Cells

| Cell # | Type | Description |
|---|---|---|
| **Cell 1** | Markdown | Title & Project Overview (`AI-Based Automated Road-Surface Condition Monitoring`) |
| **Cell 2** | Code | Clean, grouped imports & project root resolution |
| **Cell 3** | Code | Centralized configuration, reproducibility seeds (`42`), safety flags (`RUN_TRAINING = False`) |
| **Cell 4** | Code | Dataset paths & class mappings (`D00`, `D10`, `D20`, `D40`) |
| **Cell 5** | Code | Dynamic dataset split & class count verification |
| **Cell 6** | Code | PyTorch DataLoaders creation |
| **Cell 7** | Code | Sample dataset image grid visualization |
| **Cell 8** | Code | ResNet50 model initialization |
| **Cell 9** | Code | Model architecture & parameter count (~23.5M) verification |
| **Cell 10** | Code | Criterion (`CrossEntropyLoss`), Optimizer (`AdamW`), Scheduler (`ReduceLROnPlateau`) |
| **Cell 11** | Code | Controlled training loop (skipped when `RUN_TRAINING = False`) |
| **Cell 12** | Code | Model checkpoint saving logic (skipped when `RUN_TRAINING = False`) |
| **Cell 13** | Code | Training history table display |
| **Cell 14** | Code | Training vs Validation Loss plot |
| **Cell 15** | Code | Training vs Validation Accuracy plot |
| **Cell 16** | Code | Best checkpoint loading (`results/models/resnet50/best_model.pth`) |
| **Cell 17** | Code | Validation set evaluation |
| **Cell 18** | Code | Held-out test set evaluation (3,669 test samples) |
| **Cell 19** | Code | Detailed test classification report dataframe |
| **Cell 20** | Code | Test confusion matrix plot & export |
| **Cell 21** | Code | Dynamic sample test predictions visual grid |
| **Cell 22** | Code | Final experiment summary table & report |
""")

# CELL 2 — Imports
add_code([
    "import os",
    "import sys",
    "import random",
    "import time",
    "from pathlib import Path",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "from PIL import Image",
    "",
    "import torch",
    "import torch.nn as nn",
    "import torch.optim as optim",
    "from torch.optim.lr_scheduler import ReduceLROnPlateau",
    "from torch.utils.data import DataLoader",
    "from torchvision import transforms",
    "from torchvision.datasets import ImageFolder",
    "",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support",
    "",
    "# Add parent directory to sys.path to access src modules",
    "CURRENT_DIR = Path.cwd()",
    "PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == 'notebooks' else CURRENT_DIR",
    "if str(PROJECT_ROOT) not in sys.path:",
    "    sys.path.insert(0, str(PROJECT_ROOT))",
    "",
    "from src.models.resnet50 import get_resnet50_model, ResNet50RoadClassifier",
    "from src.utils import set_seed, get_device, create_dataloaders, get_data_transforms, RDD2020_CLASSES",
    "from src.evaluate import evaluate_model, generate_classification_metrics, plot_and_save_confusion_matrix",
    "",
    "# Reproducibility & Device Configuration",
    "set_seed(42)",
    "device = get_device()",
    "print(f'Using PyTorch version: {torch.__version__}')"
])

# CELL 3 — Reproducibility, Device and Configuration
add_code([
    "# Reproducibility Seed Setup",
    "SEED = 42",
    "set_seed(SEED)",
    "",
    "# Device Selection (GPU if available, else CPU)",
    "DEVICE = device",
    "",
    "# Training & Model Hyperparameters Configuration",
    "IMAGE_SIZE = (224, 224)",
    "BATCH_SIZE = 32",
    "NUM_WORKERS = 0  # 0 for cross-platform stability across OS environments",
    "NUM_CLASSES = 4",
    "LEARNING_RATE = 1e-4",
    "WEIGHT_DECAY = 1e-4",
    "EPOCHS = 10",
    "PATIENCE = 5",
    "DROPOUT_RATE = 0.2",
    "PRETRAINED = True",
    "FREEZE_BACKBONE = False",
    "",
    "# Notebook Execution Safety Flags",
    "RUN_TRAINING = False          # Default False to prevent accidental retraining",
    "SAVE_BEST_CHECKPOINT = False  # Default False to preserve existing trained weights",
    "",
    "print('Configuration set successfully.')",
    "print(f'Device: {DEVICE} | Seed: {SEED} | Image Size: {IMAGE_SIZE} | Batch Size: {BATCH_SIZE} | RUN_TRAINING: {RUN_TRAINING}')"
])

# CELL 4 — Dataset Paths and Class Names
add_code([
    "# Configurable Dataset Path & Results Directories",
    "DATA_DIR = PROJECT_ROOT / 'data' / 'classification'",
    "if not DATA_DIR.exists():",
    "    alt_dir = PROJECT_ROOT / 'data' / 'RDD2020'",
    "    if alt_dir.exists():",
    "        DATA_DIR = alt_dir",
    "",
    "DATASET_DIR = str(DATA_DIR)",
    "TRAIN_DIR = DATA_DIR / 'train'",
    "VAL_DIR = DATA_DIR / 'val'",
    "TEST_DIR = DATA_DIR / 'test'",
    "",
    "RESULTS_DIR = PROJECT_ROOT / 'results'",
    "FIGURES_DIR = RESULTS_DIR / 'figures' / 'resnet50'",
    "MODELS_DIR = RESULTS_DIR / 'models' / 'resnet50'",
    "TABLES_DIR = RESULTS_DIR / 'tables' / 'resnet50'",
    "",
    "FIGURE_SAVE_DIR = FIGURES_DIR",
    "MODEL_SAVE_DIR = MODELS_DIR",
    "TABLE_SAVE_DIR = TABLES_DIR",
    "",
    "BEST_CHECKPOINT_PATH = MODELS_DIR / 'best_model.pth'",
    "HISTORY_CSV_PATH = TABLES_DIR / 'training_history.csv'",
    "",
    "# Ensure output directories exist",
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)",
    "MODELS_DIR.mkdir(parents=True, exist_ok=True)",
    "TABLES_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "# Target Class Names Definition & Human-Readable Mapping",
    "CLASS_NAMES = ['D00', 'D10', 'D20', 'D40']",
    "READABLE_CLASS_NAMES = {",
    "    'D00': 'Longitudinal Crack',",
    "    'D10': 'Transverse Crack',",
    "    'D20': 'Alligator Crack',",
    "    'D40': 'Pothole'",
    "}",
    "",
    "print(f'Dataset Directory: {DATA_DIR}')",
    "print(f'Target Checkpoint Path: {BEST_CHECKPOINT_PATH}')",
    "print(f'Results Output Paths Configured:')",
    "print(f'  - Figures: {FIGURES_DIR}')",
    "print(f'  - Models:  {MODELS_DIR}')",
    "print(f'  - Tables:  {TABLES_DIR}')",
    "print(f'Target Output Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}')"
])

# CELL 5 — Dataset Verification
add_code([
    "def verify_dataset_structure(data_dir, class_names):",
    "    counts = {}",
    "    for split in ['train', 'val', 'test']:",
    "        split_path = Path(data_dir) / split",
    "        counts[split] = {}",
    "        if split_path.exists():",
    "            for cls in class_names:",
    "                cls_path = split_path / cls",
    "                if cls_path.exists():",
    "                    images = [f for f in cls_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]",
    "                    counts[split][cls] = len(images)",
    "                else:",
    "                    counts[split][cls] = 0",
    "        else:",
    "            print(f'Directory missing: {split_path}')",
    "    return counts",
    "",
    "dataset_counts = verify_dataset_structure(DATA_DIR, CLASS_NAMES)",
    "df_dataset_counts = pd.DataFrame(dataset_counts)",
    "df_dataset_counts.index.name = 'Class'",
    "df_dataset_counts['Total'] = df_dataset_counts.sum(axis=1)",
    "",
    "print('=== Dataset Sample Counts per Split ===')",
    "print(df_dataset_counts)",
    "total_images = df_dataset_counts['Total'].sum()",
    "print(f'Total Dataset Images: {total_images}')"
])

# CELL 6 — DataLoaders
add_code([
    "print('Initializing PyTorch DataLoaders...')",
    "if DATA_DIR.exists():",
    "    dataloaders = create_dataloaders(",
    "        data_dir=str(DATA_DIR),",
    "        batch_size=BATCH_SIZE,",
    "        num_workers=NUM_WORKERS,",
    "        img_size=IMAGE_SIZE",
    "    )",
    "else:",
    "    print(f'[INFO] Dataset directory \\'{DATA_DIR}\\' not found. Please place or point to your dataset directory when ready to run training.')",
    "    dataloaders = {}",
    "",
    "train_loader = dataloaders.get('train')",
    "val_loader = dataloaders.get('val')",
    "test_loader = dataloaders.get('test')",
    "",
    "print('DataLoader Summary:')",
    "if train_loader:",
    "    print(f'  Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)')",
    "if val_loader:",
    "    print(f'  Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)')",
    "if test_loader:",
    "    print(f'  Test:  {len(test_loader.dataset):,} samples ({len(test_loader)} batches)')"
])

# CELL 7 — Sample Dataset Images
add_code([
    "# Visualizing dataset samples dynamically",
    "if train_loader and len(train_loader.dataset) > 0:",
    "    dataset = train_loader.dataset",
    "    fig, axes = plt.subplots(2, 4, figsize=(14, 7))",
    "    axes = axes.flatten()",
    "    ",
    "    np.random.seed(SEED)",
    "    indices = np.random.choice(len(dataset), size=8, replace=False)",
    "    ",
    "    mean = np.array([0.485, 0.456, 0.406])",
    "    std = np.array([0.229, 0.224, 0.225])",
    "    ",
    "    for idx, ax in zip(indices, axes):",
    "        img_tensor, label_idx = dataset[idx]",
    "        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()",
    "        img_np = std * img_np + mean",
    "        img_np = np.clip(img_np, 0, 1)",
    "        ",
    "        cls_code = CLASS_NAMES[label_idx]",
    "        cls_desc = READABLE_CLASS_NAMES[cls_code]",
    "        ",
    "        ax.imshow(img_np)",
    "        ax.set_title(f'{cls_code}\\n{cls_desc}', fontsize=10)",
    "        ax.axis('off')",
    "        ",
    "    plt.suptitle('Sample Training Images (RDD2020 Dataset)', fontsize=14, fontweight='bold')",
    "    plt.tight_layout()",
    "    plt.show()"
])

# CELL 8 — ResNet50 Model Initialization
add_code([
    "print('Initializing ResNet50 model architecture...')",
    "model = get_resnet50_model(",
    "    num_classes=NUM_CLASSES,",
    "    pretrained=PRETRAINED,",
    "    freeze_backbone=FREEZE_BACKBONE,",
    "    dropout_rate=DROPOUT_RATE",
    ").to(device)",
    "print('ResNet50 model successfully instantiated and moved to device.')"
])

# CELL 9 — Architecture Verification
add_code([
    "# Verification of Model Parameters & Output Layer",
    "total_params = sum(p.numel() for p in model.parameters())",
    "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
    "",
    "print('=== ResNet50 Architecture Verification ===')",
    "print(f'Model Class:             {model.__class__.__name__}')",
    "print(f'Total Parameters:        {total_params:,}')",
    "print(f'Trainable Parameters:    {trainable_params:,}')",
    "print(f'Output FC Layer:         {model.backbone.fc}')",
    "",
    "# Output class verification",
    "dummy_input = torch.randn(2, 3, 224, 224).to(device)",
    "with torch.no_grad():",
    "    dummy_output = model(dummy_input)",
    "",
    "print(f'Model Output Logits Shape: {dummy_output.shape}')",
    "assert dummy_output.shape == (2, NUM_CLASSES), f'Expected shape (2, {NUM_CLASSES}), got {dummy_output.shape}'",
    "print('ResNet50 Architecture adapted successfully for 4 road damage classes.')"
])

# CELL 10 — Loss Function, Optimizer and Scheduler
add_code([
    "# Define Criterion, Optimizer, and Scheduler",
    "criterion = nn.CrossEntropyLoss()",
    "optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)",
    "scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)",
    "",
    "print(f'Criterion: {criterion.__class__.__name__}')",
    "print(f'Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})')",
    "print(f'Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)')"
])

# CELL 11 — Training Loop
add_code([
    "# Controlled Training Execution",
    "history = {",
    "    'epoch': [],",
    "    'train_loss': [],",
    "    'train_acc': [],",
    "    'val_loss': [],",
    "    'val_acc': [],",
    "    'lr': []",
    "}",
    "",
    "if RUN_TRAINING:",
    "    print('Beginning Training Loop...')",
    "    best_loss = float('inf')",
    "    ",
    "    for epoch in range(1, EPOCHS + 1):",
    "        model.train()",
    "        running_loss = 0.0",
    "        correct = 0",
    "        total = 0",
    "        ",
    "        for inputs, targets in train_loader:",
    "            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)",
    "            optimizer.zero_grad()",
    "            outputs = model(inputs)",
    "            loss = criterion(outputs, targets)",
    "            loss.backward()",
    "            optimizer.step()",
    "            ",
    "            running_loss += loss.item() * inputs.size(0)",
    "            _, preds = torch.max(outputs, 1)",
    "            correct += preds.eq(targets).sum().item()",
    "            total += targets.size(0)",
    "            ",
    "        train_loss = running_loss / total",
    "        train_acc = correct / total",
    "        ",
    "        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)",
    "        curr_lr = optimizer.param_groups[0]['lr']",
    "        scheduler.step(val_loss)",
    "        ",
    "        history['epoch'].append(epoch)",
    "        history['train_loss'].append(train_loss)",
    "        history['train_acc'].append(train_acc)",
    "        history['val_loss'].append(val_loss)",
    "        history['val_acc'].append(val_acc)",
    "        history['lr'].append(curr_lr)",
    "        ",
    "        print(f'Epoch [{epoch:02d}/{EPOCHS:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')",
    "else:",
    "    print('RUN_TRAINING is False. Skipping training execution to preserve existing trained model weights.')"
])

# CELL 12 — Save Best Model
add_code([
    "# Model Checkpoint Preservation Strategy",
    "if RUN_TRAINING and SAVE_BEST_CHECKPOINT:",
    "    torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)",
    "    print(f'Checkpoint successfully updated at: {BEST_CHECKPOINT_PATH}')",
    "else:",
    "    print(f'Skipped saving checkpoint. Existing verified checkpoint remains unchanged at:\\n  {BEST_CHECKPOINT_PATH}')"
])

# CELL 13 — Training History
add_code([
    "# Retrieve & Display Training History",
    "if RUN_TRAINING and len(history['epoch']) > 0:",
    "    df_history = pd.DataFrame(history)",
    "elif HISTORY_CSV_PATH.exists():",
    "    df_history = pd.read_csv(HISTORY_CSV_PATH)",
    "    print(f'Loaded existing training history from: {HISTORY_CSV_PATH}')",
    "else:",
    "    df_history = pd.DataFrame()",
    "",
    "if not df_history.empty:",
    "    print('=== Training History Table ===')",
    "    display(df_history) if 'display' in globals() else print(df_history.to_string(index=False))"
])

# CELL 14 — Training and Validation Loss Graph
add_code([
    "# Training and Validation Loss Plot",
    "if not df_history.empty and 'train_loss' in df_history.columns and 'val_loss' in df_history.columns:",
    "    fig, ax = plt.subplots(figsize=(8, 5))",
    "    epochs_seq = df_history['epoch'] if 'epoch' in df_history.columns else range(1, len(df_history) + 1)",
    "    ",
    "    ax.plot(epochs_seq, df_history['train_loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)",
    "    ax.plot(epochs_seq, df_history['val_loss'], 's-', label='Validation Loss', color='#ff7f0e', linewidth=2)",
    "    ",
    "    ax.set_title('ResNet50 - Training and Validation Loss', fontsize=13, fontweight='bold')",
    "    ax.set_xlabel('Epoch', fontsize=11)",
    "    ax.set_ylabel('Loss', fontsize=11)",
    "    ax.set_xticks(epochs_seq)",
    "    ax.legend()",
    "    ax.grid(True, linestyle='--', alpha=0.6)",
    "    ",
    "    loss_path = FIGURES_DIR / 'resnet50_loss_curve.png'",
    "    plt.tight_layout()",
    "    plt.savefig(loss_path, dpi=300)",
    "    plt.show()",
    "    print(f'Loss figure saved to: {loss_path}')"
])

# CELL 15 — Training and Validation Accuracy Graph
add_code([
    "# Training and Validation Accuracy Plot",
    "if not df_history.empty and 'train_acc' in df_history.columns and 'val_acc' in df_history.columns:",
    "    fig, ax = plt.subplots(figsize=(8, 5))",
    "    epochs_seq = df_history['epoch'] if 'epoch' in df_history.columns else range(1, len(df_history) + 1)",
    "    ",
    "    ax.plot(epochs_seq, df_history['train_acc'] * 100, 'o-', label='Train Accuracy', color='#2ca02c', linewidth=2)",
    "    ax.plot(epochs_seq, df_history['val_acc'] * 100, 's-', label='Validation Accuracy', color='#d62728', linewidth=2)",
    "    ",
    "    ax.set_title('ResNet50 - Training and Validation Accuracy', fontsize=13, fontweight='bold')",
    "    ax.set_xlabel('Epoch', fontsize=11)",
    "    ax.set_ylabel('Accuracy (%)', fontsize=11)",
    "    ax.set_xticks(epochs_seq)",
    "    ax.legend()",
    "    ax.grid(True, linestyle='--', alpha=0.6)",
    "    ",
    "    acc_path = FIGURES_DIR / 'resnet50_accuracy_curve.png'",
    "    plt.tight_layout()",
    "    plt.savefig(acc_path, dpi=300)",
    "    plt.show()",
    "    print(f'Accuracy figure saved to: {acc_path}')"
])

# CELL 16 — Load Best ResNet50 Checkpoint
add_code([
    "# Load Best Saved Model Checkpoint",
    "print(f'Loading best checkpoint from: {BEST_CHECKPOINT_PATH}')",
    "assert BEST_CHECKPOINT_PATH.exists(), f'Checkpoint file missing at: {BEST_CHECKPOINT_PATH}'",
    "",
    "checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)",
    "if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:",
    "    state_dict = checkpoint['model_state_dict']",
    "else:",
    "    state_dict = checkpoint",
    "",
    "if hasattr(model, 'backbone'):",
    "    try:",
    "        model.backbone.load_state_dict(state_dict)",
    "    except Exception:",
    "        model.load_state_dict(state_dict)",
    "else:",
    "    model.load_state_dict(state_dict)",
    "",
    "model.eval()",
    "print('Best ResNet50 checkpoint loaded successfully and model set to evaluation mode.')"
])

# CELL 17 — Validation Evaluation
add_code([
    "# Evaluate Loaded Model on Validation Set",
    "print('Running Validation Set Evaluation...')",
    "val_loss, val_acc, val_targets, val_preds = evaluate_model(model, val_loader, criterion, DEVICE)",
    "",
    "val_p_macro, val_r_macro, val_f1_macro, _ = precision_recall_fscore_support(val_targets, val_preds, average='macro', zero_division=0)",
    "val_p_weighted, val_r_weighted, val_f1_weighted, _ = precision_recall_fscore_support(val_targets, val_preds, average='weighted', zero_division=0)",
    "",
    "print('\\n=== Validation Evaluation Summary ===')",
    "print(f'Validation Loss:     {val_loss:.4f}')",
    "print(f'Validation Accuracy: {val_acc * 100:.2f}%')",
    "print(f'Macro F1-Score:      {val_f1_macro * 100:.2f}%')",
    "print(f'Weighted F1-Score:   {val_f1_weighted * 100:.2f}%')"
])

# CELL 18 — Final Held-Out Test Evaluation
add_code([
    "# Final Evaluation on Held-Out Test Set",
    "print('Running Final Held-Out Test Set Evaluation...')",
    "test_loss, test_acc, test_targets, test_preds = evaluate_model(model, test_loader, criterion, DEVICE)",
    "",
    "test_p_macro, test_r_macro, test_f1_macro, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro', zero_division=0)",
    "test_p_weighted, test_r_weighted, test_f1_weighted, _ = precision_recall_fscore_support(test_targets, test_preds, average='weighted', zero_division=0)",
    "",
    "print('\\n==================================================')",
    "print('      RESNET50 FINAL HELD-OUT TEST RESULTS        ')",
    "print('==================================================')",
    "print(f'Test Loss:           {test_loss:.4f}')",
    "print(f'Accuracy:            {test_acc * 100:.2f}%')",
    "print(f'Macro Precision:     {test_p_macro * 100:.2f}%')",
    "print(f'Macro Recall:        {test_r_macro * 100:.2f}%')",
    "print(f'Macro F1:            {test_f1_macro * 100:.2f}%')",
    "print(f'Weighted Precision:  {test_p_weighted * 100:.2f}%')",
    "print(f'Weighted Recall:     {test_r_weighted * 100:.2f}%')",
    "print(f'Weighted F1:         {test_f1_weighted * 100:.2f}%')",
    "print('==================================================')",
    "print(f'Evaluated on {len(test_targets):,} test samples.')"
])

# CELL 19 — Test Classification Report
add_code([
    "# Classification Report Generation",
    "df_report = generate_classification_metrics(test_targets, test_preds, CLASS_NAMES)",
    "",
    "# Map Index Names to Readable Format",
    "row_names = [f'{c} ({READABLE_CLASS_NAMES[c]})' for c in CLASS_NAMES] + ['accuracy', 'macro avg', 'weighted avg']",
    "df_report.index = row_names",
    "",
    "print('=== ResNet50 Test Classification Report ===')",
    "display(df_report) if 'display' in globals() else print(df_report.to_string())"
])

# CELL 20 — Test Confusion Matrix
add_code([
    "# Generate & Save Test Set Confusion Matrix",
    "cm_path = str(FIGURES_DIR / 'resnet50_test_confusion_matrix.png')",
    "fig_cm = plot_and_save_confusion_matrix(",
    "    y_true=test_targets,",
    "    y_pred=test_preds,",
    "    class_names=CLASS_NAMES,",
    "    save_path=cm_path,",
    "    title='ResNet50 Test Set Confusion Matrix'",
    ")",
    "plt.show()"
])

# CELL 21 — Sample Test Predictions
add_code([
    "# Sample Test Predictions Visualization",
    "if test_loader and len(test_loader.dataset) > 0:",
    "    model.eval()",
    "    test_ds = test_loader.dataset",
    "    fig, axes = plt.subplots(2, 4, figsize=(15, 8))",
    "    axes = axes.flatten()",
    "    ",
    "    np.random.seed(SEED)",
    "    sample_indices = np.random.choice(len(test_ds), size=8, replace=False)",
    "    ",
    "    mean = np.array([0.485, 0.456, 0.406])",
    "    std = np.array([0.229, 0.224, 0.225])",
    "    ",
    "    for idx, ax in zip(sample_indices, axes):",
    "        img_tensor, true_idx = test_ds[idx]",
    "        ",
    "        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)",
    "        with torch.no_grad():",
    "            output_logits = model(input_tensor)",
    "            probs = torch.softmax(output_logits, dim=1).cpu().numpy()[0]",
    "            pred_idx = np.argmax(probs)",
    "            confidence = probs[pred_idx] * 100",
    "            ",
    "        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()",
    "        img_np = std * img_np + mean",
    "        img_np = np.clip(img_np, 0, 1)",
    "        ",
    "        true_cls = CLASS_NAMES[true_idx]",
    "        pred_cls = CLASS_NAMES[pred_idx]",
    "        is_correct = (true_idx == pred_idx)",
    "        color = 'green' if is_correct else 'red'",
    "        ",
    "        ax.imshow(img_np)",
    "        ax.set_title(",
    "            f'True: {true_cls} | Pred: {pred_cls}\\n'",
    "            f'Conf: {confidence:.1f}%\\n'",
    "            f'{\"CORRECT\" if is_correct else \"INCORRECT\"}',",
    "            fontsize=9,",
    "            color=color,",
    "            fontweight='bold'",
    "        )",
    "        ax.axis('off')",
    "        ",
    "    plt.suptitle('ResNet50 Sample Test Predictions', fontsize=14, fontweight='bold')",
    "    plt.tight_layout()",
    "    plt.show()"
])

# CELL 22 — Final ResNet50 Summary
add_code([
    "# Final Experiment Summary Report",
    "summary_data = {",
    "    'Metric': [",
    "        'Model Name',",
    "        'Number of Classes',",
    "        'Classes Evaluated',",
    "        'Test Sample Count',",
    "        'Final Test Loss',",
    "        'Final Test Accuracy',",
    "        'Macro Precision',",
    "        'Macro Recall',",
    "        'Macro F1-Score',",
    "        'Weighted F1-Score'",
    "    ],",
    "    'Value': [",
    "        'ResNet50 (Pretrained)',",
    "        f'{NUM_CLASSES}',",
    "        ', '.join(CLASS_NAMES),",
    "        f'{len(test_targets):,}',",
    "        f'{test_loss:.4f}',",
    "        f'{test_acc * 100:.2f}%',",
    "        f'{test_p_macro * 100:.2f}%',",
    "        f'{test_r_macro * 100:.2f}%',",
    "        f'{test_f1_macro * 100:.2f}%',",
    "        f'{test_f1_weighted * 100:.2f}%'",
    "    ]",
    "}",
    "",
    "df_summary = pd.DataFrame(summary_data)",
    "",
    "print('==================================================================')",
    "print('                   FINAL RESNET50 EXPERIMENT SUMMARY              ')",
    "print('==================================================================')",
    "print(df_summary.to_string(index=False))",
    "print('==================================================================')",
    "print('\\nNote: All metric values were computed dynamically on the held-out test split.')"
])

notebook_content = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Successfully generated {notebook_path} with {len(cells)} cells.")
