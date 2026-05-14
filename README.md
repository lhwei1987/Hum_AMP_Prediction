# AMP_Prediction: Multi-Modal CNN for Antimicrobial Peptide Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning framework for high-accuracy antimicrobial peptide (AMP) identification, combining amino acid physicochemical properties with sequence features through a multi-modal CNN. The model integrates multi-scale feature extraction, attention mechanisms, Bayesian uncertainty quantification, and multi-task learning.

## Highlights

- **Multi-modal fusion** — Jointly models 10 physicochemical descriptors per residue alongside learned sequence embeddings
- **Multi-scale feature extraction** — Residual conv blocks with dilated convolutions across 3 kernel sizes
- **Dual attention** — Gated attention (channel-wise weighting) + channel attention (global context)
- **Bayesian uncertainty** — Efficient local reparameterization with MC Dropout for confidence estimation
- **Multi-task learning** — Simultaneously predicts antimicrobial activity, hemolytic toxicity, and protease stability
- **Biophysical constraints** — Integrates charge, hydrophobicity, amphiphilicity, and secondary structure priors into the loss

## Performance

| Metric | Value |
|--------|-------|
| Best F1 Score | **0.9504** |
| AUROC | > 0.98 |
| Successful models | 13 / 35 trials |
| Training set | 30,827 sequences |
| Validation set | 6,606 sequences |
| Test set | 6,606 sequences |

## Project Structure

```
AMP_Prediction/
├── configs/
│   └── default.yaml                 # Training hyperparameters
├── data/
│   └── processed/                   # Preprocessed datasets
│       ├── train.csv                # Training set (30,827 seqs)
│       ├── val.csv                  # Validation set (6,606 seqs)
│       └── test.csv                 # Test set (6,606 seqs)
├── src/                             # Core library
│   ├── data_loader.py              # PyTorch DataLoader wrappers
│   ├── preprocessing.py            # Sequence encoding & feature extraction
│   ├── structure.py                # Secondary structure prediction
│   └── utils.py                    # Metrics, visualization, tokenizer
├── models/                          # Model architecture definitions
│   ├── multimodal_cnn.py           # Main multi-modal CNN with Bayesian heads
│   └── cnn_model.py                # Baseline CNN variant
├── checkpoints/                     # 24 pre-trained model weights (78 MB)
│   ├── best_model_f1_0.9504.pth    # Top-performing model
│   └── best_model_f1_*.pth         # Additional 23 checkpoints (F1 ≥ 0.945)
├── scripts/                         # Executable scripts
│   ├── training.py                 # Single-run training
│   ├── continuous_training.py      # Multi-trial training pipeline
│   ├── evaluate.py                 # Single-model evaluation
│   ├── evaluate_multiple_models.py # Batch evaluation across checkpoints
│   ├── predict_AMP.py              # Inference on new sequences
│   └── combine_code.py             # Code aggregation utility
├── results/
│   └── summary/                    # Best-model training reports
├── docs/                            # Technical documentation (Chinese)
│   ├── 记录.md                     # Condensed training log
│   ├── 特征融合流程汇总图.md        # Architecture design notes
│   └── 投票.md                     # Ensemble voting strategy
├── requirements.txt
└── .gitignore
```

## Quick Start

### Installation

```bash
# Clone and install dependencies
git clone <repo-url>
cd AMP_Prediction
pip install -r requirements.txt
```

### Training

```bash
# Single training run
python scripts/training.py

# Multi-trial continuous training (26-model scheme)
python scripts/continuous_training.py
```

### Evaluation

```bash
# Evaluate all trained checkpoints
python scripts/evaluate_multiple_models.py
```

### Inference

```bash
# Batch prediction — place sequences in prediction_input/test.csv
python scripts/predict_AMP.py
```

### Loading a Pre-Trained Model

```python
import torch
from models.multimodal_cnn import AMPPredictionModel

model = AMPPredictionModel()
model.load_state_dict(torch.load('checkpoints/best_model_f1_0.9504.pth'))
model.eval()

# Predict on a single sequence
sequence = "GLFDIVKKVVGALGSL"
# ... use preprocessing + model forward pass
```

## Data Format

Input data should be CSV files with two columns:

```csv
sequence,label
GLFDIVKKVVGALGSL,1
ARTKQTARKSTGGKAP,0
```

- `sequence` — amino acid sequence (uppercase, single-letter code)
- `label` — `1` for antimicrobial peptide, `0` for non-AMP

## Model Architecture

```
Input: [Sequence Encoding] + [10 Physicochemical Properties]
  ├── Sequence Embedding (Embedding → Conv1D → 48-dim)
  ├── Property Projection (Linear → 32-dim)
  └── Feature Concatenation (80-dim → Conv1D → 128-dim)
           ↓
Multi-Scale Feature Extraction
  ├── Residual Conv Blocks (kernel sizes: 3, 5, 7)
  ├── Dilated Convolutions (expanded receptive field)
  ├── Gated Attention (channel-wise weighting)
  └── Channel Attention (global context modeling)
           ↓
Multi-Task Heads
  ├── Primary: Antimicrobial Activity (with Bayesian Linear layers)
  ├── Auxiliary 1: Hemolytic Toxicity
  └── Auxiliary 2: Protease Stability
```

**Key architectural details:**
- Sequence embedding dim: 48 | Property embedding dim: 32
- Fusion channels: 128 | Dropout rate: 0.1
- Classifier: 576 → 256 → 128 → 64 → 1
- Total parameters: 38,530
- Bayesian layers use local reparameterization with KL annealing (weight: 0.01)
- Biophysical constraint loss (charge, hydrophobicity, amphiphilicity, secondary structure) with adaptive weighting

## Pre-Trained Checkpoints

24 model checkpoints are provided in `checkpoints/`, all achieving F1 ≥ 0.945 on the validation set:

| Rank | F1 Score | Filename |
|------|----------|----------|
| 1 | 0.9504 | `best_model_f1_0.9504.pth` |
| 2 | 0.9488 | `best_model_f1_0.9488.pth` |
| 3 | 0.9479 | `best_model_f1_0.9479.pth` |
| 4 | 0.9476 | `best_model_f1_0.9476.pth` |
| 5 | 0.9473 | `best_model_f1_0.9473.pth` |
| 6 | 0.9470 | `best_model_f1_0.9470.pth` |
| 7 | 0.9468 | `best_model_f1_0.9468.pth` |
| 8 | 0.9467 | `best_model_f1_0.9467.pth` |
| 9 | 0.9465 | `best_model_f1_0.9465.pth` |
| 10 | 0.9463 | `best_model_f1_0.9463.pth` |
| 11–24 | 0.9450–0.9461 | `best_model_f1_0.94*.pth` |

Models were produced by a 35-trial continuous training pipeline (target F1 ≥ 0.945, early stopping patience = 20 epochs). Each checkpoint is ~3.2 MB.

## Citation

If you use this work, please cite:

```bibtex
@software{amp_prediction_2025,
  title = {AMP_Prediction: Multi-Modal CNN for Antimicrobial Peptide Prediction},
  year = {2025},
  note = {26-model continuous training pipeline, best F1 = 0.9504}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
