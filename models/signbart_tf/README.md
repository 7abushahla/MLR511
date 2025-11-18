# SignBART TensorFlow - Arabic Sign Language Recognition

TensorFlow/Keras implementation of SignBART for Arabic sign language gesture recognition with full quantization support (PTQ and QAT).

## 🎯 Features

- **Functional API Model**: QAT-ready architecture using Keras Functional API
- **LOSO Cross-Validation**: Leave-One-Signer-Out evaluation across 3 users
- **Full Dataset Training**: Train on all 12 users combined
- **Quantization Support**: 
  - Post-Training Quantization (PTQ)
  - Quantization-Aware Training (QAT) with optimized hyperparameters
  - Dynamic-range INT8 quantization (weights INT8, activations FP32)
- **TFLite Export**: Optimized models for mobile/edge deployment
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, FLOPs calculation

---

## 📁 Project Structure

```
signbart_tf/
├── configs/
│   └── arabic-asl-90kpts.yaml       # Model configuration (90 keypoints: body + hands + face)
├── data/
│   ├── arabic-asl-90kpts/           # Full dataset (all users)
│   │   ├── all/                     # All samples for full training
│   │   │   ├── G01/ ... G10/
│   │   ├── label2id.json
│   │   └── id2label.json
│   ├── arabic-asl-90kpts_LOSO_user01/  # LOSO split for user01
│   │   ├── train/                   # Training samples (users 08, 11)
│   │   ├── test/                    # Test samples (user01)
│   │   └── ...
│   └── ...
├── checkpoints_*/                   # Training checkpoints
├── exports/                         # Quantized models
│   ├── ptq_loso/                    # PTQ models (per user)
│   ├── qat_loso/                    # QAT models (per user)
│   ├── ptq_full/                    # PTQ model (full dataset)
│   └── qat_full/                    # QAT model (full dataset)
└── results/                         # Evaluation results
    ├── confusion_matrices/          # Confusion matrix PNGs
    ├── model_info.csv               # Parameters & FLOPs
    ├── summary_table.csv            # Accuracy comparison
    └── per_class_accuracy.csv       # Per-gesture accuracy
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n signbart_tf python=3.10
conda activate signbart_tf
pip install tensorflow tensorflow-model-optimization keras pyyaml numpy matplotlib seaborn
```

### 2. Training Workflows

#### **LOSO Training (Recommended for Research)**

Train on 3 LOSO splits (leave-one-signer-out):

```bash
python train_loso_functional.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --base_data_path data/arabic-asl-90kpts \
    --epochs 80 \
    --lr 2e-4 \
    --seed 379
```

**Output**: 3 FP32 models in `checkpoints_arabic_asl_LOSO_user01/`, `user08/`, `user11/`

---

#### **Full Dataset Training**

Train on all 12 users:

```bash
python train_full_dataset.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --base_data_path data/arabic-asl-90kpts \
    --epochs 80 \
    --lr 2e-4 \
    --seed 42
```

**Output**: `checkpoints_arabic_asl_full/final_model.h5` and `final_model_fp32.tflite`

---

### 3. Quantization

#### **Post-Training Quantization (PTQ)**

Dynamic-range INT8 quantization (weights only):

```bash
# For LOSO models
python ptq_export_batch.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --base_data_path data/arabic-asl-90kpts

# For full dataset model
python ptq_export.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --checkpoint checkpoints_arabic_asl_full/final_model.h5 \
    --output_dir exports/ptq_full
```

---

#### **Quantization-Aware Training (QAT)**

Fine-tune with simulated quantization (better accuracy than PTQ):

```bash
# For LOSO models
python train_loso_functional_qat_batch.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --base_data_path data/arabic-asl-90kpts \
    --batch_size 4 \
    --qat_epochs 10 \
    --lr 5e-5

# For full dataset model
python train_loso_functional_qat.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --data_path data/arabic-asl-90kpts \
    --checkpoint checkpoints_arabic_asl_full/final_model.h5 \
    --output_dir exports/qat_full \
    --batch_size 4 \
    --qat_epochs 10 \
    --lr 5e-5 \
    --no_validation
```

**QAT Configuration**:
- **Learning Rate**: 5e-5 (~4× lower than FP32 training)
- **Batch Size**: 4 (larger than training for stability)
- **Epochs**: 10-20 (short fine-tuning)
- **Quantized Layers**: All Dense layers (FFN, attention projections, projection layers)
- **Excluded**: Projection container (tuple output handling issue)
- **Gradient Clipping**: clipnorm=1.0
- **Early Stopping**: Patience 10 (restores best weights)

---

### 4. Evaluation

#### **Single Model Evaluation**

```bash
python evaluate_tflite_single.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --data_path data/arabic-asl-90kpts_LOSO_user01 \
    --split test \
    --tflite_path checkpoints_arabic_asl_full/final_model_fp32.tflite
```

---

#### **Comprehensive Results Collection**

Generate full report with confusion matrices, FLOPs, and accuracy tables:

```bash
python collect_results.py --run_evaluation
```

**Output**:
- `results/report_YYYYMMDD_HHMMSS.txt` - Full text report
- `results/confusion_matrices/*.png` - 9 confusion matrices (3 users × 3 models)
- `results/model_info.csv` - Parameters, FLOPs
- `results/summary_table.csv` - FP32 vs PTQ vs QAT comparison
- `results/per_class_accuracy.csv` - Per-gesture accuracy

---

## 📊 Model Architecture

```
Input: Keypoints [T, 90, 2]
  ↓
Projection Layer (proj_x1, proj_y1) → [T, d_model=144]
  ↓
Positional Embeddings (learned)
  ↓
Encoder (2 layers, 4 heads, FFN 576)
  ├─ Self-Attention (q_proj, k_proj, v_proj, out_proj)
  ├─ LayerNorm + Residual
  ├─ Feed-Forward (fc1, fc2)
  └─ LayerNorm + Residual
  ↓
Decoder (2 layers, 4 heads, FFN 576)
  ├─ Causal Self-Attention
  ├─ Cross-Attention to Encoder
  ├─ Feed-Forward (fc1, fc2)
  └─ LayerNorm + Residual
  ↓
Extract Last Valid Token
  ↓
Classification Head → [10 classes]
```

**Parameters**: 773,578 total  
**Model Size**: 2.95 MB (FP32), ~0.75 MB (INT8)  
**FLOPs**: Calculated per forward pass  

---

## 🔬 Quantization Details

### What Gets Quantized

✅ **Quantized** (Weights + Activations during training, Weights-only in TFLite):
- FFN Dense layers: `fc1`, `fc2` (in encoder & decoder)
- Attention projections: `q_proj`, `k_proj`, `v_proj`, `out_proj`
- Input projections: `proj_x1`, `proj_y1`
- Classification head: `out_proj`

❌ **Not Quantized**:
- Embeddings (positional)
- Normalization layers (LayerNorm)
- Activation functions (GELU, Softmax)
- Dropout
- Structural operations (residual connections, masking)

🚫 **Excluded from Wrapping** (Critical):
- `Projection` container (causes collapse if wrapped, but internal Dense layers ARE quantized)

### Why Dynamic-Range Quantization?

We use **weights-only INT8 quantization** (dynamic-range) instead of full INT8 because:
- ✅ Significant model size reduction (~75% smaller)
- ✅ Numerically stable (avoids INF/NaN in attention & normalization)
- ✅ No calibration dataset needed
- ❌ Full INT8 (with calibration) caused numerical instability → INF values

---

## 🎓 Key Findings (QAT Optimization)

### Training Stability Issues Solved

**Problem**: Model collapse after 3-4 QAT epochs (accuracy dropped from 95% → 11%)

**Root Cause**: The `Projection` container layer (tuple output) was sensitive to `QuantizeWrapper`, even with `NoOpQuantizeConfig`.

**Solution**: 
1. Exclude `Projection` container from wrapping entirely
2. Still quantize its internal Dense layers (`proj_x1`, `proj_y1`) via filters
3. Use lower LR (5e-5 vs 2e-4 for FP32 training)
4. Increase batch size (4 vs 1 for FP32 training)
5. Add gradient clipping (clipnorm=1.0)
6. Early stopping with best-weight restoration

**Result**: Stable QAT training reaching 95% accuracy ✅

### Attention Layers Are Safe to Quantize

**Myth**: Attention projections are too sensitive for quantization  
**Reality**: `q_proj`, `k_proj`, `v_proj`, `out_proj` can be safely quantized with proper hyperparameters

---

## 📈 Expected Results

### LOSO Cross-Validation (3 users)

| Model Type | Accuracy | Top-5 Acc | Size (MB) | Speedup |
|------------|----------|-----------|-----------|---------|
| FP32       | 94-96%   | 99-100%   | 3.00      | 1.0×    |
| INT8-PTQ   | 93-95%   | 99-100%   | 0.75      | 2-3×    |
| INT8-QAT   | 94-96%   | 99-100%   | 0.75      | 2-3×    |

**QAT advantage**: +1-2% accuracy over PTQ while maintaining same size/speed.

---

## 🛠️ Key Scripts Reference

### Training
- `train_loso_functional.py` - LOSO training (3 users)
- `train_full_dataset.py` - Full dataset training (12 users)
- `main_functional.py` - Core training logic (called by above)

### Quantization
- `ptq_export.py` - PTQ for single model
- `ptq_export_batch.py` - PTQ for all LOSO models
- `train_loso_functional_qat.py` - QAT for single model
- `train_loso_functional_qat_batch.py` - QAT for all LOSO models

### Evaluation
- `evaluate_tflite_single.py` - Evaluate any TFLite model on any dataset
- `collect_results.py` - Comprehensive report generation
- `test_tflite_models.py` - Compare FP32/PTQ/QAT side-by-side

### Utilities
- `dataset.py` - Dataset loading & preprocessing
- `model_functional.py` - Functional API model definition
- `layers.py` - Custom layers (Projection, ClassificationHead, etc.)
- `encoder.py`, `decoder.py`, `attention.py` - Architecture components

---

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: train split not found"

**Cause**: Using LOSO script on full dataset (or vice versa)

**Solution**:
- LOSO: Use `train_loso_functional_qat.py` with `data/arabic-asl-90kpts_LOSO_userXX`
- Full: Use `train_loso_functional_qat.py` with `data/arabic-asl-90kpts` (auto-detects `all` split)

---

### Issue: "Top5Accuracy deserialization error"

**Cause**: Mismatch between saved model config and metric definition

**Solution**: Already fixed in latest code (extracts `k` from kwargs)

---

### Issue: QAT model collapse

**Cause**: One of:
1. Wrapping `Projection` container
2. Learning rate too high
3. Batch size too small

**Solution**: Use provided QAT hyperparameters (lr=5e-5, batch=4)

---

## 📚 Citation

```bibtex
@article{signbart2024,
  title={SignBART: Arabic Sign Language Recognition with Quantization},
  author={Your Name},
  year={2024}
}
```

---

## 📝 License

[Your License Here]

---

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows TensorFlow/Keras best practices
- Quantization changes are tested on LOSO splits
- Documentation is updated

---

## 📧 Contact

[Your Contact Information]

