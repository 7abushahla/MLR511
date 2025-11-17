#!/usr/bin/env python
"""
train_loso_functional.py - Train models using Leave-One-Subject-Out cross-validation
using the FUNCTIONAL API model for QAT compatibility.

This is a convenience wrapper around train_loso.py that defaults to using the 
functional model API.

Usage:
    python train_loso_functional.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \
        --holdout_only user01 \
        --epochs 2 \
        --lr 2e-4 \
        --seed 42

This will automatically use the functional model (main_functional.py) which is
compatible with Quantization-Aware Training (QAT).
"""

import sys
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train Arabic ASL models with LOSO cross-validation using Functional API (QAT-ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Test with single user first (RECOMMENDED):
  python train_loso_functional.py \\
      --config_path configs/arabic-asl-90kpts.yaml \\
      --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \\
      --holdout_only user01 \\
      --epochs 2 \\
      --lr 2e-4

  # Run all LOSO experiments:
  python train_loso_functional.py \\
      --config_path configs/arabic-asl-90kpts.yaml \\
      --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \\
      --epochs 80 \\
      --lr 2e-4
        """
    )
    
    parser.add_argument("--config_path", type=str, default="configs/arabic-asl.yaml",
                        help="Path to model config file")
    parser.add_argument("--base_data_path", type=str, default="data/arabic-asl",
                        help="Base path to processed data (without _LOSO suffix)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=379,
                        help="Random seed")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained model (.h5 file, optional)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run evaluation")
    parser.add_argument("--holdout_only", type=str, default="",
                        help="Train only specific holdout user (e.g., 'user01', 'user02') - USE THIS TO TEST FIRST!")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during training (test evaluation will run after)")
    parser.add_argument("--skip_final_eval", action="store_true",
                        help="Skip final evaluation on test set after training")
    parser.add_argument("--exp_prefix", type=str, default="",
                        help="Prefix for experiment names (e.g., 'functional' to distinguish from nested)")
    
    args = parser.parse_args()
    
    # Import the main train_loso module
    # We need to add --use_functional flag
    import subprocess
    
    # Build command to call train_loso.py with --use_functional flag
    cmd = [
        sys.executable,
        "train_loso.py",
        "--config_path", args.config_path,
        "--base_data_path", args.base_data_path,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--use_functional"  # This is the key flag!
    ]
    
    if args.pretrained_path:
        cmd.extend(["--pretrained_path", args.pretrained_path])
    
    if args.skip_training:
        cmd.append("--skip_training")
    
    if args.holdout_only:
        cmd.extend(["--holdout_only", args.holdout_only])
    
    if args.no_validation:
        cmd.append("--no_validation")
    
    if args.skip_final_eval:
        cmd.append("--skip_final_eval")
    
    if args.exp_prefix:
        cmd.extend(["--exp_prefix", args.exp_prefix])
    
    # Print info
    print("="*80)
    print("FUNCTIONAL API TRAINING WRAPPER")
    print("="*80)
    print("This script trains using the Functional API model (QAT-ready)")
    print("Forwarding to train_loso.py with --use_functional flag...")
    print("="*80)
    print()
    
    # Run the command
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

