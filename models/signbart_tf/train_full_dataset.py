#!/usr/bin/env python
"""
train_full_dataset.py - Train model on full dataset (all 12 users) using Functional API.

This script trains on the complete dataset without LOSO cross-validation.
Data structure: data/arabic-asl-90kpts/all/G01/, data/arabic-asl-90kpts/all/G02/, etc.

Usage:
    python train_full_dataset.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \
        --epochs 80 \
        --lr 2e-4 \
        --seed 42
"""

import sys
import os
import argparse

# Add current directory to path to import main_functional
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_functional import main as main_functional_main
import tensorflow as tf
import keras


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Arabic ASL model on full dataset (all users) using Functional API (QAT-ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Train on full dataset:
  python train_full_dataset.py \\
      --config_path configs/arabic-asl-90kpts.yaml \\
      --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \\
      --epochs 80 \\
      --lr 2e-4 \\
      --seed 42

  # Quick test run:
  python train_full_dataset.py \\
      --config_path configs/arabic-asl-90kpts.yaml \\
      --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \\
      --epochs 2 \\
      --lr 2e-4
        """
    )
    
    parser.add_argument("--config_path", type=str, default="configs/arabic-asl.yaml",
                        help="Path to model config file")
    parser.add_argument("--base_data_path", type=str, required=True,
                        help="Base path to data directory (e.g., ~/signbart_tf/data/arabic-asl-90kpts)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=379,
                        help="Random seed")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained model (.h5 file, optional)")
    parser.add_argument("--skip_final_eval", action="store_true",
                        help="Skip final evaluation (no test set available) - default: True for full dataset")
    parser.add_argument("--exp_name", type=str, default="arabic_asl_full",
                        help="Experiment name (default: arabic_asl_full)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Data path should be the base directory (where label2id.json is)
    # SignDataset will look for {data_path}/all/G01/, etc.
    data_path = args.base_data_path
    
    # Verify data directory exists
    all_dir = os.path.join(data_path, "all")
    if not os.path.exists(all_dir):
        print(f"ERROR: Data directory not found: {all_dir}")
        print(f"Expected structure: {all_dir}/G01/, {all_dir}/G02/, etc.")
        sys.exit(1)
    
    # Check if label files exist (they should be in the base directory)
    label2id_path = os.path.join(data_path, "label2id.json")
    if not os.path.exists(label2id_path):
        print(f"WARNING: label2id.json not found in {data_path}")
        print("The dataset may not load correctly.")
    
    # Print configuration
    print("="*80)
    print("FULL DATASET TRAINING (All 12 Users)")
    print("="*80)
    print(f"Config: {args.config_path}")
    print(f"Data path: {data_path}")
    print(f"Experiment: {args.exp_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print(f"Validation: DISABLED (training on full dataset only)")
    if args.skip_final_eval:
        print(f"Final evaluation: DISABLED")
    print("="*80)
    print()
    
    # Patch prepare_data_loaders to use "all" split instead of "train"
    from main_functional import prepare_data_loaders as original_prepare_data_loaders
    from main_functional import format_batch_for_functional
    from dataset import SignDataset
    
    def prepare_data_loaders_all(data_path, joint_idx, batch_size=1, no_validation=False):
        """Modified prepare_data_loaders to use 'all' split instead of 'train'."""
        train_datasets = SignDataset(data_path, "all", shuffle=True, joint_idxs=joint_idx, augment=True)
        train_loader_raw = train_datasets.create_tf_dataset(batch_size, drop_remainder=False)
        train_loader = train_loader_raw.map(format_batch_for_functional)
        
        # Always return None for validation since we're using full dataset
        return train_loader, None, train_datasets
    
    # Monkey-patch the function
    import main_functional
    main_functional.prepare_data_loaders = prepare_data_loaders_all
    
    # Create a namespace object for main_functional arguments
    from argparse import Namespace
    
    functional_args = Namespace(
        config_path=args.config_path,
        data_path=data_path,
        experiment_name=args.exp_name,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        pretrained_path=args.pretrained_path,
        task="train",  # Required by main_functional
        no_validation=True,  # Always disable validation for full dataset
        skip_final_eval=True,  # Always skip final eval for full dataset (no test set)
        scheduler_factor=0.1,  # Default from main_functional
        scheduler_patience=5,  # Default from main_functional
        save_every=10,  # Default from main_functional
        save_all_checkpoints=False,  # Default from main_functional
        resume_checkpoints=""  # Default from main_functional
    )
    
    # Call main_functional with our arguments
    try:
        main_functional_main(functional_args)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Training failed")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

