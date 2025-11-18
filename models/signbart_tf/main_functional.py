"""
Main training script for SignBART TensorFlow - Functional API Version.
Uses the functional model API for QAT compatibility.
"""
import os
import argparse
import random
import logging
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from pathlib import Path
from datetime import datetime

from dataset import SignDataset, create_data_loaders
from model_functional import build_signbart_functional_with_dict_inputs
from utils import (
    accuracy, top_k_accuracy, save_checkpoint, load_checkpoint,
    count_model_parameters, get_keypoint_config
)

# TFLite fixed sequence length (based on dataset analysis)
# 99th percentile = 61 frames, rounded to power of 2 = 64
# This ensures consistent input/output signatures across all LOSO models
MAX_SEQ_LEN = 64


def get_default_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--config_path", type=str, default="",
                        help="Path to the config model to be used")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained weights (.h5 file)")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--task", type=str, default=False, choices=["train", "eval"],
                        help="Whether to train or evaluate the model")

    parser.add_argument("--data_path", type=str, default="",
                        help="Path to the training dataset")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the model training")

    parser.add_argument("--resume_checkpoints", type=str, default="",
                        help="Path to the checkpoints to be used for resuming training")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")
    
    # Validation control
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during training (training only)")
    
    # Checkpoint control
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--save_all_checkpoints", action="store_true",
                        help="Save checkpoint at every epoch (disk intensive)")

    return parser


def setup_logging(experiment_name):
    """Setup logging configuration."""
    log_file = experiment_name + ".log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set to: {seed}")


def load_model(config_path, pretrained_path):
    """Load functional model and config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {config_path}")
    print(f"  Config keys: {list(config.keys())}")
    
    # Use functional API model instead of nested model
    print(f"\nBuilding FUNCTIONAL API model for QAT compatibility...")
    print(f"  Keypoints: {len(config['joint_idx'])}")
    model = build_signbart_functional_with_dict_inputs(config)
    print("✓ Functional model built successfully")
    
    # The functional model is already built, just test it
    print(f"Testing functional model with dummy input...")
    dummy_data = {
        'keypoints': tf.random.normal((1, 10, len(config['joint_idx']), 2)),
        'attention_mask': tf.ones((1, 10))
    }
    dummy_output = model(dummy_data, training=False)
    print(f"✓ Model test successful! Output shape: {dummy_output.shape}")
    
    if pretrained_path:
        print(f"Loading pretrained weights from: {pretrained_path}")
        try:
            # Load weights
            model.load_weights(pretrained_path)
            print("✓ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model, config


def format_batch_for_functional(batch):
    inputs = {
        'keypoints': batch['keypoints'],
        'attention_mask': batch['attention_mask']
    }
    labels = batch['labels']
    return inputs, labels


def prepare_data_loaders(data_path, joint_idx, batch_size=1, no_validation=False):
    """Prepare training and validation data loaders."""
    train_datasets = SignDataset(data_path, "train", shuffle=True, joint_idxs=joint_idx, augment=True)
    train_loader_raw = train_datasets.create_tf_dataset(batch_size, drop_remainder=False)

    train_loader = train_loader_raw.map(format_batch_for_functional)

    if no_validation:
        return train_loader, None, train_datasets

    val_datasets = SignDataset(data_path, "test", shuffle=False, joint_idxs=joint_idx, augment=False)
    val_loader_raw = val_datasets.create_tf_dataset(batch_size, drop_remainder=False)
    val_loader = val_loader_raw.map(format_batch_for_functional)

    return train_loader, val_loader, train_datasets


def determine_keypoint_groups(config_joint_idx):
    """
    Determine how to group keypoints for normalization.
    Returns a list of lists, where each inner list is a group to normalize together.
    
    Automatically detects groups based on sequential keypoint indices.
    Assumes structure: Pose + Left Hand (21) + Right Hand (21) + Face (remaining)
    """
    if not config_joint_idx:
        return []
    
    # Sort indices to ensure they're in order
    sorted_idx = sorted(config_joint_idx)
    
    # Hand landmarks are always 21 points each (MediaPipe standard)
    # We'll detect groups by finding where hands start (gaps or pattern)
    groups = []
    current_group = []
    
    # Strategy: Detect gaps in sequence or use known hand/face sizes
    # For MediaPipe: Pose (variable) + Hand1 (21) + Hand2 (21) + Face (25)
    
    # Find the first hand (21 consecutive points) and second hand (21 consecutive points)
    # Remaining at end is face (25 points)
    
    total_kpts = len(sorted_idx)
    
    # Known structure: last 25 are face, previous 21 are right hand, previous 21 are left hand
    if total_kpts >= 67:  # At least some pose + 2 hands + face (21+21+25=67)
        # Last 25: face
        face_kpts = sorted_idx[-25:]
        # Previous 21: right hand  
        right_hand_kpts = sorted_idx[-46:-25]
        # Previous 21: left hand
        left_hand_kpts = sorted_idx[-67:-46]
        # Everything else: pose/body
        body_kpts = sorted_idx[:-67]
        
        # Add non-empty groups
        if body_kpts:
            groups.append(body_kpts)
        if left_hand_kpts:
            groups.append(left_hand_kpts)
        if right_hand_kpts:
            groups.append(right_hand_kpts)
        if face_kpts:
            groups.append(face_kpts)
    else:
        # Fallback: just use all as one group
        groups.append(sorted_idx)
    
    return groups


def main(args):
    """Main training function."""
    set_random_seed(args.seed)
    logger = setup_logging(args.experiment_name)
    
    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device = f"GPU: {gpus}"
        except RuntimeError as e:
            logger.error(e)
            device = "CPU"
    else:
        device = "CPU"
    
    print(f"\n{'='*80}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Model Type: FUNCTIONAL API (QAT-ready)")
    print(f"Device: {device}")
    if args.no_validation:
        print(f"Validation: DISABLED (training only)")
    print(f"{'='*80}\n")
    
    # Load model and config
    model, config = load_model(args.config_path, args.pretrained_path)
    
    # Get joint indices from config
    if 'joint_idx' in config and config['joint_idx']:
        config_joint_idx = config['joint_idx']
        print(f"Using joint indices from config: {len(config_joint_idx)} keypoints")
        
        joint_idx = determine_keypoint_groups(config_joint_idx)
        
        # Report keypoint groups
        # Auto-detect labels based on group structure and size
        group_names = []
        for i, group in enumerate(joint_idx):
            group_size = len(group)
            # Standard MediaPipe structure: Pose + Left Hand (21) + Right Hand (21) + Face (25)
            if i == 0:
                # First group is always body/pose (variable size: 23, 33, etc.)
                name = f"body pose (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            elif i == 1 and group_size == 21:
                # Second group of 21 points is left hand
                name = f"left hand (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            elif i == 2 and group_size == 21:
                # Third group of 21 points is right hand
                name = f"right hand (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            elif i == 3 and group_size == 25:
                # Fourth group of 25 points is face
                name = f"face (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            else:
                # Fallback for unexpected structure
                name = f"group {i+1} (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            group_names.append(name)
        
        print(f"Keypoint groups for normalization ({len(joint_idx)} groups):")
        for i, name in enumerate(group_names):
            print(f"  Group {i+1}: {name}")
    else:
        from utils import hands_only_groups
        print("Using default joint indices (hands only)")
        joint_idx = hands_only_groups
    
    checkpoint_dir = "checkpoints_" + args.experiment_name
    
    # Prepare data
    batch_size = config.get('batch_size', 1)
    train_loader, val_loader, train_datasets = prepare_data_loaders(
        args.data_path, joint_idx, batch_size, args.no_validation
    )

    # Report number of training samples
    try:
        num_train_samples = len(train_datasets)
        print(f"Training samples: {num_train_samples}")
    except Exception:
        pass

    # Evaluation loader:
    # - Prefer validation loader when available
    # - Otherwise, try to create test loader
    # - If test split is missing (e.g., full-dataset training), skip evaluation gracefully
    eval_loader = val_loader
    if eval_loader is None:
        try:
            eval_dataset = SignDataset(
                args.data_path,
                "test",
                shuffle=False,
                joint_idxs=joint_idx,
                augment=False
            )
            eval_loader_raw = eval_dataset.create_tf_dataset(batch_size, drop_remainder=False)
            # If dataset creation returns None or empty, skip evaluation
            if eval_loader_raw is not None:
                eval_loader = eval_loader_raw.map(format_batch_for_functional)
            else:
                eval_loader = None
        except FileNotFoundError:
            # Common in full-dataset training where only an "all" split exists
            print("  Note: Test split not found – skipping final evaluation.")
            eval_loader = None
    
    # Setup optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=config.get('weight_decay', 0.01))
    
    # Compile model (Keras standard workflow)
    print(f"\n{'='*80}")
    print(f"COMPILING MODEL")
    print(f"{'='*80}")
    
    # Custom metrics - use TensorFlow's built-in top-k accuracy
    @tf.keras.utils.register_keras_serializable()
    class Top5Accuracy(keras.metrics.SparseTopKCategoricalAccuracy):
        """Top-5 accuracy metric using Keras built-in."""
        def __init__(self, name='top5_accuracy', **kwargs):
            super().__init__(k=5, name=name, **kwargs)
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            Top5Accuracy(name='top5_accuracy')
        ]
    )
    print("✓ Model compiled with AdamW optimizer and SparseCategoricalCrossentropy loss")
    
    # Show model summary (Keras standard)
    print(f"\n{'='*80}")
    print(f"MODEL SUMMARY")
    print(f"{'='*80}")
    try:
        model.summary(line_length=100)
    except Exception as e:
        print(f"Note: Could not display full summary: {e}")
        # Fallback to manual parameter count
        param_counts = count_model_parameters(model)
        total_params = param_counts['total']
        trainable_params = param_counts['trainable']
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print(f"{'='*80}\n")
    
    # Count parameters for logging
    param_counts = count_model_parameters(model)
    total_params = param_counts['total']
    trainable_params = param_counts['trainable']
    logger.info(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Report configuration
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Optimizer: AdamW (weight_decay={config.get('weight_decay', 0.01)})")
    print(f"  Loss: SparseCategoricalCrossentropy (from_logits=True)")
    print(f"  Scheduler: ReduceLROnPlateau (factor={args.scheduler_factor}, patience={args.scheduler_patience})")
    print(f"  Seed: {args.seed}")
    print(f"  Validation: {'DISABLED' if args.no_validation else 'ENABLED'}")
    print(f"  Checkpoint strategy: Save every {args.save_every} epochs + best + latest + final")
    if args.save_all_checkpoints:
        print(f"  WARNING: Saving ALL epoch checkpoints (disk intensive!)")
    
    print(f"\nModel Architecture (from {args.config_path}):")
    print(f"  d_model: {config.get('d_model', 'N/A')}")
    print(f"  Encoder layers: {config.get('encoder_layers', 'N/A')}")
    print(f"  Decoder layers: {config.get('decoder_layers', 'N/A')}")
    print(f"  Attention heads (encoder): {config.get('encoder_attention_heads', 'N/A')}")
    print(f"  Attention heads (decoder): {config.get('decoder_attention_heads', 'N/A')}")
    print(f"  FFN dim (encoder): {config.get('encoder_ffn_dim', 'N/A')}")
    print(f"  FFN dim (decoder): {config.get('decoder_ffn_dim', 'N/A')}")
    print(f"  Dropout: {config.get('dropout', 'N/A')}")
    print(f"  Number of classes: {config.get('num_labels', 'N/A')}")
    print(f"  Max position embeddings: {config.get('max_position_embeddings', 'N/A')}")
    
    num_joints = len(config.get('joint_idx', []))
    if num_joints > 0:
        has_pose = any(idx < 33 for idx in config.get('joint_idx', []))
        has_left_hand = any(33 <= idx < 54 for idx in config.get('joint_idx', []))
        has_right_hand = any(54 <= idx < 75 for idx in config.get('joint_idx', []))
        has_face = any(idx >= 75 for idx in config.get('joint_idx', []))
        
        keypoint_types = []
        if has_pose:
            keypoint_types.append("body pose")
        if has_left_hand:
            keypoint_types.append("left hand")
        if has_right_hand:
            keypoint_types.append("right hand")
        if has_face:
            keypoint_types.append("face")
        
        desc = " + ".join(keypoint_types)
        print(f"  Joint indices: {num_joints} keypoints ({desc})")
    print()
    
    logger.info(f"Configuration: {config}")
    
    # Setup Keras callbacks
    callbacks = []
    
    # ModelCheckpoint for best model (use HDF5 format for compatibility)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path,
        monitor='val_loss' if not args.no_validation else 'loss',
        save_best_only=True,
        save_weights_only=False,  # Save full model in HDF5 format
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # ReduceLROnPlateau
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss' if not args.no_validation else 'loss',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=1,
        min_lr=1e-7
    )
    callbacks.append(reduce_lr_callback)
    
    # CSVLogger for training history
    csv_logger = keras.callbacks.CSVLogger(
        f'{args.experiment_name}_training.csv',
        append=True
    )
    callbacks.append(csv_logger)
    
    # TensorBoard (optional)
    tensorboard_dir = os.path.join("logs", args.experiment_name)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=0,
        write_graph=False
    )
    callbacks.append(tensorboard_callback)
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("out-imgs/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output images directory: out-imgs/{args.experiment_name}/\n")
    
    # Evaluation mode
    if args.task == "eval":
        if eval_loader is None:
            print("ERROR: Cannot evaluate without evaluation data.")
            return
        
        # Load checkpoint for evaluation
        if args.resume_checkpoints:
            checkpoint_to_load = args.resume_checkpoints
            
            # If it's a directory, look for best_model.keras
            if os.path.isdir(checkpoint_to_load):
                best_model_path = os.path.join(checkpoint_to_load, "best_model.h5")
                final_model_path = os.path.join(checkpoint_to_load, "final_model.h5")
                
                if os.path.exists(best_model_path):
                    checkpoint_to_load = best_model_path
                elif os.path.exists(final_model_path):
                    checkpoint_to_load = final_model_path
                else:
                    print(f"\n⚠ Warning: No checkpoint found in {args.resume_checkpoints}")
                    print("Expected: best_model.h5 or final_model.h5")
                    print("Using untrained model (this will give poor results!)")
            
            if os.path.exists(checkpoint_to_load):
                print(f"\nLoading checkpoint from: {checkpoint_to_load}")
                try:
                    # For functional models, we can load the full model
                    loaded_model = keras.models.load_model(checkpoint_to_load)
                    model.set_weights(loaded_model.get_weights())
                    print("✓ Model loaded successfully")
                    logger.info(f"Loaded checkpoint from {checkpoint_to_load}")
                except Exception as e:
                    print(f"✗ Error loading checkpoint: {e}")
                    print("Using untrained model (this will give poor results!)")
        else:
            print("\n⚠ Warning: No checkpoint specified (--resume_checkpoints)")
            print("Evaluating untrained model (this will give poor results!)")
        
        print("\nEvaluate model..!")
        start_time = time.time()
        results = model.evaluate(eval_loader, return_dict=True, verbose=1)
        inference_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  Top-1 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
        print(f"  Inference time: {inference_time:.2f} seconds")
        print(f"{'='*80}\n")
        
        logger.info(f"Evaluation - Loss: {results['loss']:.4f}, Acc: {results['accuracy']:.4f}, Top-5: {results['top5_accuracy']:.4f}")
        return
    
    # Train using Keras model.fit() - The standard Keras way!
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Use model.fit() - standard Keras training
    history = model.fit(
        train_loader,
        validation_data=val_loader if not args.no_validation else None,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1  # Standard Keras progress bar output
    )
    
    total_train_time = time.time() - start_time
    
    # Save final checkpoint (which contains best weights)
    print("\nSaving final checkpoint (contains best weights)...")
    if os.path.exists(best_checkpoint_path):
        print(f"Best model already saved at: {best_checkpoint_path}")
        # Copy it to final_model.h5 for consistency
        import shutil
        final_path = os.path.join(checkpoint_dir, "final_model.h5")
        shutil.copy2(best_checkpoint_path, final_path)
        print(f"Copied to final checkpoint: {final_path}")
        logger.info(f"Final checkpoint saved: {final_path}")
    else:
        print(f"Warning: Best model file not found at {best_checkpoint_path}")
        print("Saving current model as final.")
        final_path = os.path.join(checkpoint_dir, "final_model.h5")
        model.save(final_path, save_format='h5')
        print(f"Final checkpoint saved: {final_path}")
        logger.info(f"Final checkpoint saved: {final_path}")
    
    print(f"\nNote: The final model contains the BEST weights (not the last epoch's weights)")
    print(f"  Best model from: {best_checkpoint_path}")
    print(f"  Saved full model to: {final_path}")
    
    # Convert to TFLite for deployment
    print(f"\n{'='*80}")
    print("CONVERTING TO TFLITE FOR DEPLOYMENT")
    print(f"{'='*80}")
    
    try:
        import tensorflow.lite as tflite
        
        # Get number of keypoints from config
        num_keypoints = len(config['joint_idx'])
        
        # Create a concrete function with FIXED input shape for TFLite
        print(f"\nUsing fixed sequence length for TFLite: {MAX_SEQ_LEN} frames")
        print(f"  (Covers 99%+ of dataset, max was 75 frames)")
        
        @tf.function(input_signature=[
            {
                'keypoints': tf.TensorSpec(shape=[1, MAX_SEQ_LEN, num_keypoints, 2], dtype=tf.float32),
                'attention_mask': tf.TensorSpec(shape=[1, MAX_SEQ_LEN], dtype=tf.float32)
            }
        ])
        def model_predict(inputs):
            return model(inputs, training=False)
        
        print("\n1. Converting to FP32 TFLite...")
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [model_predict.get_concrete_function()]
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(checkpoint_dir, "final_model_fp32.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        tflite_size_mb = os.path.getsize(tflite_path) / (1024**2)
        print(f"✓ TFLite FP32 model saved: {tflite_path}")
        print(f"  File size: {tflite_size_mb:.2f} MB")
        
        # Evaluate both Keras and TFLite models on full test/validation set (if available)
        if eval_loader is not None:
            try:
                print("\n2. Evaluating Keras model on test set...")
                keras_start = time.time()
                keras_results = model.evaluate(eval_loader, return_dict=True, verbose=1)
                keras_time = time.time() - keras_start
                
                print(f"\n  Keras Model Results:")
                print(f"    Loss: {keras_results['loss']:.4f}")
                print(f"    Top-1 Accuracy: {keras_results['accuracy']:.4f} ({keras_results['accuracy']*100:.2f}%)")
                print(f"    Top-5 Accuracy: {keras_results['top5_accuracy']:.4f} ({keras_results['top5_accuracy']*100:.2f}%)")
                print(f"    Inference time: {keras_time:.2f} seconds")
                
                logger.info(
                    f"Keras eval - Loss: {keras_results['loss']:.4f}, "
                    f"Acc: {keras_results['accuracy']:.4f}, "
                    f"Top5: {keras_results['top5_accuracy']:.4f}"
                )
            except ValueError as eval_err:
                # Handle shape/empty-data issues gracefully, especially when no real test set exists
                print(f"  Warning: Skipping Keras evaluation due to error: {eval_err}")
                logger.warning(f"Keras evaluation skipped due to error: {eval_err}")
        else:
            print("  Note: No validation/test data available for TFLite evaluation")
        
        print(f"\n{'='*80}")
        print("✓ TFLite FP32 conversion complete!")
        print(f"{'='*80}")
        print(f"Deployment-ready models:")
        print(f"  1. Keras model: {final_path} ({os.path.getsize(final_path)/(1024**2):.2f} MB)")
        print(f"  2. FP32 TFLite: {tflite_path} ({tflite_size_mb:.2f} MB)")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n✗ TFLite conversion failed: {e}")
        print("Note: You can still use the .keras model for inference")
        import traceback
        traceback.print_exc()
        logger.warning(f"TFLite conversion failed: {e}")
    
    # Final summary
    avg_epoch_time = total_train_time / args.epochs
    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0.0
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total training time: {total_train_time/3600:.2f} hours ({total_train_time:.1f} seconds)")
    print(f"Average time per epoch: {avg_epoch_time:.1f} seconds")
    print(f"Best train accuracy: {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
    if not args.no_validation:
        print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # List saved checkpoints
    print(f"\nSaved checkpoints in {checkpoint_dir}:")
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(('.keras', '.h5'))])
        for ckpt in checkpoints:
            size_mb = os.path.getsize(os.path.join(checkpoint_dir, ckpt)) / (1024**2)
            print(f"  - {ckpt} ({size_mb:.2f} MB)")
    
    print(f"{'='*80}\n")
    
    logger.info(f"Training complete - Total time: {total_train_time:.1f}s, Best train acc: {best_train_acc:.4f}")
    if not args.no_validation:
        logger.info(f"Best val acc: {best_val_acc:.4f}")
    
    # Plot training curves using history from model.fit()
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    ax.plot(epochs_range, history.history['loss'], c="#D64436", label="Training loss", linestyle='--', linewidth=2)
    ax.plot(epochs_range, history.history['accuracy'], c="#00B09B", label="Training accuracy", linewidth=2)
    
    if not args.no_validation and 'val_accuracy' in history.history:
        ax.plot(epochs_range, history.history['val_loss'], c="#FF6B6B", label="Validation loss", linestyle='--', linewidth=2)
        ax.plot(epochs_range, history.history['val_accuracy'], c="#E0A938", label="Validation accuracy", linewidth=2)
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set(xlabel="Epoch", ylabel="Metrics", title=f"Training Curves - {args.experiment_name}")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plot_path = "out-imgs/" + args.experiment_name + "_training_curves.png"
    fig.savefig(plot_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    print(f"\nPlot saved to {plot_path}")
    logger.info("Experiment finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.experiment_name:
        print("ERROR: --experiment_name is required")
        exit(1)
    if not args.config_path:
        print("ERROR: --config_path is required")
        exit(1)
    if not args.data_path:
        print("ERROR: --data_path is required")
        exit(1)
    if not args.task:
        print("ERROR: --task is required (train or eval)")
        exit(1)
    
    main(args)

