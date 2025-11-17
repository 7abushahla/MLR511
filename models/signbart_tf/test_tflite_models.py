#!/usr/bin/env python3
"""
test_tflite_models.py

Evaluate FP32 and dynamic-range INT8 TFLite models on a test split and compare accuracy.

Usage example:
    python test_tflite_models.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --data_path ~/signbart_tf/data/arabic-asl-90kpts_LOSO_user01 \
        --fp32_tflite checkpoints_arabic_asl_LOSO_user01/final_model_fp32.tflite \
        --dynamic_tflite checkpoints_arabic_asl_LOSO_user01/signbart_dynamic_range.tflite
"""
import argparse
import os
import time
import numpy as np
import tensorflow as tf
import yaml
from dataset import SignDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FP32 vs dynamic-range INT8 TFLite models on accuracy."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config YAML (unused but kept for parity)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to processed LOSO data split (e.g., *_LOSO_user01)")
    parser.add_argument("--fp32_tflite", type=str, required=True,
                        help="Path to FP32 TFLite model (final_model_fp32.tflite)")
    parser.add_argument("--dynamic_tflite", type=str, required=True,
                        help="Path to dynamic-range TFLite model")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of test samples to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def determine_keypoint_groups(config_joint_idx):
    """
    Match the grouping logic used during training (body, left hand, right hand, face).
    """
    if not config_joint_idx:
        return None

    sorted_idx = sorted(config_joint_idx)
    total_kpts = len(sorted_idx)
    groups = []

    if total_kpts >= 67:
        face = sorted_idx[-25:]
        right_hand = sorted_idx[-46:-25]
        left_hand = sorted_idx[-67:-46]
        body = sorted_idx[:-67]

        if body:
            groups.append(body)
        if left_hand:
            groups.append(left_hand)
        if right_hand:
            groups.append(right_hand)
        if face:
            groups.append(face)
    else:
        groups.append(sorted_idx)

    return groups


def load_test_dataset(data_path, joint_groups, max_samples=None):
    dataset = SignDataset(
        root=data_path,
        split="test",
        shuffle=False,
        joint_idxs=joint_groups,
        augment=False
    )
    if max_samples is not None:
        dataset.list_key = dataset.list_key[:max_samples]
    return dataset


def preprocess_sample(dataset, file_path, max_seq_len):
    keypoints, label = dataset.load_sample(file_path)
    seq_len = min(keypoints.shape[0], max_seq_len)

    padded_keypoints = np.zeros((max_seq_len, keypoints.shape[1], 2), dtype=np.float32)
    padded_keypoints[:seq_len] = keypoints[:seq_len]

    attention_mask = np.zeros((max_seq_len,), dtype=np.float32)
    attention_mask[:seq_len] = 1.0

    return padded_keypoints, attention_mask, label


def evaluate_tflite(model_path, dataset, max_seq_len):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    keypoints_idx = None
    mask_idx = None
    for idx, detail in enumerate(input_details):
        if len(detail["shape"]) == 4:
            keypoints_idx = idx
        elif len(detail["shape"]) in (2, 3):
            mask_idx = idx

    total = 0
    correct_top1 = 0
    correct_top5 = 0
    start_time = time.time()

    for file_path in dataset.list_key:
        keypoints, mask, label = preprocess_sample(dataset, file_path, max_seq_len)

        interpreter.set_tensor(input_details[keypoints_idx]["index"], keypoints[None, ...])
        interpreter.set_tensor(input_details[mask_idx]["index"], mask[None, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        pred_top1 = np.argmax(output)
        pred_top5 = np.argsort(output)[-5:]

        correct_top1 += int(pred_top1 == label)
        correct_top5 += int(label in pred_top5)
        total += 1

    elapsed = time.time() - start_time
    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total
    size_mb = os.path.getsize(model_path) / (1024 ** 2)
    return acc_top1, acc_top5, elapsed, size_mb


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    joint_groups = determine_keypoint_groups(config.get("joint_idx", []))

    max_seq_len = 64  # fixed for TFLite models
    dataset = load_test_dataset(args.data_path, joint_groups, args.max_samples)
    print(f"[DATA] Loaded {len(dataset.list_key)} test samples from {args.data_path}")

    print("\n[FP32] Evaluating FP32 TFLite model...")
    fp32_top1, fp32_top5, fp32_time, fp32_size = evaluate_tflite(args.fp32_tflite, dataset, max_seq_len)
    print(f"  Top-1 Accuracy: {fp32_top1:.4f} ({fp32_top1*100:.2f}%)")
    print(f"  Top-5 Accuracy: {fp32_top5:.4f} ({fp32_top5*100:.2f}%)")
    print(f"  Inference time (total): {fp32_time:.2f} seconds")
    print(f"  File size: {fp32_size:.2f} MB")

    print("\n[INT8] Evaluating dynamic-range TFLite model...")
    int8_top1, int8_top5, int8_time, int8_size = evaluate_tflite(args.dynamic_tflite, dataset, max_seq_len)
    print(f"  Top-1 Accuracy: {int8_top1:.4f} ({int8_top1*100:.2f}%)")
    print(f"  Top-5 Accuracy: {int8_top5:.4f} ({int8_top5*100:.2f}%)")
    print(f"  Inference time (total): {int8_time:.2f} seconds")
    print(f"  File size: {int8_size:.2f} MB")

    print("\n[SUMMARY]")
    print(f"  FP32 TFLite  : {args.fp32_tflite}")
    print(f"  Dynamic TFLite: {args.dynamic_tflite}")
    print(f"  Δ Top-1: {(int8_top1 - fp32_top1) * 100:.2f} percentage points")
    print(f"  Δ Top-5: {(int8_top5 - fp32_top5) * 100:.2f} percentage points")
    print(f"  Δ File size: {fp32_size - int8_size:+.2f} MB")
    print(f"  Δ Inference time: {int8_time - fp32_time:+.2f} seconds")


if __name__ == "__main__":
    main()


