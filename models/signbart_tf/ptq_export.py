#!/usr/bin/env python3
"""
ptq_export.py

Post-training dynamic-range quantization for trained SignBART models.
Loads a trained .h5/.keras checkpoint, converts to TFLite FP32 and dynamic INT8 (weights only),
and saves both artifacts for deployment.

Example:
    python ptq_export.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --checkpoint checkpoints_arabic_asl_LOSO_user01/final_model.h5 \
        --output_dir exports/ptq_arabic_asl_user01
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml

from model_functional import build_signbart_functional_with_dict_inputs, ExtractLastValidToken
from layers import Projection, ClassificationHead, PositionalEmbedding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import SelfAttention, CrossAttention, CausalSelfAttention


@tf.keras.utils.register_keras_serializable()
class Top5Accuracy(keras.metrics.Metric):
    def __init__(self, name="Top5Accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.top5_correct = self.add_weight(name="top5_correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        top5_preds = tf.nn.top_k(y_pred, k=5).indices
        y_true_expanded = tf.expand_dims(tf.cast(y_true, tf.int32), axis=1)
        top5_preds = tf.cast(top5_preds, tf.int32)
        correct = tf.reduce_any(tf.equal(top5_preds, y_true_expanded), axis=1)
        self.top5_correct.assign_add(tf.reduce_sum(tf.cast(correct, tf.float32)))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.top5_correct / self.total

    def reset_state(self):
        self.top5_correct.assign(0.0)
        self.total.assign(0.0)


MAX_SEQ_LEN = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load trained SignBART model and export FP32 + dynamic-range INT8 TFLite."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model (.h5 or .keras).")
    parser.add_argument("--output_dir", type=str, default="exports/ptq_export",
                        help="Output directory for TFLite models.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_custom_objects():
    return {
        "Projection": Projection,
        "ClassificationHead": ClassificationHead,
        "PositionalEmbedding": PositionalEmbedding,
        "Encoder": Encoder,
        "EncoderLayer": EncoderLayer,
        "Decoder": Decoder,
        "DecoderLayer": DecoderLayer,
        "SelfAttention": SelfAttention,
        "CrossAttention": CrossAttention,
        "CausalSelfAttention": CausalSelfAttention,
        "ExtractLastValidToken": ExtractLastValidToken,
        "Top5Accuracy": Top5Accuracy,
    }


def load_trained_model(checkpoint_path):
    print(f"[LOAD] Loading trained model from {checkpoint_path}")
    custom_objects = get_custom_objects()
    model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    print("✓ Trained model loaded.")
    return model


def export_tflite(model, config, output_path, dynamic_range=False):
    num_keypoints = len(config["joint_idx"])

    @tf.function(input_signature=[
        {
            "keypoints": tf.TensorSpec(shape=[1, MAX_SEQ_LEN, num_keypoints, 2], dtype=tf.float32),
            "attention_mask": tf.TensorSpec(shape=[1, MAX_SEQ_LEN], dtype=tf.float32),
        }
    ])
    def serving_fn(inputs):
        return model(inputs, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

    if dynamic_range:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[INT8] Dynamic-range quantization enabled (weights INT8, activations FP32).")

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"✓ Saved TFLite model to {output_path} ({size_mb:.2f} MB)")


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config_path)
    trained_model = load_trained_model(args.checkpoint)

    num_keypoints = len(config["joint_idx"])
    dummy = {
        "keypoints": tf.random.normal((1, 10, num_keypoints, 2)),
        "attention_mask": tf.ones((1, 10)),
    }
    _ = trained_model(dummy, training=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = output_dir / "model_fp32.tflite"
    export_tflite(trained_model, config, fp32_path, dynamic_range=False)

    dynamic_path = output_dir / "model_dynamic_int8.tflite"
    export_tflite(trained_model, config, dynamic_path, dynamic_range=True)

    print("\nExport complete!")
    print(f"  FP32   : {fp32_path}")
    print(f"  INT8DR : {dynamic_path}")


if __name__ == "__main__":
    main()


