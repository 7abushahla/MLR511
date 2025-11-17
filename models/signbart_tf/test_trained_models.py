#!/usr/bin/env python3
"""
test_trained_models.py

Utility script to:
1. Load trained SignBART models saved by train_loso_functional.py (.h5 or .keras)
2. Run a quick inference test to ensure weights are valid
3. (Optional) Apply selective QAT annotation to verify the model can be prepared for QAT fine-tuning

Usage example:
    python test_trained_models.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --checkpoint_dir checkpoints_arabic_asl_LOSO_user01 \
        --qat
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load trained SignBART models (.h5/.keras), run inference, and optionally annotate for QAT."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config YAML (e.g., configs/arabic-asl-90kpts.yaml)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing best_model.h5 / final_model.h5 / .keras files")
    parser.add_argument("--qat", action="store_true",
                        help="Annotate loaded model for selective QAT to ensure pipeline works")
    parser.add_argument("--quantize_dense_names", nargs="*", default=None,
                        help="Optional substrings for Dense layer names to quantize (default matches fc/proj/attention)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_dummy_input(config, batch_size=1, seq_len=10):
    num_keypoints = len(config["joint_idx"])
    dummy = {
        "keypoints": tf.random.normal((batch_size, seq_len, num_keypoints, 2)),
        "attention_mask": tf.ones((batch_size, seq_len)),
    }
    return dummy


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


def load_model_from_path(path):
    print(f"\n[LOAD] Attempting to load model from: {path}")
    custom_objects = get_custom_objects()
    with tfmot.quantization.keras.quantize_scope(custom_objects):
        model = keras.models.load_model(path, custom_objects=custom_objects)
    print(f"✓ Model loaded successfully from {path}")
    return model


def test_inference(model, config, label=""):
    dummy = get_dummy_input(config)
    output = model(dummy, training=False)
    print(f"[TEST] {label} inference output shape: {output.shape}")
    try:
        print(f"[SUMMARY] {label}")
        model.summary(line_length=96)
    except Exception as e:
        print(f"[SUMMARY] Unable to display summary for {label}: {e}")


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


CUSTOM_LAYER_TYPES = (
    Projection,
    Encoder,
    Decoder,
    ClassificationHead,
    ExtractLastValidToken,
    PositionalEmbedding,
    EncoderLayer,
    DecoderLayer,
    SelfAttention,
    CrossAttention,
    CausalSelfAttention,
)


def annotate_for_qat(layer, dense_name_filters, dense_log, container_log):
    if isinstance(layer, keras.layers.Dense):
        if any(sub in layer.name for sub in dense_name_filters):
            dense_log.append(layer.name)
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    if isinstance(layer, CUSTOM_LAYER_TYPES):
        container_log.append(layer.name or layer.__class__.__name__)

        def record_internal_denses(obj, parent_name="", visited=None):
            if visited is None:
                visited = set()
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(obj, (list, tuple)):
                for idx, sub in enumerate(obj):
                    record_internal_denses(sub, f"{parent_name}/{idx}" if parent_name else str(idx), visited)
                return
            if isinstance(obj, dict):
                for key, sub in obj.items():
                    record_internal_denses(sub, f"{parent_name}/{key}" if parent_name else str(key), visited)
                return

            if hasattr(obj, "layers") and obj.layers:
                for sublayer in obj.layers:
                    record_internal_denses(
                        sublayer,
                        f"{parent_name}/{sublayer.name}" if parent_name else sublayer.name,
                        visited,
                    )
                return

            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                except Exception:
                    continue

                if isinstance(attr, keras.layers.Dense):
                    dense_log.append(f"{parent_name}/{attr.name}" if parent_name else attr.name)
                elif isinstance(attr, (list, tuple, dict)):
                    record_internal_denses(
                        attr,
                        f"{parent_name}/{attr_name}" if parent_name else attr_name,
                        visited,
                    )
                elif isinstance(attr, keras.layers.Layer) and not isinstance(attr, keras.Model):
                    record_internal_denses(
                        attr,
                        f"{parent_name}/{attr.name}" if parent_name else attr.name,
                        visited,
                    )

        record_internal_denses(layer, layer.name or layer.__class__.__name__)

        return tfmot.quantization.keras.quantize_annotate_layer(
            layer,
            quantize_config=NoOpQuantizeConfig()
        )

    return layer


def build_qat_model(base_model, dense_name_filters):
    dense_log = []
    container_log = []
    custom_objects = get_custom_objects()
    custom_objects["NoOpQuantizeConfig"] = NoOpQuantizeConfig

    with keras.utils.custom_object_scope(custom_objects):
        annotated_model = keras.models.clone_model(
            base_model,
            clone_function=lambda layer: annotate_for_qat(
                layer, dense_name_filters, dense_log, container_log
            )
        )

    with keras.utils.custom_object_scope(custom_objects):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    return qat_model, dense_log, container_log


def run_qat_annotation(model, config, dense_filter_override=None):
    dense_filters = dense_filter_override or [
        "fc1", "fc2",
        "proj_x1", "proj_y1",
        "q_proj", "k_proj", "v_proj", "out_proj",
    ]

    print("[SUMMARY] Model before QAT annotation:")
    try:
        model.summary(line_length=96)
    except Exception as e:
        print(f"  Unable to display summary: {e}")

    qat_model, dense_log, container_log = build_qat_model(model, dense_filters)
    print("\n[QAT] Model annotated successfully.")
    if dense_log:
        print("  Annotated Dense layers:")
        for name in sorted(set(dense_log)):
            print(f"    - {name}")
    else:
        print("  WARNING: No Dense layers matched the filter!")

    if container_log:
        print("  Containers wrapped (NoOp):")
        for name in sorted(set(container_log)):
            print(f"    - {name}")

    # Test inference to ensure annotated model still runs
    print("\n[SUMMARY] QAT model after annotation:")
    try:
        qat_model.summary(line_length=96)
    except Exception as e:
        print(f"  Unable to display QAT summary: {e}")

    dummy = get_dummy_input(config)
    output = qat_model(dummy, training=False)
    print(f"[QAT] Post-annotation inference output shape: {output.shape}")
    return qat_model


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    candidate_files = [
        "best_model.h5",
        "final_model.h5",
        "best_model.keras",
        "final_model.keras",
    ]

    for filename in candidate_files:
        path = checkpoint_dir / filename
        if not path.exists():
            continue

        model = load_model_from_path(path)
        test_inference(model, config, label=filename)

        if args.qat:
            print(f"\n[QAT] Annotating model from {filename}...")
            _ = run_qat_annotation(model, config, dense_filter_override=args.quantize_dense_names)

    print("\nAll model checks complete.")


if __name__ == "__main__":
    main()


