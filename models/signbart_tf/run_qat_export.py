#!/usr/bin/env python3
"""
run_qat_export.py

Utility script to:
1. Load the functional SignBART model
2. Export an FP32 TFLite model
3. Apply selective Quantization-Aware Training (QAT)
4. Export a quantized (INT8 weights) TFLite model from the QAT model

Usage example:
    python run_qat_export.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --output_dir exports/qat_demo
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import yaml

from model_functional import build_signbart_functional_with_dict_inputs
from model_functional import ExtractLastValidToken
from layers import Projection, ClassificationHead, PositionalEmbedding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import SelfAttention, CrossAttention, CausalSelfAttention
from model_functional_tflite import build_signbart_functional_tflite, ExtractLastValidTokenTFLite
# TFLite sequence length (matches training script)
MAX_SEQ_LEN = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export FP32 and quantized TFLite models with selective QAT."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config YAML (e.g., configs/arabic-asl-90kpts.yaml)")
    parser.add_argument("--output_dir", type=str, default="exports/qat_demo",
                        help="Where to store outputs (TFLite files, checkpoints)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--quantize_dense_names", nargs="*", default=None,
                        help="Optional list of substrings for Dense layer names to quantize "
                             "(defaults to ['fc1','fc2','proj_x1','proj_y1','q_proj','k_proj','v_proj','out_proj']).")
    parser.add_argument("--save_keras", action="store_true",
                        help="Save intermediate Keras models (FP32 + QAT) for inspection.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_functional_model(config):
    model = build_signbart_functional_with_dict_inputs(config)

    # Build model by running dummy input once
    num_keypoints = len(config["joint_idx"])
    dummy = {
        "keypoints": tf.random.normal((1, 10, num_keypoints, 2)),
        "attention_mask": tf.ones((1, 10)),
    }
    _ = model(dummy, training=False)
    return model


def build_tflite_model(config):
    model = build_signbart_functional_tflite(config)
    num_keypoints = len(config["joint_idx"])
    dummy = {
        "keypoints": tf.random.normal((1, 10, num_keypoints, 2)),
        "attention_mask": tf.ones((1, 10)),
    }
    _ = model(dummy, training=False)
    return model


def export_fp32_tflite(model, config, output_path):
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
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"[FP32] Saved TFLite model to {output_path} ({size_mb:.2f} MB)")


def export_dynamic_range_tflite(model, config, output_path):
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
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"[Dynamic-Range] Saved TFLite model to {output_path} ({size_mb:.2f} MB)")


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Pass-through quantize config for container layers."""

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
    ExtractLastValidTokenTFLite,
    PositionalEmbedding,
    EncoderLayer,
    DecoderLayer,
    SelfAttention,
    CrossAttention,
    CausalSelfAttention,
)


def record_internal_dense_layers(layer, parent_name, dense_log, visited):
    """Recursively record dense layers within a container."""
    if layer is None:
        return
    obj_id = id(layer)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # If the object is a list/tuple/dict of layers, iterate through them.
    if isinstance(layer, (list, tuple)):
        for idx, sub in enumerate(layer):
            record_internal_dense_layers(
                sub,
                f"{parent_name}/{idx}" if parent_name else str(idx),
                dense_log,
                visited,
            )
        return
    if isinstance(layer, dict):
        for key, sub in layer.items():
            record_internal_dense_layers(
                sub,
                f"{parent_name}/{key}" if parent_name else str(key),
                dense_log,
                visited,
            )
        return

    # Check keras model containers
    if hasattr(layer, "layers") and layer.layers:
        for sublayer in layer.layers:
            record_internal_dense_layers(
                sublayer,
                f"{parent_name}/{sublayer.name}" if parent_name else sublayer.name,
                dense_log,
                visited,
            )
        return

    # Check attributes for nested layers
    for attr_name in dir(layer):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(layer, attr_name)
            if isinstance(attr, keras.layers.Dense):
                dense_log.append(f"{parent_name}/{attr.name}" if parent_name else attr.name)
            elif isinstance(attr, (list, tuple, dict)):
                record_internal_dense_layers(
                    attr,
                        f"{parent_name}/{attr_name}" if parent_name else attr_name,
                        dense_log,
                        visited,
                )
            elif isinstance(attr, keras.layers.Layer) and not isinstance(attr, keras.Model):
                record_internal_dense_layers(
                    attr,
                        f"{parent_name}/{attr.name}" if parent_name else attr.name,
                        dense_log,
                        visited,
                )
        except Exception:
            continue


def annotate_for_qat(
    layer,
    dense_name_filters,
    dense_log,
    container_log,
    skipped_dense_log,
    container_details,
    base_name="",
    skipped_layers=None,
):
    """
    Annotate Dense layers whose name contains any of the substrings in dense_name_filters.
    """
    if isinstance(layer, keras.layers.Dense):
        if any(sub in layer.name for sub in dense_name_filters):
            dense_log.append(layer.name)
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        else:
            skipped_dense_log.append(layer.name)
            if skipped_layers is not None:
                skipped_layers.append((layer.name, layer.__class__.__name__))
        return layer

    if isinstance(layer, CUSTOM_LAYER_TYPES):
        container_name = base_name or layer.name or layer.__class__.__name__
        container_log.append(container_name)
        wrapped = tfmot.quantization.keras.quantize_annotate_layer(
            layer,
            quantize_config=NoOpQuantizeConfig()
        )
        annotated_internal = []
        skipped_internal = []

        def record_with_tracking(sub_layer, parent, visited, seen_layers):
            obj_id = id(sub_layer)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(sub_layer, (list, tuple)):
                for idx, s in enumerate(sub_layer):
                    record_with_tracking(s, f"{parent}/{idx}" if parent else str(idx), visited, seen_layers)
                return
            if isinstance(sub_layer, dict):
                for key, s in sub_layer.items():
                    record_with_tracking(s, f"{parent}/{key}" if parent else str(key), visited, seen_layers)
                return

            if hasattr(sub_layer, "layers") and sub_layer.layers:
                for s in sub_layer.layers:
                    record_with_tracking(s, f"{parent}/{s.name}" if parent else s.name, visited, seen_layers)
                return

            for attr_name in dir(sub_layer):
                if attr_name.startswith("_"):
                    continue
                if attr_name in seen_layers:
                    continue
                seen_layers.add(attr_name)
                try:
                    attr = getattr(sub_layer, attr_name)
                except Exception:
                    continue

                if isinstance(attr, keras.layers.Dense):
                    full_name = f"{parent}/{attr.name}" if parent else attr.name
                    if any(sub in attr.name for sub in dense_name_filters):
                        dense_log.append(full_name)
                        annotated_internal.append(full_name)
                    else:
                        skipped_log.append(full_name)
                        skipped_internal.append(full_name)
                elif isinstance(attr, (list, tuple, dict)):
                    record_with_tracking(attr, f"{parent}/{attr_name}" if parent else attr_name, visited, set())
                elif isinstance(attr, keras.layers.Layer) and not isinstance(attr, keras.Model):
                    # Record non-dense layers that remain FP32
                    if skipped_layers is not None:
                        skipped_layers.append(
                            (f"{parent}/{attr.name}" if parent else attr.name, attr.__class__.__name__)
                        )
                    record_with_tracking(attr, f"{parent}/{attr.name}" if parent else attr.name, visited, set())

        record_with_tracking(layer, container_name, visited=set(), seen_layers=set())
        container_details[container_name] = {
            "annotated": sorted(set(annotated_internal)),
            "skipped": sorted(set(skipped_internal)),
        }
        return wrapped

    if hasattr(layer, "name"):
        skipped_dense_log.append(layer.name)
        if skipped_layers is not None:
            skipped_layers.append((layer.name, layer.__class__.__name__))
    return layer


def build_qat_model(base_model, dense_name_filters):
    dense_log = []
    skipped_dense_log = []
    container_log = []
    container_details = {}
    skipped_layers = []
    custom_objects = {
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
        "ExtractLastValidTokenTFLite": ExtractLastValidTokenTFLite,
        "NoOpQuantizeConfig": NoOpQuantizeConfig,
    }

    with keras.utils.custom_object_scope(custom_objects):
        annotated_model = keras.models.clone_model(
            base_model,
            clone_function=lambda layer: annotate_for_qat(
                layer,
                dense_name_filters,
                dense_log,
                container_log,
                skipped_dense_log,
                container_details,
                base_name=layer.name,
                skipped_layers=skipped_layers,
            )
        )

    with keras.utils.custom_object_scope(custom_objects):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    return qat_model, dense_log, container_log, skipped_dense_log, container_details, skipped_layers


def representative_dataset_generator(num_keypoints, num_samples=100):
    """
    Generate synthetic calibration data for TFLite conversion.
    """
    def gen():
        for _ in range(num_samples):
            seq_len = MAX_SEQ_LEN
            keypoints = np.random.randn(1, seq_len, num_keypoints, 2).astype(np.float32)
            attention_mask = np.ones((1, seq_len), dtype=np.float32)
            # When converting from concrete functions (no signatures) representative dataset
            # must return a sequence matching the positional inputs.
            yield [keypoints, attention_mask]
    return gen


def export_quantized_tflite(qat_model, config, output_path, dynamic_range=True):
    num_keypoints = len(config["joint_idx"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, MAX_SEQ_LEN, num_keypoints, 2], dtype=tf.float32, name="keypoints"),
        tf.TensorSpec(shape=[1, MAX_SEQ_LEN], dtype=tf.float32, name="attention_mask"),
    ])
    def serving_fn(keypoints, attention_mask):
        return qat_model({"keypoints": keypoints, "attention_mask": attention_mask}, training=False)

    concrete_fn = serving_fn.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    if not dynamic_range:
        converter.representative_dataset = representative_dataset_generator(num_keypoints)
    else:
        print("[INT8] Dynamic-range quantization (weights INT8, activations FP32)")

    tflite_model = converter.convert()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"[INT8] Saved QAT TFLite model to {output_path} ({size_mb:.2f} MB)")


def main():
    args = parse_args()
    set_seed(args.seed)

    dense_filters = args.quantize_dense_names or [
        "fc1", "fc2",          # FFN
        "proj_x1", "proj_y1",  # Projection
        "q_proj", "k_proj", "v_proj", "out_proj",  # Attention projections
    ]

    config = load_config(args.config_path)
    model = build_functional_model(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_keras:
        fp32_path = output_dir / "signbart_fp32.keras"
        model.save(fp32_path)
        print(f"Saved FP32 Keras model to {fp32_path}")

    # Build TFLite-friendly base (used for exports + QAT)
    print("\n[BASE] Building TFLite-friendly model...")
    tflite_base_model = build_tflite_model(config)
    print("✓ TFLite-friendly model ready.")

    fp32_tflite_path = output_dir / "signbart_fp32.tflite"
    export_fp32_tflite(tflite_base_model, config, fp32_tflite_path)

    dynamic_tflite_path = output_dir / "signbart_dynamic_range.tflite"
    export_dynamic_range_tflite(tflite_base_model, config, dynamic_tflite_path)

    print("\n[QAT] Building selective QAT model...")
    qat_model, dense_log, container_log, skipped_dense_log, container_details, skipped_layers = build_qat_model(
        tflite_base_model, dense_filters
    )
    print("✓ QAT model constructed.")

    container_summary = sorted(set(container_log))
    dense_summary = sorted(set(dense_log))

    if dense_summary:
        print("\n[QAT] Dense layers annotated:")
        for name in dense_summary:
            print(f"  • {name}")
    else:
        print("\n[QAT] WARNING: No Dense layers were annotated!")

    # if container_summary:
    #     print("\n[QAT] Container layers wrapped with NoOpQuantizeConfig:")
    #     for name in container_summary:
    #         print(f"  • {name}")
    #         details = container_details.get(name, {})
    #         annotated = details.get("annotated", [])
    #         skipped = details.get("skipped", [])
    #         if annotated:
    #             print("    Annotated:")
    #             for entry in annotated:
    #                 print(f"      - {entry}")
    #         if skipped:
    #             print("    Skipped:")
    #             for entry in skipped:
    #                 print(f"      - {entry}")
    # if skipped_dense_log:
    #     print("\n[QAT] Dense layers NOT quantized (filter mismatch):")
    #     for name in sorted(set(skipped_dense_log)):
    #         print(f"  • {name}")

    if skipped_layers:
        print("\n[QAT] Non-Dense layers kept in FP32:")
        for name, cls in sorted(set(skipped_layers)):
            print(f"  • {name} ({cls})")

    if args.save_keras:
        qat_path = output_dir / "signbart_qat.keras"
        qat_model.save(qat_path)
        print(f"Saved QAT Keras model to {qat_path}")

    qat_tflite_path = output_dir / "signbart_qat_int8.tflite"
    export_quantized_tflite(qat_model, config, qat_tflite_path, dynamic_range=True)

    print("\nAll exports complete!")
    print(f"  FP32 TFLite : {fp32_tflite_path}")
    print(f"  Dynamic TFLite: {dynamic_tflite_path}")
    print(f"  QAT INT8 TFLite: {qat_tflite_path}")


if __name__ == "__main__":
    main()


