#!/usr/bin/env python3
"""
train_loso_functional_qat.py

QAT fine-tuning pipeline:
1. Load trained Functional SignBART model (.h5/.keras) for a LOSO split
2. Annotate selectively for QAT (Dense layers only)
3. Fine-tune for N epochs on the same LOSO train split
4. Evaluate on test split and export dynamic-range INT8 TFLite

Example:
    python train_loso_functional_qat.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --data_path ~/signbart_tf/data/arabic-asl-90kpts_LOSO_user01 \
        --checkpoint checkpoints_arabic_asl_LOSO_user01/final_model.h5 \
        --qat_epochs 3 \
        --output_dir exports/qat_finetune_user01
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import yaml

from dataset import SignDataset
from model_functional import build_signbart_functional_with_dict_inputs, ExtractLastValidToken
from model_functional_tflite import build_signbart_functional_tflite, ExtractLastValidTokenTFLite
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from layers import Projection, ClassificationHead, PositionalEmbedding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import SelfAttention, CrossAttention, CausalSelfAttention

MAX_SEQ_LEN = 64


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
        description="Fine-tune trained SignBART model with QAT and export TFLite."
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to LOSO dataset (e.g., .../arabic-asl-90kpts_LOSO_user01)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained Functional model (.h5/.keras)")
    parser.add_argument("--output_dir", type=str, default="exports/qat_finetune")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--qat_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize_dense_names", nargs="*", default=None,
                        help="Dense layer name substrings to quantize (default targets fc/proj/attention)")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def determine_keypoint_groups(config_joint_idx):
    if not config_joint_idx:
        return None
    sorted_idx = sorted(config_joint_idx)
    total = len(sorted_idx)
    groups = []
    if total >= 67:
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


def create_dataset(root, joint_groups, batch_size, split, augment):
    dataset = SignDataset(
        root=root,
        split=split,
        shuffle=(split == "train"),
        joint_idxs=joint_groups,
        augment=augment
    )
    ds = dataset.create_tf_dataset(batch_size=batch_size, drop_remainder=False)

    def split_batch(batch):
        inputs = {
            "keypoints": batch["keypoints"],
            "attention_mask": batch["attention_mask"],
        }
        labels = batch["labels"]
        return inputs, labels

    ds = ds.map(split_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


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
        "ExtractLastValidTokenTFLite": ExtractLastValidTokenTFLite,
        "Top5Accuracy": Top5Accuracy,
    }


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


def record_internal_dense_layers(layer, parent_name, dense_filters, dense_log, visited):
    if layer is None:
        return
    obj_id = id(layer)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(layer, (list, tuple)):
        for idx, sub in enumerate(layer):
            record_internal_dense_layers(sub, f"{parent_name}/{idx}" if parent_name else str(idx),
                                         dense_filters, dense_log, visited)
        return
    if isinstance(layer, dict):
        for key, sub in layer.items():
            record_internal_dense_layers(sub, f"{parent_name}/{key}" if parent_name else str(key),
                                         dense_filters, dense_log, visited)
        return

    if hasattr(layer, "layers") and layer.layers:
        for sublayer in layer.layers:
            record_internal_dense_layers(
                sublayer,
                f"{parent_name}/{sublayer.name}" if parent_name else sublayer.name,
                dense_filters,
                dense_log,
                visited,
            )
        return

    for attr_name in dir(layer):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(layer, attr_name)
        except Exception:
            continue

        if isinstance(attr, keras.layers.Dense):
            name = f"{parent_name}/{attr.name}" if parent_name else attr.name
            if any(f in attr.name for f in dense_filters):
                dense_log.append(name)
        elif isinstance(attr, (list, tuple, dict)):
            record_internal_dense_layers(
                attr, f"{parent_name}/{attr_name}" if parent_name else attr_name,
                dense_filters, dense_log, visited,
            )
        elif isinstance(attr, keras.layers.Layer) and not isinstance(attr, keras.Model):
            record_internal_dense_layers(
                attr, f"{parent_name}/{attr.name}" if parent_name else attr.name,
                dense_filters, dense_log, visited,
            )


def annotate_dense_layers(layer, dense_filters, dense_log, container_log):
    if isinstance(layer, keras.layers.Dense):
        if any(name in layer.name for name in dense_filters):
            dense_log.append(layer.name)
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    if isinstance(layer, CUSTOM_LAYER_TYPES):
        container_name = layer.name or layer.__class__.__name__
        container_log.append(container_name)
        wrapped = tfmot.quantization.keras.quantize_annotate_layer(
            layer,
            quantize_config=NoOpQuantizeConfig(),
        )
        record_internal_dense_layers(
            layer,
            container_name,
            dense_filters,
            dense_log,
            visited=set()
        )
        return wrapped
    return layer


def build_qat_model(base_model, dense_filters):
    dense_log = []
    container_log = []
    custom_objects = get_custom_objects()
    custom_objects["NoOpQuantizeConfig"] = NoOpQuantizeConfig

    with keras.utils.custom_object_scope(custom_objects):
        annotated = keras.models.clone_model(
            base_model,
            clone_function=lambda layer: annotate_dense_layers(layer, dense_filters, dense_log, container_log)
        )
    with keras.utils.custom_object_scope(custom_objects):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated)
    return qat_model, dense_log, container_log


def copy_weights(source, target):
    source_layers = {layer.name: layer for layer in source.layers}
    for layer in target.layers:
        name = layer.name
        src = source_layers.get(name) or source_layers.get(f"quant_{name}")
        if src is None:
            continue
        if isinstance(src, QuantizeWrapper):
            weights = src.layer.get_weights()
        else:
            weights = src.get_weights()
        if weights:
            layer.set_weights(weights)


def export_dynamic_tflite(model, config, output_path):
    print("[INT8] Building TFLite-friendly model for export...")
    dummy = {
        "keypoints": tf.random.normal((1, 10, len(config["joint_idx"]), 2)),
        "attention_mask": tf.ones((1, 10)),
    }

    float_model = build_signbart_functional_with_dict_inputs(config)
    _ = float_model(dummy, training=False)
    copy_weights(model, float_model)

    tflite_model = build_signbart_functional_tflite(config)
    _ = tflite_model(dummy, training=False)
    copy_weights(float_model, tflite_model)
    print("✓ Weights copied to TFLite-friendly graph.")

    num_keypoints = len(config["joint_idx"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, MAX_SEQ_LEN, num_keypoints, 2], dtype=tf.float32, name="keypoints"),
        tf.TensorSpec(shape=[1, MAX_SEQ_LEN], dtype=tf.float32, name="attention_mask"),
    ])
    def serving_fn(keypoints, attention_mask):
        return tflite_model({"keypoints": keypoints, "attention_mask": attention_mask}, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    tflite_bytes = converter.convert()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_bytes)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"✓ Saved dynamic-range TFLite to {output_path} ({size_mb:.2f} MB)")


def main():
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.config_path)
    joint_groups = determine_keypoint_groups(config.get("joint_idx", []))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[DATA] Preparing datasets...")
    train_ds = create_dataset(args.data_path, joint_groups, args.batch_size, "train", augment=True)
    test_ds = create_dataset(args.data_path, joint_groups, args.batch_size, "test", augment=False)

    print(f"[LOAD] Loading base model from {args.checkpoint}")
    custom_objects = get_custom_objects()
    base_model = keras.models.load_model(args.checkpoint, custom_objects=custom_objects)
    print("✓ Base model loaded.")

    dense_filters = args.quantize_dense_names or [
        "fc1", "fc2",
        "proj_x1", "proj_y1",
        "q_proj", "k_proj", "v_proj", "out_proj",
    ]
    print("[SUMMARY] Base model before QAT annotation:")
    try:
        base_model.summary(line_length=96)
    except Exception as e:
        print(f"  Unable to display summary: {e}")

    print("[QAT] Annotating model...")
    qat_model, dense_log, container_log = build_qat_model(base_model, dense_filters)
    if dense_log:
        print(f"  Annotated Dense layers: {len(set(dense_log))}")
    else:
        print("  WARNING: No Dense layers matched filters!")
    if container_log:
        print("  Containers wrapped:")
        for name in sorted(set(container_log)):
            print(f"    - {name}")

    print("\n[SUMMARY] QAT model after annotation:")
    try:
        qat_model.summary(line_length=96)
    except Exception as e:
        print(f"  Unable to display QAT summary: {e}")

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    qat_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            Top5Accuracy(name="top5_accuracy"),
        ]
    )

    print("\n[TRAIN] Starting QAT fine-tuning...")
    history = qat_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.qat_epochs,
        verbose=1
    )

    print("\n[EVAL] Evaluating QAT model on test set...")
    results = qat_model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    qat_path = output_dir / "qat_model.keras"
    qat_model.save(qat_path)
    print(f"✓ Saved QAT Keras model to {qat_path}")

    tflite_path = output_dir / "qat_dynamic_int8.tflite"
    export_dynamic_tflite(qat_model, config, tflite_path)

    print("\n[SUMMARY]")
    print(f"  Checkpoint used : {args.checkpoint}")
    print(f"  QAT epochs      : {args.qat_epochs}")
    print(f"  QAT model       : {qat_path}")
    print(f"  Dynamic TFLite  : {tflite_path}")


if __name__ == "__main__":
    main()


