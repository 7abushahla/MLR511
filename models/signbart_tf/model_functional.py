"""
Functional API implementation of SignBART for QAT compatibility.
This version can be used with tensorflow_model_optimization.quantization.keras.quantize_model()
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers import PositionalEmbedding, ClassificationHead, Projection
from encoder import Encoder
from decoder import Decoder


@tf.keras.utils.register_keras_serializable()
class ExtractLastValidToken(layers.Layer):
    """Extract last valid token based on attention mask."""
    
    def call(self, inputs):
        decoder_outputs, attention_mask = inputs
        # Find last valid position for each sequence
        # Sum mask to get sequence length, then subtract 1 for 0-indexing
        seq_lengths = tf.cast(tf.reduce_sum(attention_mask, axis=1) - 1, tf.int32)
        seq_lengths = tf.maximum(seq_lengths, 0)  # Ensure non-negative
        
        batch_size = tf.shape(decoder_outputs)[0]
        batch_indices = tf.range(batch_size)
        
        # Gather last valid token
        gather_indices = tf.stack([batch_indices, seq_lengths], axis=1)
        last_output = tf.gather_nd(decoder_outputs, gather_indices)
        
        return last_output
    
    def get_config(self):
        """Return config for serialization."""
        return super().get_config()


def build_signbart_functional(config):
    """
    Build SignBART model using Functional API for QAT compatibility.
    
    Args:
        config: dict with model configuration
        
    Returns:
        keras.Model: Functional API model
    """
    # Define inputs
    keypoints_input = layers.Input(
        shape=(None, len(config['joint_idx']), 2),
        dtype=tf.float32,
        name='keypoints'
    )
    attention_mask_input = layers.Input(
        shape=(None,),
        dtype=tf.float32,
        name='attention_mask'
    )
    
    # Projection
    projection = Projection(config, name='projection')
    x_embed, y_embed = projection(keypoints_input)
    
    # Encoder
    encoder = Encoder(config, name='encoder')
    encoder_outputs = encoder(
        x_embed=x_embed,
        attention_mask=attention_mask_input,
        training=None  # Will be set by model.fit()
    )
    
    # Decoder
    decoder = Decoder(config, name='decoder')
    decoder_outputs = decoder(
        encoder_hidden_states=encoder_outputs,
        encoder_attention_mask=attention_mask_input,
        y_embed=y_embed,
        attention_mask=attention_mask_input,
        training=None
    )
    
    # Get last valid token from decoder outputs
    # We need to extract the last valid position for each sequence
    # This is a bit tricky in Functional API since we can't use Python logic
    
    # Simple approach: use the last timestep (will work for fixed-length sequences)
    # For variable length, we'd need a custom layer
    last_output = decoder_outputs[:, -1, :]  # Take last timestep
    
    # Classification head
    classification_head = ClassificationHead(
        config['d_model'],
        config['num_labels'],
        config['classifier_dropout'],
        name='classification_head'
    )
    logits = classification_head(last_output)
    
    # Build model
    model = keras.Model(
        inputs=[keypoints_input, attention_mask_input],
        outputs=logits,
        name='signbart_functional'
    )
    
    return model


def build_signbart_functional_with_dict_inputs(config):
    """
    Build SignBART model using Functional API that accepts dict inputs.
    This matches the signature of the subclassed model.
    
    Args:
        config: dict with model configuration
        
    Returns:
        keras.Model: Functional API model with dict inputs
    """
    # Define inputs as a dict
    inputs = {
        'keypoints': layers.Input(
            shape=(None, len(config['joint_idx']), 2),
            dtype=tf.float32,
            name='keypoints'
        ),
        'attention_mask': layers.Input(
            shape=(None,),
            dtype=tf.float32,
            name='attention_mask'
        )
    }
    
    keypoints_input = inputs['keypoints']
    attention_mask_input = inputs['attention_mask']
    
    # Projection
    projection = Projection(config, name='projection')
    x_embed, y_embed = projection(keypoints_input)
    
    # Encoder
    encoder = Encoder(config, name='encoder')
    encoder_outputs = encoder(
        x_embed=x_embed,
        attention_mask=attention_mask_input,
        training=None
    )
    
    # Decoder
    decoder = Decoder(config, name='decoder')
    decoder_outputs = decoder(
        encoder_hidden_states=encoder_outputs,
        encoder_attention_mask=attention_mask_input,
        y_embed=y_embed,
        attention_mask=attention_mask_input,
        training=None
    )
    
    # Extract last valid token
    extract_layer = ExtractLastValidToken(name='extract_last_token')
    last_output = extract_layer([decoder_outputs, attention_mask_input])
    
    # Classification head
    classification_head = ClassificationHead(
        config['d_model'],
        config['num_labels'],
        config['classifier_dropout'],
        name='classification_head'
    )
    logits = classification_head(last_output)
    
    # Build model with dict inputs
    model = keras.Model(
        inputs=inputs,
        outputs=logits,
        name='signbart_functional'
    )
    
    return model


if __name__ == "__main__":
    import yaml
    
    # Test the functional model
    print("Testing Functional API SignBART model...")
    
    # Load config
    with open("configs/arabic-asl.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    print("\nBuilding Functional API model...")
    model = build_signbart_functional_with_dict_inputs(config)
    
    # Show summary
    print("\nModel Summary:")
    model.summary(line_length=120)
    
    # Test forward pass
    print("\nTesting forward pass...")
    import numpy as np
    test_data = {
        'keypoints': np.random.randn(2, 10, len(config['joint_idx']), 2).astype(np.float32),
        'attention_mask': np.ones((2, 10), dtype=np.float32)
    }
    
    output = model(test_data, training=False)
    print(f"Output shape: {output.shape}")
    print("✓ Functional API model works!")
    
    # Test QAT compatibility
    print("\nTesting QAT compatibility...")
    try:
        import tensorflow_model_optimization as tfmot
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print("✓ QAT conversion successful!")
        
        qat_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ QAT model compiled!")
        
    except Exception as e:
        print(f"✗ QAT failed: {e}")

