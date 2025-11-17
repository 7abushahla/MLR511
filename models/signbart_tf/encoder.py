"""
TensorFlow/Keras implementation of SignBART encoder.
Converted from PyTorch implementation.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from attention import SelfAttention
from layers import PositionalEmbedding


@tf.keras.utils.register_keras_serializable()
class EncoderLayer(layers.Layer):
    """
    Single encoder layer with self-attention and feed-forward network.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.d_model = config['d_model']
        self.dropout_rate = config['dropout']
        self.activation_dropout = config['activation_dropout']
        
        # Self-attention
        self.self_attn = SelfAttention(
            d_model=self.d_model,
            num_heads=config['encoder_attention_heads'],
            dropout=config['attention_dropout'],
            name='self_attn'
        )
        self.self_attn_layer_norm = layers.LayerNormalization(epsilon=1e-5, name='self_attn_layer_norm')
        
        # Feed-forward network
        self.fc1 = layers.Dense(config['encoder_ffn_dim'], name='fc1')
        self.fc2 = layers.Dense(self.d_model, name='fc2')
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-5, name='final_layer_norm')
        
        # Dropout layers
        self.dropout = layers.Dropout(self.dropout_rate)
        self.activation_dropout_layer = layers.Dropout(self.activation_dropout)
    
    def call(self, hidden_states, attention_mask=None, training=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, 1, seq_len, seq_len) - additive mask
        
        Returns:
            hidden_states: (batch_size, seq_len, d_model)
        """
        # Self-attention block
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = tf.nn.gelu(hidden_states)
        hidden_states = self.activation_dropout_layer(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Check for inf/nan (for fp16 compatibility)
        # Note: In practice, you'd only use this for mixed precision training
        # hidden_states = tf.debugging.check_numerics(hidden_states, "EncoderLayer output")
        
        return hidden_states
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
            'activation_dropout': self.activation_dropout,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class Encoder(layers.Layer):
    """
    SignBART Encoder with multiple encoder layers.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = config
        self.dropout_rate = config['dropout']
        self.layerdrop = config['encoder_layerdrop']
        
        embed_dim = config['d_model']
        
        # Positional embeddings
        self.embed_positions = PositionalEmbedding(
            config['max_position_embeddings'],
            embed_dim,
            name='embed_positions'
        )
        
        # Stack of encoder layers
        self.encoder_layers = [
            EncoderLayer(config, name=f'encoder_layer_{i}')
            for i in range(config['encoder_layers'])
        ]
        
        # Layer normalization for embeddings
        self.layernorm_embedding = layers.LayerNormalization(epsilon=1e-5, name='layernorm_embedding')
        self.dropout = layers.Dropout(self.dropout_rate)
    
    def call(self, x_embed, attention_mask=None, training=None):
        """
        Args:
            x_embed: (batch_size, seq_len, d_model) - input embeddings
            attention_mask: (batch_size, seq_len) - binary mask (1 = valid, 0 = padding)
        
        Returns:
            hidden_states: (batch_size, seq_len, d_model) - encoder output
        """
        # Add positional embeddings
        pos_embed = self.embed_positions(x_embed, training=training)
        hidden_states = x_embed + pos_embed
        
        # Layer normalization and dropout
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        
        # Create attention mask in the right format
        # Convert from (batch_size, seq_len) to (batch_size, 1, 1, seq_len)
        # Then expand to (batch_size, 1, seq_len, seq_len)
        if attention_mask is not None:
            attention_mask = create_attention_mask(attention_mask, hidden_states.dtype)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            # Note: LayerDrop is typically only used during training
            # For simplicity, we're not implementing it here
            # You could add: if training and tf.random.uniform([]) < self.layerdrop: continue
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )
        
        return hidden_states
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'config_dict': self.config_dict,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Reconstruct layer from config."""
        config_dict = config.pop('config_dict')
        return cls(config_dict, **config)


def create_attention_mask(mask, dtype):
    """
    Create attention mask for encoder.
    
    Args:
        mask: (batch_size, seq_len) - binary mask (1 = valid, 0 = padding)
        dtype: data type for output mask
    
    Returns:
        attention_mask: (batch_size, 1, seq_len, seq_len) - additive mask
    """
    batch_size = tf.shape(mask)[0]
    src_len = tf.shape(mask)[1]
    
    # Expand dimensions: (batch_size, 1, 1, src_len)
    expanded_mask = tf.cast(mask[:, None, None, :], dtype)
    
    # Invert mask (0 = attend, 1 = mask)
    inverted_mask = 1.0 - expanded_mask
    
    # Convert to additive mask (0 = attend, -inf = mask)
    # Use a large negative value instead of actual -inf for numerical stability
    if dtype == tf.float32:
        min_val = -1e9
    elif dtype == tf.float16:
        min_val = -1e4
    else:
        min_val = -1e9
    
    attention_mask = tf.where(
        tf.cast(inverted_mask, tf.bool),
        tf.fill(tf.shape(inverted_mask), min_val),
        tf.zeros_like(inverted_mask)
    )
    
    return attention_mask


if __name__ == "__main__":
    # Test the encoder
    print("Testing TensorFlow SignBART Encoder...")
    
    config = {
        'd_model': 256,
        'encoder_attention_heads': 8,
        'encoder_ffn_dim': 1024,
        'encoder_layers': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.1,
        'encoder_layerdrop': 0.0,
        'max_position_embeddings': 512,
    }
    
    # Test EncoderLayer
    print("\n1. Testing EncoderLayer:")
    encoder_layer = EncoderLayer(config)
    test_input = tf.random.normal((2, 10, 256))
    output = encoder_layer(test_input, training=False)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test Encoder
    print("\n2. Testing Encoder:")
    encoder = Encoder(config)
    test_input = tf.random.normal((2, 10, 256))
    test_mask = tf.ones((2, 10))  # All valid
    output = encoder(test_input, attention_mask=test_mask, training=False)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Mask shape: {test_mask.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test with padding
    print("\n3. Testing Encoder with padding:")
    test_mask = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=tf.float32)
    output = encoder(test_input, attention_mask=test_mask, training=False)
    print(f"   Mask shape: {test_mask.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n✓ All encoder tests passed!")

