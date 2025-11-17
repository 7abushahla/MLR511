"""
TensorFlow/Keras implementation of SignBART attention mechanisms.
Converted from PyTorch implementation.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class BaseAttention(layers.Layer):
    """
    Base class for all attention mechanisms.
    """
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.head_dim = d_model // num_heads
        
        if (self.head_dim * num_heads) != self.d_model:
            raise ValueError(
                f"d_model must be divisible by num_heads (got `d_model`: {self.d_model}"
                f" and `num_heads`: {num_heads})."
            )
        
        self.scaling = self.head_dim ** -0.5
        
        # Projection layers
        self.k_proj = layers.Dense(d_model, use_bias=bias, name='k_proj')
        self.v_proj = layers.Dense(d_model, use_bias=bias, name='v_proj')
        self.q_proj = layers.Dense(d_model, use_bias=bias, name='q_proj')
        self.out_proj = layers.Dense(d_model, use_bias=bias, name='out_proj')
        
        self.dropout_layer = layers.Dropout(dropout)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim).
        Transpose to shape: (batch_size, num_heads, seq_len, head_dim)
        """
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Reconstruct layer from config."""
        return cls(
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            dropout=config['dropout_rate'],
            name=config.get('name'),
            trainable=config.get('trainable', True),
            dtype=config.get('dtype')
        )


@tf.keras.utils.register_keras_serializable()
class SelfAttention(BaseAttention):
    """
    Self-attention mechanism for encoder.
    """
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(d_model, num_heads, dropout, bias, **kwargs)
    
    def call(self, hidden_states, attention_mask=None, training=None):
        """
        Args:
            hidden_states: (batch_size, tgt_len, d_model)
            attention_mask: (batch_size, 1, tgt_len, tgt_len) - additive mask
        
        Returns:
            attn_output: (batch_size, tgt_len, d_model)
        """
        batch_size = tf.shape(hidden_states)[0]
        tgt_len = tf.shape(hidden_states)[1]
        
        # Project and scale
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Split heads
        query_states = self.split_heads(query_states, batch_size)
        key_states = self.split_heads(key_states, batch_size)
        value_states = self.split_heads(value_states, batch_size)
        
        # Attention scores: (batch_size, num_heads, tgt_len, tgt_len)
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)
        
        # Apply attention mask (additive)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout_layer(attn_weights, training=training)
        
        # Apply attention to values
        attn_output = tf.matmul(attn_probs, value_states)
        
        # Reshape back: (batch_size, tgt_len, d_model)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (batch_size, tgt_len, self.d_model))
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


@tf.keras.utils.register_keras_serializable()
class CrossAttention(BaseAttention):
    """
    Cross-attention mechanism for decoder (attending to encoder outputs).
    """
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(d_model, num_heads, dropout, bias, **kwargs)
    
    def call(self, hidden_states, key_value_states, attention_mask=None, training=None):
        """
        Args:
            hidden_states: (batch_size, tgt_len, d_model) - decoder hidden states
            key_value_states: (batch_size, src_len, d_model) - encoder hidden states
            attention_mask: (batch_size, 1, tgt_len, src_len) - additive mask
        
        Returns:
            attn_output: (batch_size, tgt_len, d_model)
        """
        batch_size = tf.shape(hidden_states)[0]
        tgt_len = tf.shape(hidden_states)[1]
        src_len = tf.shape(key_value_states)[1]
        
        # Query from decoder, keys and values from encoder
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(key_value_states)
        value_states = self.v_proj(key_value_states)
        
        # Split heads
        query_states = self.split_heads(query_states, batch_size)
        
        # For key/value, reshape with src_len
        key_states = tf.reshape(key_states, (batch_size, src_len, self.num_heads, self.head_dim))
        key_states = tf.transpose(key_states, perm=[0, 2, 1, 3])
        
        value_states = tf.reshape(value_states, (batch_size, src_len, self.num_heads, self.head_dim))
        value_states = tf.transpose(value_states, perm=[0, 2, 1, 3])
        
        # Attention scores: (batch_size, num_heads, tgt_len, src_len)
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)
        
        # Apply attention mask (additive)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout_layer(attn_weights, training=training)
        
        # Apply attention to values
        attn_output = tf.matmul(attn_probs, value_states)
        
        # Reshape back: (batch_size, tgt_len, d_model)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (batch_size, tgt_len, self.d_model))
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


@tf.keras.utils.register_keras_serializable()
class CausalSelfAttention(BaseAttention):
    """
    Causal self-attention mechanism for decoder (with future masking).
    """
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(d_model, num_heads, dropout, bias, **kwargs)
    
    def call(self, hidden_states, attention_mask=None, training=None):
        """
        Args:
            hidden_states: (batch_size, tgt_len, d_model)
            attention_mask: (batch_size, 1, tgt_len, tgt_len) - additive mask
        
        Returns:
            attn_output: (batch_size, tgt_len, d_model)
        """
        batch_size = tf.shape(hidden_states)[0]
        tgt_len = tf.shape(hidden_states)[1]
        
        # Project and scale
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Split heads
        query_states = self.split_heads(query_states, batch_size)
        key_states = self.split_heads(key_states, batch_size)
        value_states = self.split_heads(value_states, batch_size)
        
        # Attention scores: (batch_size, num_heads, tgt_len, tgt_len)
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)
        
        # Create causal mask (lower triangular)
        causal_mask = tf.linalg.band_part(tf.ones((tgt_len, tgt_len)), -1, 0)
        causal_mask = tf.reshape(causal_mask, (1, 1, tgt_len, tgt_len))
        
        # Apply causal mask (set future positions to -inf)
        attn_weights = tf.where(
            causal_mask == 0,
            tf.fill(tf.shape(attn_weights), float('-inf')),
            attn_weights
        )
        
        # Apply additional attention mask (additive)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout_layer(attn_weights, training=training)
        
        # Apply attention to values
        attn_output = tf.matmul(attn_probs, value_states)
        
        # Reshape back: (batch_size, tgt_len, d_model)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (batch_size, tgt_len, self.d_model))
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


if __name__ == "__main__":
    # Test the attention mechanisms
    print("Testing TensorFlow SignBART Attention Mechanisms...")
    
    # Test SelfAttention
    print("\n1. Testing SelfAttention:")
    self_attn = SelfAttention(d_model=256, num_heads=8, dropout=0.1)
    test_input = tf.random.normal((2, 10, 256))  # (batch, seq_len, d_model)
    output = self_attn(test_input, training=False)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test CrossAttention
    print("\n2. Testing CrossAttention:")
    cross_attn = CrossAttention(d_model=256, num_heads=8, dropout=0.1)
    decoder_input = tf.random.normal((2, 5, 256))  # (batch, tgt_len, d_model)
    encoder_output = tf.random.normal((2, 10, 256))  # (batch, src_len, d_model)
    output = cross_attn(decoder_input, encoder_output, training=False)
    print(f"   Decoder input shape: {decoder_input.shape}")
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test CausalSelfAttention
    print("\n3. Testing CausalSelfAttention:")
    causal_attn = CausalSelfAttention(d_model=256, num_heads=8, dropout=0.1)
    test_input = tf.random.normal((2, 10, 256))  # (batch, seq_len, d_model)
    output = causal_attn(test_input, training=False)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n✓ All attention tests passed!")

