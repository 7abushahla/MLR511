"""
TensorFlow/Keras implementation of SignBART decoder.
Converted from PyTorch implementation.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from attention import CrossAttention, CausalSelfAttention
from layers import PositionalEmbedding


@tf.keras.utils.register_keras_serializable()
class DecoderLayer(layers.Layer):
    """
    Single decoder layer with causal self-attention, cross-attention, and feed-forward network.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.d_model = config['d_model']
        self.dropout_rate = config['dropout']
        self.activation_dropout = config['activation_dropout']
        
        # Causal self-attention
        self.self_attn = CausalSelfAttention(
            d_model=self.d_model,
            num_heads=config['decoder_attention_heads'],
            dropout=config['attention_dropout'],
            name='self_attn'
        )
        self.self_attn_layer_norm = layers.LayerNormalization(epsilon=1e-5, name='self_attn_layer_norm')
        
        # Cross-attention to encoder
        self.encoder_attn = CrossAttention(
            d_model=self.d_model,
            num_heads=config['decoder_attention_heads'],
            dropout=config['attention_dropout'],
            name='encoder_attn'
        )
        self.encoder_attn_layer_norm = layers.LayerNormalization(epsilon=1e-5, name='encoder_attn_layer_norm')
        
        # Feed-forward network
        self.fc1 = layers.Dense(config['decoder_ffn_dim'], name='fc1')
        self.fc2 = layers.Dense(self.d_model, name='fc2')
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-5, name='final_layer_norm')
        
        # Dropout layers
        self.dropout = layers.Dropout(self.dropout_rate)
        self.activation_dropout_layer = layers.Dropout(self.activation_dropout)
    
    def call(self, hidden_states, attention_mask, encoder_hidden_states, 
             encoder_attention_mask, training=None):
        """
        Args:
            hidden_states: (batch_size, tgt_len, d_model) - decoder inputs
            attention_mask: (batch_size, 1, tgt_len, tgt_len) - causal mask for self-attention
            encoder_hidden_states: (batch_size, src_len, d_model) - encoder outputs
            encoder_attention_mask: (batch_size, 1, tgt_len, src_len) - mask for cross-attention
        
        Returns:
            hidden_states: (batch_size, tgt_len, d_model)
        """
        # Self-attention block (causal)
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Cross-attention block (to encoder)
        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = tf.nn.gelu(hidden_states)
        hidden_states = self.activation_dropout_layer(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
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
class Decoder(layers.Layer):
    """
    SignBART Decoder with multiple decoder layers.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = config
        self.dropout_rate = config['dropout']
        self.layerdrop = config['decoder_layerdrop']
        
        embed_dim = config['d_model']
        
        # Positional embeddings
        self.embed_positions = PositionalEmbedding(
            config['max_position_embeddings'],
            embed_dim,
            name='embed_positions'
        )
        
        # Stack of decoder layers
        self.decoder_layers = [
            DecoderLayer(config, name=f'decoder_layer_{i}')
            for i in range(config['decoder_layers'])
        ]
        
        # Layer normalization for embeddings
        self.layernorm_embedding = layers.LayerNormalization(epsilon=1e-5, name='layernorm_embedding')
        self.dropout = layers.Dropout(self.dropout_rate)
    
    def call(self, encoder_hidden_states, encoder_attention_mask, y_embed, 
             attention_mask, training=None):
        """
        Args:
            encoder_hidden_states: (batch_size, src_len, d_model) - encoder outputs
            encoder_attention_mask: (batch_size, src_len) - binary mask for encoder
            y_embed: (batch_size, tgt_len, d_model) - decoder input embeddings
            attention_mask: (batch_size, tgt_len) - binary mask for decoder inputs
        
        Returns:
            hidden_states: (batch_size, tgt_len, d_model) - decoder output
        """
        batch_size = tf.shape(y_embed)[0]
        tgt_len = tf.shape(y_embed)[1]
        
        # Create causal attention mask for decoder self-attention
        decoder_attention_mask = create_causal_attention_mask(
            attention_mask, 
            (batch_size, tgt_len),
            y_embed
        )
        
        # Create encoder attention mask for cross-attention
        encoder_attn_mask = create_encoder_attention_mask(
            encoder_attention_mask,
            y_embed.dtype,
            tgt_len=tgt_len
        )
        
        # Add positional embeddings
        pos_embed = self.embed_positions(y_embed, training=training)
        hidden_states = y_embed + pos_embed
        
        # Layer normalization and dropout
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attn_mask,
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


def create_causal_attention_mask(attention_mask, input_shape, inputs_embeds):
    """
    Create causal attention mask for decoder self-attention.
    Combines padding mask with causal mask.
    
    Args:
        attention_mask: (batch_size, query_length) - binary mask (1 = valid, 0 = padding)
        input_shape: tuple (batch_size, query_length)
        inputs_embeds: tensor with dtype information
    
    Returns:
        attention_mask: (batch_size, 1, query_length, query_length) - additive causal mask
    """
    batch_size, query_length = input_shape
    dtype = inputs_embeds.dtype
    
    # Expand padding mask: (batch_size, 1, 1, query_length)
    expanded_mask = tf.cast(attention_mask[:, None, None, :], dtype)
    
    # Expand to (batch_size, 1, query_length, query_length)
    expanded_mask = tf.tile(expanded_mask, [1, 1, query_length, 1])
    
    # Invert mask (0 = attend, 1 = mask)
    inverted_mask = 1.0 - expanded_mask
    
    # Convert to additive mask
    if dtype == tf.float32:
        min_val = -1e9
    elif dtype == tf.float16:
        min_val = -1e4
    else:
        min_val = -1e9
    
    expanded_mask = tf.where(
        tf.cast(inverted_mask, tf.bool),
        tf.fill(tf.shape(inverted_mask), min_val),
        tf.zeros_like(inverted_mask)
    )
    
    # Create causal mask (lower triangular)
    causal_mask = tf.linalg.band_part(
        tf.ones((query_length, query_length), dtype=dtype), 
        -1, 0
    )
    causal_mask = tf.reshape(causal_mask, (1, 1, query_length, query_length))
    
    # Combine: add causal structure (note: causal_mask has 1s for valid positions)
    # We need to convert causal_mask to additive format too
    causal_additive = tf.where(
        causal_mask > 0,
        tf.zeros_like(causal_mask),
        tf.fill(tf.shape(causal_mask), min_val)
    )
    
    # Combine both masks
    expanded_mask = expanded_mask + causal_additive
    
    return expanded_mask


def create_encoder_attention_mask(encoder_mask, dtype, tgt_len):
    """
    Create attention mask for decoder-to-encoder cross-attention.
    
    Args:
        encoder_mask: (batch_size, src_len) - binary mask (1 = valid, 0 = padding)
        dtype: data type for output mask
        tgt_len: target sequence length (decoder length)
    
    Returns:
        attention_mask: (batch_size, 1, tgt_len, src_len) - additive mask
    """
    batch_size = tf.shape(encoder_mask)[0]
    src_len = tf.shape(encoder_mask)[1]
    
    # Expand dimensions: (batch_size, 1, 1, src_len)
    expanded_mask = tf.cast(encoder_mask[:, None, None, :], dtype)
    
    # Expand to (batch_size, 1, tgt_len, src_len)
    expanded_mask = tf.tile(expanded_mask, [1, 1, tgt_len, 1])
    
    # Invert mask (0 = attend, 1 = mask)
    inverted_mask = 1.0 - expanded_mask
    
    # Convert to additive mask
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
    # Test the decoder
    print("Testing TensorFlow SignBART Decoder...")
    
    config = {
        'd_model': 256,
        'decoder_attention_heads': 8,
        'decoder_ffn_dim': 1024,
        'decoder_layers': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.1,
        'decoder_layerdrop': 0.0,
        'max_position_embeddings': 512,
    }
    
    # Test DecoderLayer
    print("\n1. Testing DecoderLayer:")
    decoder_layer = DecoderLayer(config)
    decoder_input = tf.random.normal((2, 5, 256))
    encoder_output = tf.random.normal((2, 10, 256))
    
    # Create dummy masks
    self_attn_mask = tf.zeros((2, 1, 5, 5))
    cross_attn_mask = tf.zeros((2, 1, 5, 10))
    
    output = decoder_layer(
        decoder_input, 
        self_attn_mask, 
        encoder_output, 
        cross_attn_mask,
        training=False
    )
    print(f"   Decoder input shape: {decoder_input.shape}")
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test Decoder
    print("\n2. Testing Decoder:")
    decoder = Decoder(config)
    encoder_output = tf.random.normal((2, 10, 256))
    decoder_input = tf.random.normal((2, 5, 256))
    encoder_mask = tf.ones((2, 10))
    decoder_mask = tf.ones((2, 5))
    
    output = decoder(
        encoder_output, 
        encoder_mask, 
        decoder_input, 
        decoder_mask,
        training=False
    )
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Decoder input shape: {decoder_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test with padding
    print("\n3. Testing Decoder with padding:")
    encoder_mask = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=tf.float32)
    decoder_mask = tf.constant([[1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 1]], dtype=tf.float32)
    
    output = decoder(
        encoder_output,
        encoder_mask,
        decoder_input,
        decoder_mask,
        training=False
    )
    print(f"   Encoder mask shape: {encoder_mask.shape}")
    print(f"   Decoder mask shape: {decoder_mask.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n✓ All decoder tests passed!")

