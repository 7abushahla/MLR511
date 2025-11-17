"""
TensorFlow/Keras implementation of SignBART layers.
Converted from PyTorch implementation.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    """
    Learned positional embeddings for SignBART.
    Equivalent to PyTorch's PositionalEmbedding.
    """
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.offset = 2
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding layer with offset
        self.embedding = layers.Embedding(
            input_dim=num_embeddings + self.offset,
            output_dim=embedding_dim,
            name='position_embedding'
        )
    
    def call(self, inputs_embeds, training=None):
        """
        Args:
            inputs_embeds: Tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            position_embeddings: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size = tf.shape(inputs_embeds)[0]
        seq_len = tf.shape(inputs_embeds)[1]
        
        # Create position indices: (batch_size, seq_len)
        positions = tf.range(0, seq_len, dtype=tf.int32)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [batch_size, 1])
        
        # Add offset
        positions = positions + self.offset
        
        # Get embeddings
        position_embeddings = self.embedding(positions)
        
        return position_embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Reconstruct layer from config."""
        return cls(
            num_embeddings=config['num_embeddings'],
            embedding_dim=config['embedding_dim'],
            name=config.get('name'),
            trainable=config.get('trainable', True),
            dtype=config.get('dtype')
        )


@tf.keras.utils.register_keras_serializable()
class FeedForwardLayer(layers.Layer):
    """
    Feed-forward network with residual connection and layer normalization.
    """
    def __init__(self, d_model, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        
        self.fc1 = layers.Dense(ffn_dim, name='fc1')
        self.fc2 = layers.Dense(d_model, name='fc2')
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-5, name='final_layer_norm')
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, training=None):
        residual = x
        x = self.fc1(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'ffn_dim': self.ffn_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ClassificationHead(layers.Layer):
    """
    Classification head for final predictions.
    """
    def __init__(self, input_dim, num_classes, dropout, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        self.dropout = layers.Dropout(rate=dropout, name='dropout')
        self.out_proj = layers.Dense(num_classes, name='out_proj')
    
    def call(self, x, training=None):
        x = self.dropout(x, training=training)
        logits = self.out_proj(x)
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Reconstruct layer from config."""
        return cls(
            input_dim=config['input_dim'],
            num_classes=config['num_classes'],
            dropout=config['dropout_rate'],
            name=config.get('name'),
            trainable=config.get('trainable', True),
            dtype=config.get('dtype')
        )


@tf.keras.utils.register_keras_serializable()
class Projection(layers.Layer):
    """
    Projects x and y coordinates to embedding dimension.
    Processes keypoints by separating x and y coordinates and projecting them.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = config
        self.num_joints = len(config['joint_idx'])
        self.d_model = config['d_model']
        
        self.proj_x1 = layers.Dense(config['d_model'], name='proj_x1')
        self.proj_y1 = layers.Dense(config['d_model'], name='proj_y1')
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: Tensor of shape (batch_size, seq_len, num_joints, 2)
                    Last dimension is [x, y] coordinates
        
        Returns:
            x_embed: Tensor of shape (batch_size, seq_len, d_model)
            y_embed: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Split coordinates: inputs[:, :, :, 0] for x, inputs[:, :, :, 1] for y
        x_coord = inputs[:, :, :, 0]  # (batch_size, seq_len, num_joints)
        y_coord = inputs[:, :, :, 1]  # (batch_size, seq_len, num_joints)
        
        # Project to d_model dimension
        x_embed = self.proj_x1(x_coord)  # (batch_size, seq_len, d_model)
        y_embed = self.proj_y1(y_coord)  # (batch_size, seq_len, d_model)
        
        return x_embed, y_embed
    
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


if __name__ == "__main__":
    # Test the layers
    print("Testing TensorFlow SignBART Layers...")
    
    # Test PositionalEmbedding
    print("\n1. Testing PositionalEmbedding:")
    pos_embed = PositionalEmbedding(num_embeddings=512, embedding_dim=256)
    test_input = tf.random.normal((2, 10, 256))  # (batch, seq_len, embed_dim)
    pos_output = pos_embed(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {pos_output.shape}")
    
    # Test ClassificationHead
    print("\n2. Testing ClassificationHead:")
    clf_head = ClassificationHead(input_dim=256, num_classes=64, dropout=0.1)
    test_input = tf.random.normal((2, 256))
    clf_output = clf_head(test_input, training=True)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {clf_output.shape}")
    
    # Test Projection
    print("\n3. Testing Projection:")
    config = {'joint_idx': list(range(100)), 'd_model': 256}
    projection = Projection(config)
    test_keypoints = tf.random.normal((2, 10, 100, 2))  # (batch, seq, joints, xy)
    x_embed, y_embed = projection(test_keypoints)
    print(f"   Input shape: {test_keypoints.shape}")
    print(f"   X embed shape: {x_embed.shape}")
    print(f"   Y embed shape: {y_embed.shape}")
    
    print("\n✓ All layer tests passed!")

