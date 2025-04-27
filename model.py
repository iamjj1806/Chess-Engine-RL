import tensorflow as tf
from config import Config

class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with 2 convolutional layers and a skip connection."""
    
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = x + inputs  # Skip connection
        return tf.nn.relu(x)

class ChessModel(tf.keras.Model):
    """Neural network model for chess with policy and value heads."""
    
    def __init__(self):
        super().__init__()
        
        # Initial convolutional layer
        self.conv = tf.keras.layers.Conv2D(
            Config.NUM_FILTERS, kernel_size=3, padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        
        # Residual blocks
        self.residual_blocks = [ResidualBlock(Config.NUM_FILTERS) 
                               for _ in range(Config.NUM_RESIDUAL_BLOCKS)]
        
        # Policy head
        self.policy_conv = tf.keras.layers.Conv2D(
            2, kernel_size=1, padding='same', use_bias=False)
        self.policy_bn = tf.keras.layers.BatchNormalization()
        self.policy_dense = tf.keras.layers.Dense(Config.OUTPUT_SIZE)
        
        # Value head
        self.value_conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, padding='same', use_bias=False)
        self.value_bn = tf.keras.layers.BatchNormalization()
        self.value_dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.value_dense2 = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        
        for block in self.residual_blocks:
            x = block(x, training=training)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy, training=training)
        policy = tf.nn.relu(policy)
        policy = tf.keras.layers.Flatten()(policy)
        policy = self.policy_dense(policy)
        
        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value, training=training)
        value = tf.nn.relu(value)
        value = tf.keras.layers.Flatten()(value)
        value = self.value_dense1(value)
        value = self.value_dense2(value)
        
        return policy, value

def create_model():
    """Create and compile the chess model."""
    model = ChessModel()
    
    # Define inputs
    inputs = tf.keras.Input(shape=Config.INPUT_SHAPE)
    
    # Connect model to inputs
    policy, value = model(inputs)
    
    # Create model with explicit inputs and outputs
    full_model = tf.keras.Model(inputs=inputs, outputs=[policy, value])
    
    # Compile model
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            tf.keras.losses.MeanSquaredError()
        ],
        loss_weights=[1.0, 1.0]  # Equal weight to policy and value losses
    )
    
    return full_model

def save_model(model, path):
    """Save the model to disk."""
    model.save_weights(path)
    print(f"Model saved to {path}")

def load_model(path):
    """Load the model from disk."""
    model = create_model()
    model.load_weights(path)
    print(f"Model loaded from {path}")
    return model
