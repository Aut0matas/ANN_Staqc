import keras.backend as backend
from keras.layers import Layer
import tensorflow as tf


class SelfAttention(Layer):
    def __init__(self, num_heads, attention_dim, name, **kwargs):
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.scope = name
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight_1 = self.add_weight(
            name=f"weight_1_{self.scope}",
            shape=(input_shape[2], self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.weight_2 = self.add_weight(
            name=f"weight_2_{self.scope}",
            shape=(self.attention_dim, self.num_heads),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        weight_1_output = backend.dot(inputs, self.weight_1)
        weight_1_output = tf.tanh(tf.transpose(weight_1_output))
        weight_1_output = tf.transpose(weight_1_output)
        weight_2_output = backend.softmax(backend.dot(weight_1_output, self.weight_2))
        attention_score = backend.permute_dimensions(weight_2_output, (0, 2, 1))
        weighted_input_vectors = tf.matmul(attention_score, inputs)
        identity_matrix = tf.tile(tf.eye(self.num_heads), [tf.shape(inputs)[0], 1])
        identity_matrix = tf.reshape(
            identity_matrix, [-1, self.num_heads, self.num_heads]
        )
        penalty_term = tf.square(
            tf.norm(
                tf.matmul(attention_score, weight_2_output) - identity_matrix,
                axis=[-2, -1],
                ord="fro",
            )
        )
        return [weighted_input_vectors, penalty_term]

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[0], self.attention_dim, self.num_heads),
            (input_shape[0],),
        ]
