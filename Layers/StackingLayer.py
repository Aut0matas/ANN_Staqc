import keras.backend as backend
from keras.layers import Layer
import tensorflow as tf


class StackingLayer(Layer):
    def __init__(self, **kwargs):
        super(StackingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StackingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sentence_token_level_outputs = tf.stack(inputs, axis=0)
        sentence_token_level_outputs = backend.permute_dimensions(
            sentence_token_level_outputs, (1, 0, 2)
        )
        return sentence_token_level_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], len(input_shape), input_shape[0][1])
