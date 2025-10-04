import numpy as np
from tensorflow import keras
from keras.layers import Layer, Add, Conv2D, Dropout
from keras.layers import Activation, ELU, LeakyReLU, ReLU
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization, TimeDistributed
from keras.layers import LayerNormalization
from keras.regularizers import l2
import tensorflow as tf

class WarmUpCosineDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate = 1e-4, peak_lr=1e-3, initial_lr=1e-4, warmup_steps=360, first_decay_steps=32, t_mul=1.5, m_mul=0.9, alpha= 0.1):
        self.learning_rate = learning_rate
        self.peak_lr = peak_lr
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.cosine_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        
    def __call__(self, step):
        # Linear warm-up
        # Convert step to a Tensor, if it is not already one
        step = tf.cast(step, tf.float32)
        
        # Linear warm-up
        warmup_lr = self.initial_lr + (self.peak_lr - self.initial_lr) * (step / self.warmup_steps)
        
        # Cosine annealing after warm-up
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=self.peak_lr, 
            first_decay_steps=self.cosine_decay_steps, 
            t_mul=self.t_mul, 
            m_mul=self.m_mul, 
            alpha=self.alpha
        )
        
        # Use tf.cond to handle the warm-up phase vs cosine decay
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,  # If step is in warm-up phase
            lambda: lr_schedule(step - self.warmup_steps)  # If step is beyond warm-up
        )  
    # Add the get_config method for serialization
    def get_config(self):
        config = {
            "peak_lr": self.peak_lr,
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "first_decay_steps": self.cosine_decay_steps,
            "t_mul": self.t_mul,
            "m_mul": self.m_mul,
            "alpha": self.alpha
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1,1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    def build(self, input_shape):
        super(ReflectionPadding2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0], 
            None if input_shape[1] is None else input_shape[1] + 2 * self.padding[0],
            None if input_shape[2] is None else input_shape[2] + 2 * self.padding[1],
            input_shape[3]
        )

    def call(self, x):
        (i_pad, j_pad) = self.padding
        return tf.pad(x, [[0,0], [i_pad, i_pad], [j_pad, j_pad], [0,0]], 'REFLECT')

class ZeroLikeLayer(Layer):
    def call(self, y):
        return tf.zeros_like(y[:, 0, ...])


class ConvBlock(Layer):
    def __init__(self, channels, conv_size=(3,3), time_dist=False,
                 norm=None, stride=1, activation='relu', padding='same',
                 order=("conv", "act", "dropout", "norm"), scale_norm=False,
                 dropout=0, **kwargs):

        super().__init__(**kwargs)
        TD = TimeDistributed if time_dist else (lambda x: x)

        if padding == 'reflect':
            pad = tuple((s-1)//2 for s in conv_size)
            self.padding = TD(ReflectionPadding2D(padding=pad))
        else:
            self.padding = lambda x: x

        self.conv = TD(Conv2D(
            channels, conv_size,
            padding='valid' if padding == 'reflect' else padding,
            strides=(stride, stride),
        ))

        if activation == 'leakyrelu':
            self.act = LeakyReLU(0.2)
        elif activation == 'relu':
            self.act = ReLU()
        elif activation == 'elu':
            self.act = ELU()
        else:
            self.act = Activation(activation)

        if norm == "batch":
            self.norm = BatchNormalization(momentum=0.95, scale=scale_norm)
        elif norm == "layer":
            self.norm = LayerNormalization(scale=scale_norm)
        else:
            self.norm = lambda x: x

        if dropout > 0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = lambda x: x

        self.order = order

    def call(self, x):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(self.padding(x))
            elif layer == "act":
                x = self.act(x)
            elif layer == "norm":
                x = self.norm(x)
            elif layer == "dropout":
                x = self.dropout(x)
            else:
                raise ValueError(f"Unknown layer {layer}")
        return x


class ResBlock(Layer):
    def __init__(self, channels, time_dist=False, stride=1, dropout=0.0, norm=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.stride = stride
        TD = TimeDistributed if time_dist else (lambda x: x)

        if self.stride > 1:
            self.pool = TD(AveragePooling2D(pool_size=(self.stride, self.stride)))
        else:
            self.pool = lambda x: x
        
        self.proj = TD(Conv2D(self.channels, kernel_size=(1,1)))

        # Pass the arguments to ConvBlock instances
        self.conv_block_1 = ConvBlock(channels, stride=self.stride, dropout=dropout, norm=norm, time_dist=time_dist)
        self.conv_block_2 = ConvBlock(channels, activation='leakyrelu', dropout=dropout, norm=norm, time_dist=time_dist)
        self.add = Add()

    def call(self, x):
        x_in = self.pool(x)
        in_channels = int(x.shape[-1])
        if in_channels != self.channels:
            x_in = self.proj(x_in)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.add([x, x_in])


class GRUResBlock(ResBlock):
    def __init__(self, channels, conv_size=(3, 3), padding='same', final_activation='sigmoid', **kwargs):
        super().__init__(channels, **kwargs)
        self.conv_size = conv_size
        self.padding = padding
        self.final_act = Activation(final_activation)

    def call(self, x):
        x_in = self.proj(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.add([x, x_in])
        return self.final_act(x)


class ConvGRU(Layer):
    def __init__(self, channels, conv_size=(3,3),
                 return_sequences=False, time_steps=1,
                 **kwargs):

        super().__init__(**kwargs)

        self.update_gate = Conv2D(channels, conv_size, activation='sigmoid',
                                  padding='same')
        self.reset_gate = Conv2D(channels, conv_size, activation='sigmoid',
                                 padding='same')
        self.output_gate = Conv2D(channels, conv_size, padding='same')
        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def build(self, input_shape):
        super(ConvGRU, self).build(input_shape)

    def iterate(self, x, h):
        xh = tf.concat((x, h), axis=-1)
        z = self.update_gate(xh)
        r = self.reset_gate(xh)
        o = self.output_gate(tf.concat((x, r*h), axis=-1))
        h = z * h + (1.0 - z) * tf.math.tanh(o)
        return h

    def call(self, inputs):
        xt, h = inputs
        h_all = []
        for t in range(self.time_steps):
            x = xt[:, t, ...]
            h = self.iterate(x, h)
            if self.return_sequences:
                h_all.append(h)
        return tf.stack(h_all, axis=1) if self.return_sequences else h


class ResGRU(ConvGRU):
    def __init__(self, channels, conv_size=(3,3),
                 return_sequences=False, time_steps=1,
                 **kwargs):

        dropout = kwargs.pop("dropout", 0.0)
        norm = kwargs.pop("norm", None)
        super().__init__(channels=channels, conv_size=conv_size,
                         return_sequences=return_sequences, time_steps=time_steps, **kwargs)

        self.update_gate = GRUResBlock(channels, conv_size=conv_size,
                                       final_activation='sigmoid', padding='same', dropout=dropout,
                                       norm=norm)

        self.reset_gate = GRUResBlock(channels, conv_size=conv_size,
                                      final_activation='sigmoid', padding='same', dropout=dropout,
                                      norm=norm)

        self.output_gate = GRUResBlock(channels, conv_size=conv_size,
                                       final_activation='linear', padding='same', dropout=dropout,
                                       norm=norm)

    def build(self, input_shape):
        super(ResGRU, self).build(input_shape)

from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

def slice_to_n_steps(x):
    # Keep only steps [0..4]
    n = 12
    return x[:, :n, :, :, :]  # if x is 5D: (batch, time, height, width, channels)

def slice_output_shape(input_shape):
    n = 12
    # input_shape = (batch_size, 10, height, width, channels)
    return (input_shape[0], n, input_shape[2], input_shape[3], input_shape[4])

def reshape_and_stack(x):
    """Reshape and stack function for model compatibility"""
    return x

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import numpy as np

class WarmUpCosineDecayScheduler(LearningRateSchedule):
    """
    Warm-up cosine decay learning rate scheduler.
    
    This scheduler implements a warm-up phase followed by cosine decay.
    """
    
    def __init__(self, peak_lr=0.001, initial_lr=0.0001, warmup_steps=360, 
                 first_decay_steps=32, t_mul=1.5, m_mul=0.9, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.peak_lr = peak_lr
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warm-up phase
        warmup_lr = self.initial_lr + (self.peak_lr - self.initial_lr) * (step / self.warmup_steps)
        
        # Cosine decay phase
        decay_steps = tf.maximum(step - self.warmup_steps, 0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_steps / self.first_decay_steps))
        decay_lr = self.alpha + (self.peak_lr - self.alpha) * cosine_decay
        
        # Choose between warmup and decay
        lr = tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
        
        return lr
    
    def get_config(self):
        return {
            'peak_lr': self.peak_lr,
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'first_decay_steps': self.first_decay_steps,
            't_mul': self.t_mul,
            'm_mul': self.m_mul,
            'alpha': self.alpha
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
