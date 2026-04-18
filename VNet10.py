import tensorflow as tf



VNET_PROPERTIES = [
    {'depth': 0, 'layers': 1, 'in_channels': 16},
    {'depth': 1, 'layers': 2, 'in_channels': 32},
    {'depth': 2, 'layers': 3, 'in_channels': 64},
    {'depth': 3, 'layers': 3, 'in_channels': 128},
    {'depth': 4, 'layers': 3, 'in_channels': 256},
]

def get_initialization(initializer_name):
    if initializer_name == "normal":
        return tf.keras.initializers.RandomNormal(mean=0, stddev=1)
    elif initializer_name == "xavier":
        return tf.keras.initializers.GlorotNormal()
    elif initializer_name == "he":
        return tf.keras.initializers.HeNormal()
    elif initializer_name == "orthogonal":
        tf.keras.initializers.Orthogonal()
    else:
        return tf.keras.initializers.GlorotNormal()


def get_activation_function(activation_function_name):
    if activation_function_name == "prelu":
        return tf.keras.layers.PReLU()
    elif activation_function_name == "lrelu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.2)
    elif activation_function_name == "sigmoid":
        return tf.keras.layers.Activation("sigmoid")
    elif activation_function_name == "tanh":
        return tf.keras.layers.Activation("tanh")
    elif activation_function_name == "softmax":
        return tf.keras.layers.Activation("softmax")
    else:
        return tf.keras.layers.Activation("relu")


@tf.keras.utils.register_keras_serializable()
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, number, output_channels, initializer_name, activation_function_name, dropoutrate=0.0,
                 kernel_size=(5, 5, 5), strides=(1, 1, 1), concatination=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.number = number
        self.dropoutrate = dropoutrate
        self.concatination = concatination
        self.initializer_name = initializer_name
        self.activation_function_name = activation_function_name
        self.initialization = get_initialization(initializer_name)
        self.activation_function = get_activation_function(activation_function_name)
        self.conv = tf.keras.layers.Conv3D(filters=output_channels, kernel_size=kernel_size, strides=strides,
                                        padding='same', kernel_initializer=self.initialization)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation_function = self.activation_function

        if dropoutrate != 0.0:
            self.dropout = tf.keras.layers.Dropout(dropoutrate)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation_function(x)
        if self.dropoutrate != 0.0:
            x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "number": self.number,
            "output_channels": self.output_channels,
            "initializer_name": self.initializer_name,
            "activation_function_name": self.activation_function_name,
            "dropoutrate": self.dropoutrate,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "concatination": self.concatination,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class ConvResampleBlock(tf.keras.layers.Layer):
    def __init__(self, number, in_channels, sample_type, initializer_name, activation_function_name, dropoutrate=0.0,
                 kernel_size=(2, 2, 2), strides=(2, 2, 2), **kwargs):
        super(ConvResampleBlock, self).__init__()
        self.number = number
        self.dropoutrate = dropoutrate
        self.initializer_name = initializer_name
        self.activation_function_name = activation_function_name
        self.initialization = get_initialization(initializer_name)
        self.activation_function = get_activation_function(activation_function_name)
        out_channels = self.get_out_channels(in_channels, sample_type)

        self.conv = self.get_conv(sample_type, self.initialization, out_channels=out_channels,
                                  kernel_size=kernel_size, strides=strides)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation_function = self.activation_function

        if dropoutrate != 0.0:
            self.dropout = tf.keras.layers.Dropout(dropoutrate)

    def get_conv(self, sample_type, initialization, out_channels, kernel_size, strides):
        # Upsample or down sample based on the sample type provided
        if sample_type == "Up":
            result = tf.keras.layers.Conv3DTranspose(filters=out_channels, kernel_size=kernel_size, strides=strides,
                                                  padding='same', kernel_initializer=initialization)
        else:
            result = tf.keras.layers.Conv3D(filters=out_channels, kernel_size=kernel_size, strides=strides,
                                            padding='same', kernel_initializer=initialization)
        return result

    def get_out_channels(self, in_channels, sample_type):
        scale = 2 if sample_type == "Down" else 0.5
        out_channels = int(in_channels * scale)
        return out_channels

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation_function(x)
        if self.dropoutrate != 0.0:
            x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "number": self.number,
            "in_channels": self.in_channels,
            "sample_type": self.sample_type,
            "initializer_name": self.initializer_name,
            "activation_function_name": self.activation_function_name,
            "dropoutrate": self.dropoutrate,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class Layer(tf.keras.layers.Layer):
    def __init__(self, number, in_channels, layer_quantity, initializer_name, activation_function_name, dropoutrate,
                 sample_type=None, concatination=False, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.number = number
        self.layer_quantity = layer_quantity
        self.sample_type = sample_type
        self.concatination = concatination
        out_channels = in_channels
        self.initializer_name = initializer_name
        self.activation_function_name = activation_function_name
        self.initialization = get_initialization(initializer_name)
        self.activation_function = get_activation_function(activation_function_name)

        # Special case where initial conv requires concatination (for up sampling) need because it cannot handle if once there are 256 and the ot
        # time there are 128 as an input, therefore two with the same output size needed and cannot be used twice
        if self.concatination is not None:
            self.concatination_block = ConvBlock(number, out_channels, self.initializer_name, self.activation_function_name,
                                                 dropoutrate, concatination=self.concatination)
        # Initialise convolution block
        self.conv = ConvBlock(number, out_channels, self.initializer_name, self.activation_function_name, dropoutrate)
        # Initialise resample convolution block if upsample or downsample is specified
        self.resample_conv = ConvResampleBlock(number, in_channels, self.sample_type, self.initializer_name,
                                               self.activation_function_name, dropoutrate)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input, training=None, mask=None, residual=None):
        out = input

        if residual is not None:
            out = tf.concat([out, residual], axis=4)

        # Pass data through number of convolutional layers determined by the amount of layers required
        for i in range(self.layer_quantity):
            if self.concatination and i == 0:
                out = self.concatination_block(out)
            else:
                out = self.conv(out)
        residual = out

        out = self.resample_conv(out) if self.sample_type is not None else out
        if self.sample_type == "Down":
            return out, residual
        else:
            return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "number": self.number,
            "in_channels": self.in_channels,
            "layer_quantity": self.layer_quantity,
            "initializer_name": self.initializer_name,
            "activation_function_name": self.activation_function_name,
            "dropoutrate": self.dropoutrate,
            "sample_type": self.sample_type,
            "concatination": self.concatination,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VNet(tf.keras.Model):
    def __init__(self, initializer_name, activation_function_name, dropoutrate, **kwargs):
        super().__init__(**kwargs)

        self.initializer_name = initializer_name
        self.activation_function_name = activation_function_name
        self.initialization = get_initialization(initializer_name)
        self.activation_function = get_activation_function(activation_function_name)
        self.dropoutrate = dropoutrate

        self.beginning_conv = ConvBlock(0, 16, self.initializer_name, self.activation_function_name, dropoutrate)
        # Encoder
        # Down depth layers

        self.down_depth0 = Layer(1, self.get_in_channels(0), self.get_layer_quantity(0), self.initializer_name,
                                 self.activation_function_name, dropoutrate, "Down")  # in 16 out 32 out 0
        self.down_depth1 = Layer(2, self.get_in_channels(1), self.get_layer_quantity(1), self.initializer_name,
                                 self.activation_function_name, dropoutrate, "Down")  # in 32 out 64 out 1

        self.up_depth2 = Layer(3, self.get_in_channels(2), self.get_layer_quantity(2), self.initializer_name,
                               self.activation_function_name, dropoutrate, "Up")  # in 64 out 32
        
        self.up_depth1 = Layer(4, self.get_in_channels(1), self.get_layer_quantity(1), self.initializer_name,
                               self.activation_function_name, dropoutrate, "Up", concatination=True)  # in 32 residual 32 concat 64 layers 32 out 16
        self.up_depth0 = Layer(5, self.get_in_channels(0), self.get_layer_quantity(0), self.initializer_name,
                               self.activation_function_name, dropoutrate, None, concatination=True)  # in 16 residual 16 concat 33 layer 16 out 16

        self.final_conv = ConvBlock(6, 3, self.initializer_name, self.activation_function_name) #change53

        self.activation = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=None, mask=None):
        out = self.beginning_conv(inputs)
        out, residual0 = self.down_depth0(out)
        out, residual1 = self.down_depth1(out)
        out = self.up_depth2(out)  # residual4 not needed because it is the lowest layer
        out = self.up_depth1(out, residual=residual1)
        out = self.up_depth0(out, residual=residual0)
        out = self.final_conv(out)
        out = self.activation(out)
        return out

    def get_config(self):
        config = super(VNet, self).get_config()
        config.update({
            "initializer_name": self.initializer_name,
            "activation_function_name": self.activation_function_name,
            "dropoutrate": self.dropoutrate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_layer_quantity(self, depth_value):
        # Returns the number of layers based on depth level
        return next(item for item in VNET_PROPERTIES if item["depth"] == depth_value)['layers']

    def get_in_channels(self, depth_value):
        # Returns the number of layers based on depth level
        return next(item for item in VNET_PROPERTIES if item["depth"] == depth_value)['in_channels']
