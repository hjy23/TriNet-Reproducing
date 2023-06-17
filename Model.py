import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, SimpleRNN, Bidirectional, Flatten, \
    BatchNormalization, AvgPool1D
from tensorflow.keras.layers import Conv2D, MaxPool2D, Reshape
from encoder import Encoder2
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


def Model(params):
    # stage1：dcgr得到的158*8特征矩阵，输出2528维特征向量
    input_left = tf.keras.Input(shape=(158, 8))
    l = tf.expand_dims(input_left, -1)  # shape = (Batch_size, height, width, channels)
    l = Conv2D(16, kernel_size=(1, 8), strides=1, padding='valid')(l)
    ca = channel_attention(l, in_planes=16, ratio=8)  ##通道注意力机制,in_planes is the same as the feature maps of Conv2D
    l = tf.multiply(l, ca)
    # l_ca = tf.multiply(l, ca)
    # l=spatial_attention(l_ca)
    l = BatchNormalization()(l)
    l = Flatten()(l)
    l = Dense(params.dense_unit_dcgr, activation=params.activation_type)(l)
    output_left = Dropout(rate=params.d1)(l)

    # stage2：pssm输入进入双向lstm，输出256维特征向量
    input_middle = tf.keras.Input(shape=(50, 20))  ##pssm
    m = Bidirectional(LSTM(params.dense_unit_pssm, return_sequences=False))(input_middle)
    output_middle = BatchNormalization()(m)

    # stage3：理化性质输入，作为词向量输入自注意力机制，进行平均池化，输出50维向量
    input_right = tf.keras.Input(shape=(50, 8))  ##chemiphysical
    r = Encoder2(dff=params.dff, num_heads=params.head, num_layers=6)(input_right)
    print(r.shape)
    r = tf.transpose(r, (0, 2, 1))
    r = tf.nn.avg_pool(r, ksize=[8], strides=[1], padding='VALID')
    output_right = BatchNormalization()(r)
    output_right = Flatten()(output_right)

    concatenated = keras.layers.concatenate([output_left, output_middle, output_right])
    x = Dense(params.dense_unit_all, activation=params.activation_type)(concatenated)
    x = Dropout(rate=params.d2)(x)
    final_output = Dense(1, activation='sigmoid')(x)
    final_model = keras.models.Model(inputs=[input_left, input_middle, input_right], outputs=final_output)
    return final_model


def channel_attention(inputs, in_planes, ratio):  ##inputs为输入，in_planes为通道数，ratio可调整隐藏层大小
    avgpool = tf.keras.layers.GlobalAveragePooling2D(name='channel_avgpool')(inputs)  ##1*158维向量的全局平均池化
    maxpool = tf.keras.layers.GlobalMaxPooling2D(name='channel_maxpool')(inputs)  ##1*158维向量的最大池化

    # Shared MLP
    Dense_layer1 = tf.keras.layers.Dense(in_planes // ratio, activation='relu', name='channel_fc1')
    Dense_layer2 = tf.keras.layers.Dense(in_planes, activation='relu', name='channel_fc2')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = tf.keras.layers.add([avg_out, max_out])
    channel = tf.keras.layers.Activation('sigmoid', name='channel_sigmoid')(channel)
    channel_att = tf.keras.layers.Reshape((1, 1, in_planes), name='channel_reshape')(channel)
    # channel = tf.keras.layers.Activation('sigmoid', name='channel_sigmoid')(max_out)
    # channel_att = tf.keras.layers.Reshape((1, 1, in_planes), name='channel_reshape')(channel)
    return channel_att


def spatial_attention(channel_out):  ##inputs为输入，in_planes为通道数，ratio可调整隐藏层大小
    avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True, name='spatial_avgpool')
    maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True, name='spatial_maxpool')
    spatial = tf.keras.layers.Concatenate(axis=3)([avgpool, maxpool])

    spatial = Conv2D(1, (3, 3), strides=1, padding='same', name='spatial_conv2d')(spatial)
    spatial_out = Activation('sigmoid', name='spatial_sigmoid')(spatial)

    CBAM_out = tf.multiply(channel_out, spatial_out)
    return CBAM_out
