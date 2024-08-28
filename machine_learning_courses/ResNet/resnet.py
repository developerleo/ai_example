import tensorflow as tf
import keras
from keras import layers, Sequential
from keras.src.layers import BatchNormalization


class BasicBlock(layers.Layer):

    # filters_num 把输入的channel转化为BasicBlock中的channel
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        # Padding 设置为same，保证输入输出的shape一致，不会被卷积操作表小
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, traininig=None):

        # input.shape = [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


# ResNet - 18
class ResNet(keras.Model):

    def __init__(self, layer_dimensions, num_classes=100):  # [2, 2, 2, 2]
        super(ResNet, self).__init__()

        # 预处理层
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])

        self.layers1 = self.build_resBlock(64, layer_dimensions[0])
        self.layers2 = self.build_resBlock(128, layer_dimensions[1], stride=2)
        self.layers3 = self.build_resBlock(256, layer_dimensions[2], stride=2)
        self.layers4 = self.build_resBlock(512, layer_dimensions[3], stride=2)

        # output: [b, 512, h, w]
        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)

        # 【b, c]
        x = self.avgpool(x)
        # 【b, 100]
        x = self.fc(x)

        return x

    def build_resBlock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()

        # 第一个block的步长可能不为1，进行下采样
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])
