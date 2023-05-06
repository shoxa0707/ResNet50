import tensorflow as tf
from tensorflow import keras

class ResNet50():
    def __init__(self, include_top=True):
        self.include_top = include_top
        # pooling layers
        self.maxpool = keras.layers.MaxPooling2D((3,3), strides=2, padding='same', name='Maxpool')
        self.avpool = keras.layers.AveragePooling2D(pool_size=(7,7), padding='valid', strides=1, name='average_pool')
        # Fully connected layers and Relu
        self.relu = keras.layers.ReLU(name='relu')
        self.fc = keras.layers.Dense(1000, activation='softmax', name='fc')

        # BLOCK 1
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu', name='conv1')
        self.batch1 = keras.layers.BatchNormalization(name='batch_norm_1')

        # BLOCK 2
        self.conv2_11 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', name='conv2_11')
        self.batch2_11 = keras.layers.BatchNormalization(name='batch_norm_2_11')
        self.conv2_12 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv2_12')
        self.batch2_12 = keras.layers.BatchNormalization(name='batch_norm_2_12')
        self.conv2_13 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='conv2_13')
        self.batch2_13 = keras.layers.BatchNormalization(name='batch_norm_2_13')
        self.skipconv2_1 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='skipconv2_1')
        self.skipbatch2_1 = keras.layers.BatchNormalization(name='skipbatch2_1')

        self.conv2_21 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', name='conv2_21')
        self.batch2_21 = keras.layers.BatchNormalization(name='batch_norm_2_21')
        self.conv2_22 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv2_22')
        self.batch2_22 = keras.layers.BatchNormalization(name='batch_norm_2_22')
        self.conv2_23 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='conv2_23')
        self.batch2_23 = keras.layers.BatchNormalization(name='batch_norm_2_23')

        self.conv2_31 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', name='conv2_31')
        self.batch2_31 = keras.layers.BatchNormalization(name='batch_norm_2_31')
        self.conv2_32 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv2_32')
        self.batch2_32 = keras.layers.BatchNormalization(name='batch_norm_2_32')
        self.conv2_33 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='conv2_33')
        self.batch2_33 = keras.layers.BatchNormalization(name='batch_norm_2_33')
        # BLOCK 3
        self.conv3_11 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=2, padding='same', name='conv3_11')
        self.batch3_11 = keras.layers.BatchNormalization(name='batch_norm_3_11')
        self.conv3_12 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='conv3_12')
        self.batch3_12 = keras.layers.BatchNormalization(name='batch_norm_3_12')
        self.conv3_13 = keras.layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', name='conv3_13')
        self.batch3_13 = keras.layers.BatchNormalization(name='batch_norm_3_13')
        self.skipconv3_1 = keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='same', name='skipconv3_1')
        self.skipbatch3_1 = keras.layers.BatchNormalization(name='skipbatch3_1')

        self.conv3_21 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', name='conv3_21')
        self.batch3_21 = keras.layers.BatchNormalization(name='batch_norm_3_21')
        self.conv3_22 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='conv3_22')
        self.batch3_22 = keras.layers.BatchNormalization(name='batch_norm_3_22')
        self.conv3_23 = keras.layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', name='conv3_23')
        self.batch3_23 = keras.layers.BatchNormalization(name='batch_norm_3_23')

        self.conv3_31 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', name='conv3_31')
        self.batch3_31 = keras.layers.BatchNormalization(name='batch_norm_3_31')
        self.conv3_32 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='conv3_32')
        self.batch3_32 = keras.layers.BatchNormalization(name='batch_norm_3_32')
        self.conv3_33 = keras.layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', name='conv3_33')
        self.batch3_33 = keras.layers.BatchNormalization(name='batch_norm_3_33')

        self.conv3_41 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', name='conv3_41')
        self.batch3_41 = keras.layers.BatchNormalization(name='batch_norm_3_41')
        self.conv3_42 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='conv3_42')
        self.batch3_42 = keras.layers.BatchNormalization(name='batch_norm_3_42')
        self.conv3_43 = keras.layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', name='conv3_43')
        self.batch3_43 = keras.layers.BatchNormalization(name='batch_norm_3_43')
        # BLOCK 4
        self.conv4_11 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='same', name='conv4_11')
        self.batch4_11 = keras.layers.BatchNormalization(name='batch_norm_4_11')
        self.conv4_12 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='conv4_12')
        self.batch4_12 = keras.layers.BatchNormalization(name='batch_norm_4_12')
        self.conv4_13 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same', name='conv4_13')
        self.batch4_13 = keras.layers.BatchNormalization(name='batch_norm_4_13')
        self.skipconv4_1 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=2, padding='same', name='skipconv4_1')
        self.skipbatch4_1 = keras.layers.BatchNormalization(name='skipbatch4_1')

        self.conv4_21 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='conv4_21')
        self.batch4_21 = keras.layers.BatchNormalization(name='batch_norm_4_21')
        self.conv4_22 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='conv4_22')
        self.batch4_22 = keras.layers.BatchNormalization(name='batch_norm_4_22')
        self.conv4_23 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same', name='conv4_23')
        self.batch4_23 = keras.layers.BatchNormalization(name='batch_norm_4_23')

        self.conv4_31 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='conv4_31')
        self.batch4_31 = keras.layers.BatchNormalization(name='batch_norm_4_31')
        self.conv4_32 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='conv4_32')
        self.batch4_32 = keras.layers.BatchNormalization(name='batch_norm_4_32')
        self.conv4_33 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same', name='conv4_33')
        self.batch4_33 = keras.layers.BatchNormalization(name='batch_norm_4_33')
        
        self.conv4_41 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='conv4_41')
        self.batch4_41 = keras.layers.BatchNormalization(name='batch_norm_4_41')
        self.conv4_42 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='conv4_42')
        self.batch4_42 = keras.layers.BatchNormalization(name='batch_norm_4_42')
        self.conv4_43 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same', name='conv4_43')
        self.batch4_43 = keras.layers.BatchNormalization(name='batch_norm_4_43')

        self.conv4_51 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='conv4_51')
        self.batch4_51 = keras.layers.BatchNormalization(name='batch_norm_4_51')
        self.conv4_52 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='conv4_52')
        self.batch4_52 = keras.layers.BatchNormalization(name='batch_norm_4_52')
        self.conv4_53 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same', name='conv4_53')
        self.batch4_53 = keras.layers.BatchNormalization(name='batch_norm_4_53')

        self.conv4_61 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', name='conv4_61')
        self.batch4_61 = keras.layers.BatchNormalization(name='batch_norm_4_61')
        self.conv4_62 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='conv4_62')
        self.batch4_62 = keras.layers.BatchNormalization(name='batch_norm_4_62')
        self.conv4_63 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same', name='conv4_63')
        self.batch4_63 = keras.layers.BatchNormalization(name='batch_norm_4_63')
        # BLOCK 5
        self.conv5_11 = keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='same', name='conv5_11')
        self.batch5_11 = keras.layers.BatchNormalization(name='batch_norm_5_11')
        self.conv5_12 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='conv5_12')
        self.batch5_12 = keras.layers.BatchNormalization(name='batch_norm_5_12')
        self.conv5_13 = keras.layers.Conv2D(filters=2048, kernel_size=(1,1), padding='same', name='conv5_13')
        self.batch5_13 = keras.layers.BatchNormalization(name='batch_norm_5_13')
        self.skipconv5_1 = keras.layers.Conv2D(filters=2048, kernel_size=(1,1), strides=2, padding='same', name='skipconv5_1')
        self.skipbatch5_1 = keras.layers.BatchNormalization(name='skipbatch5_1')

        self.conv5_21 = keras.layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', name='conv5_21')
        self.batch5_21 = keras.layers.BatchNormalization(name='batch_norm_5_21')
        self.conv5_22 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='conv5_22')
        self.batch5_22 = keras.layers.BatchNormalization(name='batch_norm_5_22')
        self.conv5_23 = keras.layers.Conv2D(filters=2048, kernel_size=(1,1), padding='same', name='conv5_23')
        self.batch5_23 = keras.layers.BatchNormalization(name='batch_norm_5_23')

        self.conv5_31 = keras.layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', name='conv5_31')
        self.batch5_31 = keras.layers.BatchNormalization(name='batch_norm_5_31')
        self.conv5_32 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='conv5_32')
        self.batch5_32 = keras.layers.BatchNormalization(name='batch_norm_5_32')
        self.conv5_33 = keras.layers.Conv2D(filters=2048, kernel_size=(1,1), padding='same', name='conv5_33')
        self.batch5_33 = keras.layers.BatchNormalization(name='batch_norm_5_33')

    def __call__(self, x):
        # BLOCK 1
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.maxpool(x)
        # BLOCK 2
        x_skip = x
        x_skip = self.skipconv2_1(x_skip)
        x_skip = self.skipbatch2_1(x_skip)

        x = self.conv2_11(x)
        x = self.batch2_11(x)
        x = self.relu(x)
        x = self.conv2_12(x)
        x = self.batch2_12(x)
        x = self.relu(x)
        x = self.conv2_13(x)
        x = self.batch2_13(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        
        x_skip = x

        x = self.conv2_21(x)
        x = self.batch2_21(x)
        x = self.relu(x)
        x = self.conv2_22(x)
        x = self.batch2_22(x)
        x = self.relu(x)
        x = self.conv2_23(x)
        x = self.batch2_23(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)

        x_skip = x

        x = self.conv2_31(x)
        x = self.batch2_31(x)
        x = self.relu(x)
        x = self.conv2_32(x)
        x = self.batch2_32(x)
        x = self.relu(x)
        x = self.conv2_33(x)
        x = self.batch2_33(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        # BLOCK 3
        x_skip = x
        x_skip = self.skipconv3_1(x_skip)
        x_skip = self.skipbatch3_1(x_skip)

        x = self.conv3_11(x)
        x = self.batch3_11(x)
        x = self.relu(x)
        x = self.conv3_12(x)
        x = self.batch3_12(x)
        x = self.relu(x)
        x = self.conv3_13(x)
        x = self.batch3_13(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        
        x_skip = x

        x = self.conv3_21(x)
        x = self.batch3_21(x)
        x = self.relu(x)
        x = self.conv3_22(x)
        x = self.batch3_22(x)
        x = self.relu(x)
        x = self.conv3_23(x)
        x = self.batch3_23(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)

        x_skip = x

        x = self.conv3_31(x)
        x = self.batch3_31(x)
        x = self.relu(x)
        x = self.conv3_32(x)
        x = self.batch3_32(x)
        x = self.relu(x)
        x = self.conv3_33(x)
        x = self.batch3_33(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)

        x_skip = x

        x = self.conv3_41(x)
        x = self.batch3_41(x)
        x = self.relu(x)
        x = self.conv3_42(x)
        x = self.batch3_42(x)
        x = self.relu(x)
        x = self.conv3_43(x)
        x = self.batch3_43(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        # BLOCK 4
        x_skip = x
        x_skip = self.skipconv4_1(x_skip)
        x_skip = self.skipbatch4_1(x_skip)

        x = self.conv4_11(x)
        x = self.batch4_11(x)
        x = self.relu(x)
        x = self.conv4_12(x)
        x = self.batch4_12(x)
        x = self.relu(x)
        x = self.conv4_13(x)
        x = self.batch4_13(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        
        x_skip = x

        x = self.conv4_21(x)
        x = self.batch4_21(x)
        x = self.relu(x)
        x = self.conv4_22(x)
        x = self.batch4_22(x)
        x = self.relu(x)
        x = self.conv4_23(x)
        x = self.batch4_23(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)

        x_skip = x

        x = self.conv4_31(x)
        x = self.batch4_31(x)
        x = self.relu(x)
        x = self.conv4_32(x)
        x = self.batch4_32(x)
        x = self.relu(x)
        x = self.conv4_33(x)
        x = self.batch4_33(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)

        x_skip = x

        x = self.conv4_41(x)
        x = self.batch4_41(x)
        x = self.relu(x)
        x = self.conv4_42(x)
        x = self.batch4_42(x)
        x = self.relu(x)
        x = self.conv4_43(x)
        x = self.batch4_43(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        
        x_skip = x

        x = self.conv4_51(x)
        x = self.batch4_51(x)
        x = self.relu(x)
        x = self.conv4_52(x)
        x = self.batch4_52(x)
        x = self.relu(x)
        x = self.conv4_53(x)
        x = self.batch4_53(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)

        x_skip = x

        x = self.conv4_61(x)
        x = self.batch4_61(x)
        x = self.relu(x)
        x = self.conv4_62(x)
        x = self.batch4_62(x)
        x = self.relu(x)
        x = self.conv4_63(x)
        x = self.batch4_63(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        # BLOCK 5
        x_skip = x
        x_skip = self.skipconv5_1(x_skip)
        x_skip = self.skipbatch5_1(x_skip)

        x = self.conv5_11(x)
        x = self.batch5_11(x)
        x = self.relu(x)
        x = self.conv5_12(x)
        x = self.batch5_12(x)
        x = self.relu(x)
        x = self.conv5_13(x)
        x = self.batch5_13(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        
        x_skip = x

        x = self.conv5_21(x)
        x = self.batch5_21(x)
        x = self.relu(x)
        x = self.conv5_22(x)
        x = self.batch5_22(x)
        x = self.relu(x)
        x = self.conv5_23(x)
        x = self.batch5_23(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)

        x_skip = x

        x = self.conv5_31(x)
        x = self.batch5_31(x)
        x = self.relu(x)
        x = self.conv5_32(x)
        x = self.batch5_32(x)
        x = self.relu(x)
        x = self.conv5_33(x)
        x = self.batch5_33(x)
        x = keras.layers.Add()([x, x_skip])
        x = self.relu(x)
        # FC layer
        if self.include_top:
            x = self.avpool(x)
            x = self.fc(x)
        return x
