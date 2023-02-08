import tensorflow as tf


class MLModel:
    def __init__(self):
        self.dropout_rate = 0.2
        self.len_classes = 3
        self.model = tf.keras.Model
        self.BATCH_SIZE = 4

    def fcn_model(self):
        # Input layer
        input = tf.keras.layers.Input(shape=(None, None, 3))

        # A convolution block
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1)(input)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)


        x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Fully connected layer 2
        x = tf.keras.layers.Conv2D(filters=self.len_classes, kernel_size=1, strides=1)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        predictions = tf.keras.layers.Activation('softmax')(x)

        self.model = tf.keras.Model(inputs=input, outputs=predictions)

    def train_model(self, x_train, y_train):

        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy, metrics="accuracy")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("\\model.h5", save_best_only=True, verbose=1),
            tf.keras.callbacks.CSVLogger('\\training.log')]

        self.model.fit(x=x_train, y=y_train, batch_size=self.BATCH_SIZE, epochs=30, callbacks=callbacks, verbose=1)


