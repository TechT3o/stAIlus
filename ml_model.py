import tensorflow as tf
from sklearn.svm import SVC
from data_processing import DataProcessor
from statics import add_noise, time_stretch, time_shift
from tqdm import tqdm

class MLModel:
    def __init__(self, data_path):


        self.dropout_rate = 0.2
        self.len_classes = 3
        self.model = tf.keras.Model
        self.BATCH_SIZE = 1
        self.processor = DataProcessor(data_path)

        #self.processor.pre_process_spectrogram()
        # self.X_train, self.X_test, self.y_train, self.y_test = self.processor.train_test_split_data()
        # self.X_train, self.X_test, self.y_train, self.y_test = self.processor.augmented_train_test_split_data()

    def augmentation(self):
        self.processor.augment_data()
        # self.processor.pre_process_representation_ensemble()
        self.processor.pre_process_augmented_ensemble()
        self.split_augmented_data()

    def normal(self):
        self.processor.pre_process_representation_ensemble()
        self.split_data()

    def fcn_model(self):
        # Input layer
        input = tf.keras.layers.Input(shape=(None, None, 1))

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
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)

        x = tf.keras.layers.Dense(self.len_classes)(x)
        predictions = tf.keras.layers.Activation('softmax')(x)

        self.model = tf.keras.Model(inputs=input, outputs=predictions)

    def double_input_model(self, input_shape):
        # Input layer
        input_1 = tf.keras.layers.Input(shape=input_shape)
        input_2 = tf.keras.layers.Input(shape=(1))
        # A convolution block
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(input_1)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

        # Fully connected layer 2
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

        x = tf.keras.layers.Flatten()(x)
        concatted = tf.keras.layers.Concatenate()([x, input_2])
        x = tf.keras.layers.Dense(254)(concatted)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(self.len_classes)(x)
        predictions = tf.keras.layers.Activation('softmax')(x)

        self.model = tf.keras.Model(inputs=[input_1, input_2], outputs=predictions)

    def train_model(self, x_train, y_train, x_val = None, y_val = None):
        print(self.model.summary())
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy(), metrics="accuracy")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("\\model.h5", save_best_only=True, verbose=1),
            tf.keras.callbacks.CSVLogger('\\training.log')]

        self.model.fit(x=x_train, y=y_train, validation_data= (x_val, y_val), batch_size=self.BATCH_SIZE, epochs=30, callbacks=callbacks, verbose=1)
        # else:
        #     self.model.fit(x=x_train, y=y_train, batch_size=self.BATCH_SIZE, epochs=30, callbacks=callbacks, verbose=1)

    def test_model(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        print(y_pred,y_test)
    # def CTCLoss(self, y_true, y_pred):
    #     # Compute the training-time loss value
    #     batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    #     input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    #     label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    #
    #     input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    #     label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    #
    #     loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    #     return loss
    #
    # def build_model(self, input_dim, output_dim, rnn_layers=5, rnn_units=128):
    #     """Model similar to DeepSpeech2."""
    #     # Model's input
    #     input_spectrogram = tf.keras.layers.Input((None, input_dim), name="input")
    #     # Expand the dimension to use 2D CNN.
    #     x = tf.keras.layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    #     # Convolution layer 1
    #     x = tf.keras.layers.Conv2D(
    #         filters=32,
    #         kernel_size=[11, 41],
    #         strides=[2, 2],
    #         padding="same",
    #         use_bias=False,
    #         name="conv_1",
    #     )(x)
    #     x = tf.keras.layers.BatchNormalization(name="conv_1_bn")(x)
    #     x = tf.keras.layers.ReLU(name="conv_1_relu")(x)
    #     # Convolution layer 2
    #     x = tf.keras.layers.Conv2D(
    #         filters=32,
    #         kernel_size=[11, 21],
    #         strides=[1, 2],
    #         padding="same",
    #         use_bias=False,
    #         name="conv_2",
    #     )(x)
    #     x = tf.keras.layers.BatchNormalization(name="conv_2_bn")(x)
    #     x = tf.keras.layers.ReLU(name="conv_2_relu")(x)
    #     # Reshape the resulted volume to feed the RNNs layers
    #     x = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    #     # RNN layers
    #     for i in range(1, rnn_layers + 1):
    #         recurrent = tf.keras.layers.GRU(
    #             units=rnn_units,
    #             activation="tanh",
    #             recurrent_activation="sigmoid",
    #             use_bias=True,
    #             return_sequences=True,
    #             reset_after=True,
    #             name=f"gru_{i}",
    #         )
    #         x = tf.keras.layers.Bidirectional(
    #             recurrent, name=f"bidirectional_{i}", merge_mode="concat"
    #         )(x)
    #         if i < rnn_layers:
    #             x = tf.keras.layers.Dropout(rate=0.5)(x)
    #     # Dense layer
    #     x = tf.keras.layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    #     x = tf.keras.layers.ReLU(name="dense_1_relu")(x)
    #     x = tf.keras.layers.Dropout(rate=0.5)(x)
    #     # Classification layer
    #     output = tf.keras.layers.Dense(units=output_dim + 1, activation="softmax")(x)
    #     # Model
    #     model = tf.keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    #     # Optimizer
    #     opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    #     # Compile the model and return
    #     model.compile(optimizer=opt, loss=self.CTCLoss)
    #     return model
    #
    # def decode_batch_predictions(self, pred):
    #     input_len = np.ones(pred.shape[0]) * pred.shape[1]
    #     # Use greedy search. For complex tasks, you can use beam search
    #     results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    #     # Iterate over the results and get back the text
    #     output_text = []
    #     for result in results:
    #         result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
    #         output_text.append(result)
    #     return output_text

    def build_train_test_svm(self):
        model = SVC(kernel='linear', probability= True)
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_test, self.y_test)
        print(score)
        #print(model.predict_proba(self.X_test))
        return model

    def build_artificial_dataset(self, num_samples: int):


        features_dataset = tf.data.Dataset.from_tensor_slices(self.processor.processed_data)
        labels_dataset = tf.data.Dataset.from_tensor_slices(self.processor.labels)
        signal_lengths_dataset = tf.data.Dataset.from_tensor_slices(self.processor.signal_lengths)

        dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

        return dataset

    def split_augmented_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.processor.augmented_train_test_split_data()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.processor.train_test_split_data()
