# setup
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
from keras_self_attention import SeqSelfAttention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--WEIGHT') # weight 경로
    parser.add_argument('--TEST_IMAGE_DIR') # test데이터 입력 경로
    parser.add_argument('--RESULTS_DIR') # test결과 출력 경로 

    args = parser.parse_args()
    WEIGHT = args.WEIGHT
    TEST_IMAGE_DIR = args.TEST_IMAGE_DIR
    RESULTS_DIR = args.RESULTS_DIR

    if WEIGHT:
        reconstructed_model = keras.models.load_model(WEIGHT, custom_objects={'CTCLayer': CTCLayer})
        
        test_path = TEST_IMAGE_DIR
        list_test_png = [os.path.join(test_path, image) for image in os.listdir(test_path) if image.endswith('png')]

        test_images = sorted(list(map(str, list_test_png)))
        test_labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in test_images]

        x_test = np.array(test_images)
        y_test = np.array(test_labels)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = (
            test_dataset.map(
                encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        def split_data(images, labels, train_size=0.9, shuffle=True):
            # 1. Get the total size of the dataset
            size = len(images)
            # 2. Make an indices array and shuffle it, if required
            indices = np.arange(size)
            if shuffle:
                np.random.shuffle(indices)
            # 3. Get the size of training samples
            train_samples = int(size * train_size)
            # 4. Split data into training and validation sets
            x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
            x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
            return x_train, x_valid, y_train, y_valid

        def encode_single_sample(img_path, label):
            # 1. Read image
            img = tf.io.read_file(img_path)
            # 2. Decode and convert to grayscale
            img = tf.io.decode_png(img, channels=1)
            # 3. Convert to float32 in [0, 1] range
            img = tf.image.convert_image_dtype(img, tf.float32)
            # 4. Resize to the desired size
            img = tf.image.resize(img, [img_height, img_width])
            # 5. Transpose the image because we want the time
            # dimension to correspond to the width of the image.
            img = tf.transpose(img, perm=[1, 0, 2])
            # 6. Map the characters in label to numbers
            label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
            # 7. Return a dict as our model is expecting two inputs
            return {"image": img, "label": label}

        class CTCLayer(layers.Layer):
            def __init__(self, name=None, **kwargs):
                super().__init__(name=name)
                self.loss_fn = keras.backend.ctc_batch_cost

            def call(self, y_true, y_pred):
                # Compute the training-time loss value and add it
                # to the layer using `self.add_loss()`.
                batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
                input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
                label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

                input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
                label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

                loss = self.loss_fn(y_true, y_pred, input_length, label_length)
                self.add_loss(loss)

                # At test time, just return the computed predictions
                return y_pred


        def build_model():
            # Inputs to the model
            input_img = layers.Input(
                shape=(img_width, img_height, 1), name="image", dtype="float32"
            )
            labels = layers.Input(name="label", shape=(None,), dtype="float32")

            # First conv block
            x = layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                padding="same",
                name="Conv1",
            )(input_img)
            x = layers.MaxPooling2D((2, 2), name="pool1")(x)

            # Second conv block
            x = layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                padding="same",
                name="Conv2",
            )(x)
            x = layers.MaxPooling2D((2, 2), name="pool2")(x)

            # We have used two max pool with pool size and strides 2.
            # Hence, downsampled feature maps are 4x smaller. The number of
            # filters in the last layer is 64. Reshape accordingly before
            # passing the output to the RNN part of the model
            new_shape = ((img_width // 4), (img_height // 4) * 64)
            x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
            x = layers.Dense(64, activation="relu", name="dense1")(x)
            x = layers.Dropout(0.2)(x)

            # RNNs
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
            x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

            # Output layer
            x = layers.Dense(
                len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
            )(x)

            # Add CTC layer for calculating CTC loss at each step
            output = CTCLayer(name="ctc_loss")(labels, x)

            # Define the model
            model = keras.models.Model(
                inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
            )
            # Optimizer
            opt = keras.optimizers.Adam()
            # Compile the model and return
            model.compile(optimizer=opt)
            return model

        # Get the prediction model by extracting layers till the output layer
        prediction_model = keras.models.Model(
            reconstructed_model.get_layer(name="image").input, reconstructed_model.get_layer(name="dense2").output
        )
        def decode_batch_predictions(pred):
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            # Use greedy search. For complex tasks, you can use beam search
            results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                :, :max_length
            ]
            # Iterate over the results and get back the text
            output_text = []
            for res in results:
                res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
                output_text.append(res)
            return output_text

        for batch in test_dataset.take(101):
            batch_images = batch["image"]
            batch_labels = batch["label"]
            preds = prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds)
            plt.imshow(np.squeeze(np.array(batch_images)).transpose())
            plt.title(pred_texts)
            plt.savefig(os.path.join(RESULTS_DIR,'{}.png'.format(pred_texts[0])))
            print(pred_texts, 'saved')
            plt.close()
        
        print('finished')

    else:
        path = r'/home/test/captcha'
        list_png = [os.path.join(path, image) for image in os.listdir(path) if image.endswith('png')]
        print('take png files for training..')
        print('\n')

        test_path = TEST_IMAGE_DIR
        list_test_png = [os.path.join(test_path, image) for image in os.listdir(test_path) if image.endswith('png')]
        print('take png files for test..')
        print('\n')

        # Path to the data directory
        data_dir = path

        # Get list of all the images
        images = sorted(list(map(str, list_png)))
        labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
        test_images = sorted(list(map(str, list_test_png)))
        test_labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in test_images]

        characters = set(char for label in labels for char in label)

        print("Number of images found: ", len(images))
        print("Number of labels found: ", len(labels))
        print("Number of unique characters: ", len(characters))
        print("Characters present: ", characters)
        print('\n')

        # Batch size for training and validation
        batch_size = 1

        # Desired image dimensions
        img_width = 200
        img_height = 50

        # Factor by which the image is going to be downsampled
        # by the convolutional blocks. We will be using two
        # convolution blocks and each block will have
        # a pooling layer which downsample the features by a factor of 2.
        # Hence total downsampling factor would be 4.
        downsample_factor = 4

        # Maximum length of any captcha in the dataset
        max_length = max([len(label) for label in labels])


        # Mapping characters to integers
        char_to_num = layers.StringLookup(
            vocabulary=list(characters), mask_token=None
        )

        # Mapping integers back to original characters
        num_to_char = layers.StringLookup(
            vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
        )


        def split_data(images, labels, train_size=0.9, shuffle=True):
            # 1. Get the total size of the dataset
            size = len(images)
            # 2. Make an indices array and shuffle it, if required
            indices = np.arange(size)
            if shuffle:
                np.random.shuffle(indices)
            # 3. Get the size of training samples
            train_samples = int(size * train_size)
            # 4. Split data into training and validation sets
            x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
            x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
            return x_train, x_valid, y_train, y_valid


        # Splitting data into training and validation sets
        x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
        x_test = np.array(test_images)
        y_test = np.array(test_labels)

        def encode_single_sample(img_path, label):
            # 1. Read image
            img = tf.io.read_file(img_path)
            # 2. Decode and convert to grayscale
            img = tf.io.decode_png(img, channels=1)
            # 3. Convert to float32 in [0, 1] range
            img = tf.image.convert_image_dtype(img, tf.float32)
            # 4. Resize to the desired size
            img = tf.image.resize(img, [img_height, img_width])
            # 5. Transpose the image because we want the time
            # dimension to correspond to the width of the image.
            img = tf.transpose(img, perm=[1, 0, 2])
            # 6. Map the characters in label to numbers
            label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
            # 7. Return a dict as our model is expecting two inputs
            return {"image": img, "label": label}



        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset.map(
                encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset.map(
                encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = (
            test_dataset.map(
                encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        class CTCLayer(layers.Layer):
            def __init__(self, name=None, **kwargs):
                super().__init__(name=name)
                self.loss_fn = keras.backend.ctc_batch_cost

            def call(self, y_true, y_pred):
                # Compute the training-time loss value and add it
                # to the layer using `self.add_loss()`.
                batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
                input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
                label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

                input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
                label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

                loss = self.loss_fn(y_true, y_pred, input_length, label_length)
                self.add_loss(loss)

                # At test time, just return the computed predictions
                return y_pred

        def build_model():
            # Inputs to the model
            input_img = layers.Input(
                shape=(img_width, img_height, 1), name="image", dtype="float32"
            )
            labels = layers.Input(name="label", shape=(None,), dtype="float32")

            # First conv block
            x = layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                padding="same",
                name="Conv1",
            )(input_img)
            x = layers.MaxPooling2D((2, 2), name="pool1")(x)

            # Second conv block
            x = layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                padding="same",
                name="Conv2",
            )(x)
            x = layers.MaxPooling2D((2, 2), name="pool2")(x)

            # We have used two max pool with pool size and strides 2.
            # Hence, downsampled feature maps are 4x smaller. The number of
            # filters in the last layer is 64. Reshape accordingly before
            # passing the output to the RNN part of the model
            new_shape = ((img_width // 4), (img_height // 4) * 64)
            x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
            x = layers.Dense(64, activation="relu", name="dense1")(x)
            x = layers.Dropout(0.2)(x)

            # RNNs
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
            x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

            # Output layer
            x = layers.Dense(
                len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
            )(x)

            # Add CTC layer for calculating CTC loss at each step
            output = CTCLayer(name="ctc_loss")(labels, x)

            # Define the model
            model = keras.models.Model(
                inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
            )
            # Optimizer
            opt = keras.optimizers.Adam()
            # Compile the model and return
            model.compile(optimizer=opt)
            return model


        # Get the model
        model = build_model()

        # Training
        epochs = 100
        early_stopping_patience = 10
        # Add early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_acc", patience=early_stopping_patience, restore_best_weights=True
        )

        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=[early_stopping],
        )
        
        # Get the prediction model by extracting layers till the output layer
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )
        def decode_batch_predictions(pred):
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            # Use greedy search. For complex tasks, you can use beam search
            results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                :, :max_length
            ]
            # Iterate over the results and get back the text
            output_text = []
            for res in results:
                res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
                output_text.append(res)
            return output_text

        for batch in test_dataset.take(101):
            batch_images = batch["image"]
            batch_labels = batch["label"]
            preds = prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds)
            plt.imshow(np.squeeze(np.array(batch_images)).transpose())
            plt.title(pred_texts)
            plt.savefig(os.path.join(RESULTS_DIR,'{}.png'.format(pred_texts[0])))
            print(pred_texts, ' saved')
            plt.close()
        
        print('finished')

if __name__ == "__main__":
    main()

exit()
