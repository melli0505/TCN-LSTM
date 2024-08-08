from .model import *
from .preprocessing import *

import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


class Trainer:
    def __init__(self, data_path, class_num=4, learning_rate=0.0013) -> None:
        scaler_data_path = input(
            "Enter initial correct data path to normalize training data(csv): "
        )
        self.training_data, self.training_label = get_data(
            data_path, scaler_data_path, training=True
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.training_data, self.training_label, test_size=0.25
        )
        self.label = None

        self.model = TCN_LSTM(
            output_size=class_num, num_channels=[40] * 8, kernel_size=6, dropout=0.05
        )
        self.model.build(input_shape=([1, 800]))

        self.opt = Adam(lr=learning_rate)

    def training(self, resume=False, epochs=1000, batch_size=100, model_name="test"):
        history = {"loss": [], "accuracy": [], "val_loss:": [], "val_acurracy": []}
        h_loss, h_acc, h_val_loss, h_val_acc = [], [], [], []
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
            .shuffle(self.X_train.shape[0])
            .batch(batch_size)
        )
        self.val_data, self.val_labels = tf.convert_to_tensor(
            self.X_val
        ), tf.convert_to_tensor(self.y_val)
        # self.test_data, self.test_labels = tf.convert_to_tensor(
        #     self.X_test
        # ), tf.convert_to_tensor(self.y_test)

        if resume:
            checkpoint_path = input("Enter Checkpoint path: ")
            self.model.load_weights(checkpoint_path)

        clip = -1
        scc_train = tf.keras.losses.SparseCategoricalCrossentropy()
        sca_train = tf.keras.metrics.SparseCategoricalAccuracy()
        scc_test = tf.keras.losses.SparseCategoricalCrossentropy()
        sca_test = tf.keras.metrics.SparseCategoricalAccuracy()

        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.Mean(name="train_acc")
        for epoch in range(epochs):
            for batch, (train_x, train_y) in enumerate(self.train_dataset):
                # loss
                with tf.GradientTape() as tape:
                    y = self.model(train_x, training=True)
                    loss = scc_train(train_y, y)
                acc = sca_train(train_y, y)

                # gradient
                gradient = tape.gradient(loss, self.model.trainable_variables)
                if clip != -1:
                    gradient, _ = tf.clip_by_global_norm(gradient, clip)
                self.opt.apply_gradients(zip(gradient, self.model.trainable_variables))

                train_loss(loss)
                train_accuracy(acc)

            # accuracy = sca_train.result()
            val_y = self.model(self.val_data, training=False)
            val_loss = scc_test(self.val_labels, val_y)
            # sca_test.update_state
            val_acc = sca_test(self.val_labels, val_y)
            print(
                f"Epoch {epoch+1}/{epochs}: \tTrain loss: ",
                train_loss.result().numpy(),
                "\tAccuracy:",
                train_accuracy.result().numpy(),
                "\tVal_loss:",
                val_loss.numpy(),
                "\tVal_accuracy:",
                val_acc.numpy(),
                end="\n",
            )

            h_loss.append(train_loss.result().numpy())
            h_acc.append(train_accuracy.result().numpy())
            h_val_loss.append(val_loss.numpy())
            h_val_acc.append(val_acc.numpy())

            sca_train.reset_state()
            sca_test.reset_state()
            train_loss.reset_state()
            train_accuracy.reset_state()
            if epoch % 100 == 0 or epoch == epochs - 1:
                self.model.save_weights(
                    f"./checkpoints/{model_name}.h5", save_format="h5"
                )

        history = {
            "loss": h_loss,
            "accuracy": h_acc,
            "val_loss": h_val_loss,
            "val_accuracy": h_val_acc,
        }

        pickle.dump(history, open(f"./training_reports/tcn_lstm", "wb"))

    def metrics(self):
        y_pre = self.model.predict(self.X_test)
        y_pre = np.array(y_pre, dtype=np.float32).reshape(
            y_pre.shape[0], self.y_test.shape[1]
        )
        label_pre = np.argmax(y_pre, axis=1)
        label_true = np.argmax(self.y_test, axis=1)
        confusion_mat = metrics.confusion_matrix(label_true, label_pre)
        self.plot_confusion_matrix(
            confusion_mat,
            classes=["Normal", "Misalignment", "Unbalance", "Damaged Bearing"],
        )

        Accuracy = metrics.accuracy_score(label_true, label_pre)
        F1_score = metrics.f1_score(label_true, label_pre, average="micro")
        probs = y_pre
        lr_auc = metrics.roc_auc_score(self.y_test, probs, multi_class="ovr")

        print("No. of Samples =", 800, " |   Test datasets =", len(self.X_test))
        print("ROC AUC = %.3f" % (lr_auc))
        print("F1 Score =", F1_score)
        print("Accuracy = %.3f" % (Accuracy * 100), "%")

    def plot_confusion_matrix(
        self, cm, classes, title="Confusion matrix", cmap=plt.cm.Blues, normalize=False
    ):
        plt.imshow(cm, cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_mark = np.arange(len(classes))
        plt.xticks(tick_mark, classes, rotation=40)
        plt.yticks(tick_mark, classes)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = "%.2f" % cm
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="black")
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predict label")
