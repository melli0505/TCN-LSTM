from .model import *
from .preprocessing import *

import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import metrics


class Trainer:
    def __init__(self, data_path, class_num=4, learning_rate=0.0013) -> None:
        self.training_data, self.training_label = get_data(data_path, training=True)
        self.label = None
        
        self.model = TCN_LSTM(output_size=class_num, num_channels=[40] * 8, kernel_size=6, dropout=0.05)
        self.model.build(input_shape=([1, 800]))

        self.opt = Adam(lr=learning_rate)

    def training(self, resume=False, epochs=1000, batch_size=100):
        X, X_test, y, y_test = train_test_split(self.training_data, self.training_label, test_size=0.1)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(batch_size)
        self.val_data, self.val_labels = tf.convert_to_tensor(X_val), tf.convert_to_tensor(y_val)
        self.test_data, self.test_labels = tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test)

        if resume:
            checkpoint_path = input("Enter Checkpoint path: ")
            self.model.load_weights(checkpoint_path)
        
        clip = -1

        for epoch in range(epochs):
            for batch, (train_x, train_y) in enumerate(self.train_dataset):
                # loss
                with tf.GradientTape() as tape:
                    y = self.model(train_x, training=True)
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=y))
                # gradient
                gradient = tape.gradient(loss, self.model.trainable_variables)
                if clip != -1:
                    gradient, _ = tf.clip_by_global_norm(gradient, clip)  
                self.opt.apply_gradients(zip(gradient, self.model.trainable_variables))
                if batch % 99 == 0:
                    print("- Train loss:", loss.numpy(), end="\n")

            # if epoch % 200 == 0 or epoch == epochs-1:
            #     self.model.save_weights(f"./checkpoints/TCN_LSTM_{epoch}.h5", save_format="h5")

            self.model.save_weights("./checkpoints/tcn_lstm_recent.h5", save_format="h5")

    def metrics(self):
        y_pre = self.model.predict(self.X_test)
        y_pre = np.array(y_pre, dtype=np.float32).reshape(y_pre.shape[0], self.y_test.shape[1])
        label_pre = np.argmax(y_pre, axis=1)
        label_true = np.argmax(self.y_test, axis=1)
        confusion_mat = metrics.confusion_matrix(label_true, label_pre)
        metrics.plot_confusion_matrix(confusion_mat, classes=["Normal", "Misalignment", "Unbalance", "Damaged Bearing"])

        Accuracy = metrics.accuracy_score(label_true, label_pre)
        F1_score = metrics.f1_score(label_true, label_pre, average='micro')
        probs = y_pre
        lr_auc = metrics.roc_auc_score(self.y_test, probs, multi_class='ovr')

        print("No. of Samples =", 800, " |   Test datasets =", len(self.X_test))
        print('ROC AUC = %.3f' % (lr_auc))
        print("F1 Score =", F1_score)
        print("Accuracy = %.3f" % (Accuracy*100), "%")

    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, normalize=False):
        plt.imshow(cm , cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_mark = np.arange(len(classes))
        plt.xticks(tick_mark, classes, rotation=40)
        plt.yticks(tick_mark, classes)
        if normalize:
            cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm = '%.2f'%cm
        thresh = cm.max()/2.0
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j,i,cm[i,j], horizontalalignment='center',color='black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predict label')