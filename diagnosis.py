from .model import *
from .preprocessing import *
import numpy as np


class Diagnosis:
    """
    Fault Diagnosis Class
    """

    def __init__(self) -> None:
        self.model = TCN_LSTM(
            output_size=4, num_channels=[40] * 8, kernel_size=6, dropout=0.05
        )
        self.model.build(input_shape=([1, 800]))
        self.model.load_weights("./checkpoints/tcn_lstm.h5")

    def diagnosis(self, data_path):
        """
        Motor fault diagnosis based on pretrained MCNN-LSTM

        Args:
            motor (int): motor number which gonna diagnose status
        Returns:
            status (str): result of deep learning diagnosis
        """

        # get current data
        data = get_data(data_path)

        # get diagnosis result
        y_pre = self.model(data, training=False)
        y_pre = np.array(y_pre, dtype=np.float32).reshape(y_pre.shape[0], 4)
        label_pre = np.argmax(y_pre, axis=1)
        predicted = [
            len(np.where(label_pre == 0)[0]),
            len(np.where(label_pre == 1)[0]),
            len(np.where(label_pre == 2)[0]),
            len(np.where(label_pre == 3)[0]),
        ]
        print("[WARN] Predicted rate(N-M-U-B):", predicted)
        result = predicted.index(max(predicted))

        if max(predicted) < len(label_pre) // 2 + 5:
            return "Normal"

        if result == 0:
            return "Normal"
        elif result == 1:
            return "Misalignment"
        elif result == 2:
            return "Unbalance"
        elif result == 3:
            return "Damaged Bearing"


if __name__ == "__main__":
    tcn_lstm = Diagnosis()
    current_data_dir = input("Enter path of current vibration data directory: ")
    result = tcn_lstm.diagnosis(current_data_dir)
