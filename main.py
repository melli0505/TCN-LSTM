from .trainer import Trainer

data_path = input("Enter data path: ")
learning_rate = float(input("Enter learning rate for TCN-LSTM(default: 0.0013): "))
class_num = int(input("Enter class number of your dataset: "))
resume = True if input("Training resume? (T/F): ") == "T" else False
epochs = int(input("Enter Training epochs(default: 1000): "))
batch_size = int(input("Enter Batch size(default: 100): "))

trainer = Trainer(data_path=data_path, class_num=class_num, learning_rate=learning_rate)
model = trainer.training(resume=resume, epochs=epochs, batch_size=batch_size)
trainer.metrics()