# Fault Diagnosis with Time-Domain Vibration Data

This repository aims to diagnose types of faults based on time-domain vibration data using neural networks. It covers experiments conducted with Temporal Convolution Network-LSTM models.

## Experimental Setup

- 4 DC Motors
  - Normal | Misalignment | Unbalance | Damaged Bearing
- 26.5kHz Sampling Rate
- Data Collection: 102400 samples every 5 seconds

## Dependencies

Version of libraries may not make a problem except tensorflow.

- tensorflow == 2.9.0
- scikit-learn == 1.0.2
- numpy == 1.21.6
- scipy == 1.9.1
- matplotlib == 3.5.2

## Temporal Convolution Network-LSTM

The TCN implementation is based on @philipperemy's work (https://github.com/philipperemy/keras-tcn).

The TCN-LSTM architecture is created by myself.

<p align="center">
  <img src="assets/tcn.png">
  TCN Framework<br>
</p>

- input shape: (, 800)
- Train data: 13639
- Validation data: 3410
- Test data: 1895

### Train Data Structure

- data(`data_path`)
  - label 1
    - data1.csv
    - data2.csv
    - ...
  - label 2
  - label 3
  - label 4

## Experimental results

Even if the accuracy for sampling points is not 100%, with a total of over 100,000 data points available for real-time diagnosis, the final fault diagnosis is determined based on the majority of predictions obtained during real-time diagnosis. In the experimental environment, as mentioned earlier, the test results confirmed successful fault type diagnosis.

## +) 240808

- Apply Normalization with MinMaxScaler()
- It needs one normal status vibration data as csv

### TCN-LSTM

- Test datasets = 1895
- ROC AUC = 0.995
- F1 Score = 0.96328125
- Accuracy = 96.328 %

<p align="center">
  <img src="assets/tcn-output.png">
  <br>TCN Confusion matrix<br>
</p>
