# Gesture Classification of sEMG Signals for Prosthesis Control 

A simple Support Vector Classifier for the following dataset: https://www.kaggle.com/kyr7plus/emg-4

This report attempts to define a supervised machine learning model called a Support Vector Classifier to accurately classify gestures based on sEMG signals. The classifier was trained on a labelled dataset consisting of four classes. Each class represents a hand gesture (rock, paper, scissors, ok). The full report can be viewed [here](https://github.com/tmastrom/GestureSVC/blob/master/Final%20Report.pdf).

The time series sEMG data was preprocessed using the Daubechies discrete wavelet transform to characterize the waveform before fitting the data to the classifier. After fitting the model and optimizing the parameters, 92% accurate classification was achieved on the test dataset. This is on par with current prosthetic device classification accuracy. Further testing is required to determine if the program can be executed in real-time (less than 75ms) on a typical embedded processor that would be used in a prosthesis today.
