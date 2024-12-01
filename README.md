# Brain Computer Interface

Brain Computer Interface for left/right imagined movement from
3 EEG electrodes (C3, Cz, C4). Applies a convolutional network
with temporal filters, then spatial filters (across electrodes)
followed by another convolutional network and dense layer. 
Validation accuracy ~80%. 

Based on approach in https://arxiv.org/pdf/1611.08024. <br>
Training data: https://www.bbci.de/competition/iv/desc_2b.pdf

## Training

```
python training.py
```

Data preprocessed into data/preprocessed_data.pt <br>
Checkpoint saved into data/checkpoint.pt

## Inference

```
app = EEGInferenceApp()
```

EEG data should be a numpy array of size [3, 256] corresponding to
channels (C3, Cz, C4) x time series (2s of data sampled at 128 Hz, 
bandpass 4-50 Hz, uV scale). EEG data epoch should start sync'd to 
stimulus trigger (tone 1 kHz, 70 ms plus prompt "Think left or right")

```
if app.predict_imagined_movement(data) == EEGInferenceApp.LEFT:
    print("You imagined left")
elif app.predict_imagined_movement(data) == EEGInferenceApp.RIGHT:
    print("You imagined right")
```
