# Brain Computer Interface AI Model

AI model for a Brain Computer Interface. Detects left/right imagined movement from
3 EEG electrodes (C3, C4, Cz). Applies a convolutional network
with temporal filters, then spatial filters (across electrodes),
followed by another convolutional layer, and dense layer. 
Validation accuracy ~76%. 

Based on approach in https://arxiv.org/pdf/1611.08024. <br>
Training data: https://www.bbci.de/competition/iv/desc_2b.pdf

Package install
```
pip install -r requirements.txt
````

## Inference

EEG data should be a numpy array of size [3, 384] corresponding to
channels (C3, C4, Cz) x time series (3s of data sampled at 128 Hz, 
bandpass 4-50 Hz, in volts). EEG data epoch should start sync'd to 
stimulus trigger (tone 1 kHz, 70 ms plus prompt, e.g.  "Think left or right")

```
from inference import EEGInferenceApp

app = EEGInferenceApp("data/checkpoint.pt")
movement = app.predict_imagined_movement(data)

if movement == EEGInferenceApp.LEFT:
    print("You imagined left")
elif movement == EEGInferenceApp.RIGHT:
    print("You imagined right")
```

## Training

```
python training.py
```

Data preprocessed into data/preprocessed_data.pt <br>
Checkpoint saved into data/checkpoint.pt
