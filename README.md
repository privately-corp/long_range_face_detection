# Long range face detection model (TF Lite)
This repository provides an implementation of RetinaFace architecture with MobileNet0.25 as backbone architecture. It contains the pre-processing and post processing script to integrate the TF lite model

## Convert Pytorch Models to tflite
    [AI Edge Torch](https://ai.google.dev/edge/lite/models/convert_pytorch)

## Steps to create the virtual enivronment
1. Create and activate the conda environment
```
conda create -n lr_face python=3.11
conda activate lr_face
```
2. Install the required packages
```
pip install -r requirements.txt
```

## Tflite inference of Retina face model
Download the Tflite model from s3 bucket and keep it on the parent directory.

Tflite model :```s3://ml-models-production/face_landmarks_models/RetinaFace/tflite_models/RetinaFace_mobilenet0.25_640.tflite```

Raw pytorch weights : ```s3://ml-models-production/face_landmarks_models/RetinaFace/rawWeights/```

```
python tflite_inference.py
```