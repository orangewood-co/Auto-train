# AutoTrain
**Auto training pipeline for object detction models**

This pipeline trains object detection model(YOLOv8) using real time inference data. It is for automating the supervised learning, specifically cutting out the manual labelling task and training the model for it to remember the object as per the label we want.

## Requirements
* Python >= 3.8
* [Libraries](requirements.txt)
* CUDA (optional but prefferable)

## Setup
* Make sure to have python version 3.8 or above
* Install the required dependencies
```
pip install -r requirements.txt
```
* Add your camera source to the system, and make sure it is free to use
* Run auto_train.py to launch the pipeline
```
python3 auto_train.py --new_weights --roboflow
```
Get the arguments to be parsed using
```
python3 auto_train.py -h
```
* **Input the object you want to detect**, *give in generic prompt that describes the object you want to train on*
* **Input the label to give to object**, *the name by which it should be detected*
* Camera initialises, take the frames from different angles
*You can monitor the video feed to check if the object is being correctly labelled*
* Once reached the cap of image_threshold provided, camera stops and starts training on the captured labelled data
* You have the new YOLOv8 weights file

### Capabilities
* Creates new annotated data 
* Combines previous annotated data with newly captured
* Annotates the data with visible bounding boxes, given images and corresponding v8 txt label files
* Create new weights (pt) file for given dataset
* Creates analysis graphs and metrices for validation

### To-dos
* Add multiple objects annotation in single frame
* RnD on Florence capabilities for giving in text prompt to ZSD model
* RnD on discarding faulty annotations from procured dataset