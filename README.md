# Object Detection in an Urban Environment

## Project overview

Goal of the project is, to apply the skills gained in the computer vision course, which is part of the Self Driving Car Engineer Nanodegree program. Object detection is useful in autonomous vehicle as camera are relatively cheap sensor with a high resoluion, which allows detection of mupltiple object of different sizes.
A convolutional neural network will be used to detect and classify objects originating from [Waymo Open dataset](https://waymo.com/open/). Specifically, the classes of object shall be detected in image taken by the front camera of a vehicle: cyclists, pedestrians and vehicles.

In the first step, an exploratory data analysis will be performed to gain some knowlegde about the data set. Then a pretrained Resnet50 model will be trained on the training data, validated and hyperparameters tuned if needed.

## Structure

### Data

The data files can be downloaded directly from the Waymo website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records.
The Udacity project workspace has been used for the project. The workspace has the data already readily available.

The data in the workspace was already partitioned into 3 splits:<br>
/home/workspace/home/data/train - 86 files<br>
/home/workspace/home/data/val - 10 files<br>
/home/workspace/home/data/test - 3 files<br>


### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment1 .. 4/ - experiments with different parameters
    - final/ - final training of the model
    - label_map.pbtxt
```

## Prerequisites

Workspace has been used as provided, no changes where made. Udacity provided information about the requirments for running the scripts on a machine other than the workspace, those infos can be found in folder "build".

## Exploratory Data Analysis
### Preview of images
The following images are randomly picked from the data set. The image title is the brightness of the image, the intention is to get an idea how the brightness value maps to an image so the histogram in the following analysis can be interpreted.<br>
Object classes of the boundig boxes are color coded:
|Object class|Color|
|------------|-------|
|Vehicles|red|
|Pedestrian|green|
|Cyclists|blue|

|<img src="RawImages/0.png" width="300" height="300">|<img src="RawImages/1.png" width="300" height="300">|<img src="RawImages/2.png" width="300" height="300">|
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="RawImages/3.png" width="300" height="300">|<img src="RawImages/4.png" width="300" height="300">|<img src="RawImages/5.png" width="300" height="300">|
|<img src="RawImages/6.png" width="300" height="300">|<img src="RawImages/7.png" width="300" height="300">|<img src="RawImages/8.png" width="300" height="300">|

### Plan for further data analysis
The training and validation data set shall be compared.<br> 
The following stats shall be calculated for every frame in the datasets:
- Number of objects
- Number of vehicles
- Number of pedestrians
- Number of cyclist
- Brightness of image

### Histograms
#### Number of objects
![local image](EDA/HistNumObj.png)
#### Number of vehicles
![local image](EDA/HistNumVeh.png)
#### Number of pedestrians
![local image](EDA/HistNumPed.png)
#### Number of cyclist
![local image](EDA/HistNumCyc.png)
#### Brightness of image
![local image](EDA/HistBrightness.png)
### EDA Summary
The training data set contains 1719 and the validation data set 198 individual pictures. According to the histogram plot as well as to the median in the descriptive statistics, the validation data set generally contains more objects per frame compared with the training data set.
In both data set, there are almost no cyclists. Most of the objects are vehicles.
The average image brightness is quite compareable in both data sets, though the validation set does not contain really dim images.

## Training and evalutation of the object detection algorithm
### Reference experiment
A reference experiment has been provided which has been used as a baseline for further experiments. More information on the 'Single Shot Detector' can be found here [here](https://arxiv.org/pdf/1512.02325.pdf).<br>
The reference uses a pretrained SSD Resnet 50 640x640 model, which was also used for all further experiments.<br>
Step1: Download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `experiments/pretrained_model/`.<br>
Step2: Edit the config file by running `python edit_config.py --train_dir data/train/ --eval_dir data/val/ --batch_size 2 --checkpoint experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map experiments/label_map.pbtxt`<br>
Step3: Move the config file `pipeline_new.config` to folder `experiments/reference`<br>
Step4: Launch training process by running `python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config`<br>
Step5: Launch evaluation process by running `python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/`<br>
Step6: Check training and evaluation by running `python -m tensorboard.main --logdir experiments/reference/`<br>
### Improve the performances
In order to improve the object detection performance, five different configuration changes have been tested. The pretrained model (SSD Resnet 50 640x640) was used for all experiments.<br>
|Name|Change compared to reference|
|---|----|
|experiment1|learning_rate_base changed from 0.04 to 0.08|
|experiment2|augmentation added: random_adjust_brightness, random_patch_gaussian, learning_rate_base: 0.04|
|experiment3|optimizer: adam instead of momentum, learning_rate_base=0.04|
|experiment4|optimizer adam, learning_rate_base=0.0004|
|final|optimizer adam, manual_step_learning_rate scheme used starting at initial_learning_rate=0.0002|
### Performance of the experiments
#### Metrics
![local image](experiments/Tensorboard/TensorBoardLoss.png)
aaaa
![local image](experiments/Tensorboard/TensorBoardPrecision.png)
aaaa
![local image](experiments/Tensorboard/TensorBoardRecall.png)



##### Comparison with ground truth
The picture respresent a side-by-side comparison of the objects detected by the reference model and the ground truth. It can be seen, that only one vehicle is detected with a medium probabilty.
![local image](experiments/reference/ref.png)




Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
