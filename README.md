
# WELCOME TO GINO'S DATA SCIENCE PROJECT!

# Overview
This is an object localization project based on deep Convolutional Neural Network (CNN). I will explore the state-of-the-art CNN with ResNet, YOLO, and InceptionNet using dataset from [Open Images Challenge 2018](https://storage.googleapis.com/openimages/web/index.html). The goal is to familiarize myself with hyperparameter tuning, network architecture, and data visualization. I will most likely use pre-trained weights to perform "transfered learning" on the output layer.

# Setting up Docker-enabled Google Cloud Platform ([GCP](https://console.cloud.google.com/home/))
* Create VM instances (with preemptibilty to save cost) with Container-Optimized OS (COS) 66-10452.109.0 stable. 
* Add an additional *Standard Persistent Disk* to share with multiple VM by [mounting](https://cloud.google.com/compute/docs/disks/add-persistent-disk#create_disk) the disk `openimage` under the path `/mnt/disks/`.

# Data acquisition
* Download the [Open Image dataset](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/) to the openimage disk by simply using `wget`.

* To `unzip` in COS, I run the docker image from GCP [cos-toolbox](gcr.io/google-containers/toolbox) with `-v` to mount local directory into the `toolbox` container.
(*update: I could've used a OS-based docker image with necessary daemon and pre-installed python version, which is the more direct and compact approach. 
The current approach may save a bit on memory.*)

> docker run --name toolbox -v /mnt/disks/openimage:/mnt -d gcr.io/google-containers/toolbox tail -f /dev/null
* Run an interactive bash session in the `toolbox` container

> docker exec -it toolbox bash

* `unzip` the data in the mounted directory `/mnt` in the container, pointing to the persistent disk /mnt/disks/openimage (mounted on my local VM directory).

> unzip train00.zip

# Setting up Python packages with Docker
*  Create `Dockerfile` to manage the python environment for this project. 

> cat Dockerfile

```
# Use the official Python runtime as a parent image
FROM python:3.5.6-stretch
# Set the working directory to /app in the containerA
WORKDIR /app
# Copy the current local directory contents into the container at /app
ADD . /app
# Install the project related packages
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip2 install --trusted-host pypi.python.org -r requirements.txt
```

> cat requirements.txt

```
opencv-python \ keras \ pandas \ numpy==1.14.5 \ cython \ tensorflow \ matplotlib \ h5py \ scipy \ pydot \ argparse \ cython
```

* Build the `Dockerfile` to produce an image with name `openimage` in the repository `gchen0119` with the tag `python3.5.6-stretch`

> docker build .

> docker tag <IMAGE_ID> gchen0119/openimage:python-3.5.6-stretch

* Login my public registry 

> docker  login --username=gchen0119 \-\-email=gchen0119@gmail.com

* Push the image onto my public registry [gchen0119](https://hub.docker.com/r/gchen0119) 

> docker push gchen0119/python-openimage:python-3.5.6-stretch

* Check out my Random Notes on how to push to GCP `Container Registry`.

# Start Python in the container

* Detach and run the image in the `python-openimage` container

> docker run -v /mnt/disks/openimage:/mnt --name python-openimage -d gchen0119/openimage tail -f /dev/null

* Enter the `python-openimage` container interactively in the bash session

> docker exec -it python-openimage bash

* Start using python in the container and import all the necessary packages



```python
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
from pydarknet import Detector, Image # YOLOv3 package
# Note: one can simply build YOLOv3 related packages with Dockerfile via [madhawav's](https://github.com/madhawav/YOLO3-4-Py/blob/master/docker/).
import cv2

%matplotlib inline
```

* The YOLOv3 is developed by [pjreddie](https://pjreddie.com/darknet/yolo/)
  with [source codes](https://github.com/pjreddie/darknet.git), 

[//]: # "The python wrapper is from [madhawav](https://github.com/madhawav/YOLO3-4-Py) and on [pypi](https://pypi.org/project/yolo34py/#description)."

* Download the YOLOv3 weights 

> wget https://pjreddie.com/media/files/yolov3.weights 

* Use the Keras implementation of YOLOv3 (Tensorflow backend) by
  [qqwweee](https://github.com/qqwweee/keras-yolo3) to build the model 

> python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5




# Recap YOLOv2 and YOLO9000 strengths:

* YOLOv2: 
    * Dimension clusters to find the archor boxes, with distance measure using IoU.  
    * Anchor boxes according to k-means clusters obtain the most frequent aspect ratios for all objects, therefore learning these boxes accelerate the training.
    * Anchor boxes also allow multiple overlapped objects to be detected.
    * Bounding box predictions are constrained in the grid cell by using sigmoid/logistic function ranging between 0 and 1, and accelerating the training.
   
* YOLO9000: Combine YOLOv2 with WordTree!


# YOLOv3 Model Details



# TO BE CONTINUED...
--------------------------------------------------------------------------------------
# Random Notes
* Transferring data between my laptop and the VM instance using the save ssh private key

> scp -i ~/.ssh/google_compute_engine /path/to/file gino@<REMOTE_IP>

where `<REMOTE_IP>` is assigned at the start of a VM instance.

* Transferring data between my laptop and google bucket 

> gcloud compute ssh gchen@<LOCAL_IP> --command='gsutil cp /path/to/file gs://mybucket' 

where `<LOCAL_IP>` can be located by `ifconfig` on macbook or `ip addr show` on linux.

* In order to push the image to [Container Registry](https://cloud.google.com/container-registry/docs/pushing-and-pulling#pushing_an_image_to_a_registry)  hosted by google
on `gcr.io`, first login to google cloud shell and do

> gcloud config set project openimagechallenge 

where `openimagechallenge` is the `Project ID` shown on the console. 

Then login to `gcr.io` to get authorization 

> gcloud auth login

and follow the steps to authenticate.

Then obtain the access-token for docker

> gcloud auth print-access-token

Copy the access-token and go to the COS VM instance

> docker login -u oauth2accesstoken -p "<paste-the-access-token-here\>" https://gcr.io

Finally `push` the image into the google cloud Container Registry (just like Docker Hub)

> docker push <HOSTNAME\>/<PROJECT_ID\>/<IMAGENAME\>:TAG

In my case

> docker push gcr.io/openimageschallenge/openimage:python3.5.6-stretch

Check out the image save on the GCP registry by going the left-hand-side of the panel and click "Contianer Registry".

Note access token changes every time, so always login the cloud shell to `print-access-token`.
