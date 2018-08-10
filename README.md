
# WELCOME TO GINO'S DATA SCIENCE PROJECT!

# Overview
This is an object recognization project based on deep Convolutional Neural Network (CNN). I will explore the state-of-the-art CNN with ResNet, YOLO, and InceptionNet using dataset from [Open Images Challenge 2018](https://storage.googleapis.com/openimages/web/index.html). The goal is to familiarize myself with hyperparameter tuning, network architecture, and data visualization. I will most likely use pre-trained weights to perform "transfered learning".

# Setting up Google Cloud Platform (GCP)
This section is about how I set up the platform to enable the use of docker container for this project. 
* Create VM instances (with preemptibilty to save cost) with Container-Optimized OS (COS) 66-10452.109.0 stable. 
* Add an additional Standard Persistent Disk (500GB) to share with multiple VM by [mounting](https://cloud.google.com/compute/docs/disks/add-persistent-disk#create_disk) the disk `openimage` under the path `/mnt/disks/`.

# Downloading data 
* Download the [Open Image dataset](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/) to the openimage disk by simply using `wget`.

* To `unzip` in COS, I run the docker image from GCP [cos-toolbox](gcr.io/google-containers/toolbox) with `-v` to mount local directory into the `toolbox` container.

> docker run --name toolbox -v /mnt/disks/openimage:/mnt -d gcr.io/google-containers/toolbox tail -f /dev/null

* Run an interactive bash session in the `toolbox` container

> docker exec -it toolbox bash

* `unzip` the data in the mounted directory `/mnt` in the container, pointing to the 500GB disk /mnt/disks/openimage (mounted on my local VM directory).

> unzip train00.zip

# Setting up Python packages using Docker
*  Create `Dockerfile` to manage the python environment for this project. 

> cat Dockerfile

```
# Use the official Python runtime as a parent image
FROM python:2.7-stretch
# Install the project related packages
RUN pip2 install -r requirements.txt
```

> cat requirements.txt

```
pandas \ tensorflow \ numpy \ matplotlib \ h5py \ keras \ scipy \ pydot \ argparse \ yolo_utils \ yad2k \ cv2 \ fr_utils \ inception_blocks_v2
```

* Build the `Dockerfile` to produce an image and call the repository `python-openimage` with the tag `latest`

> docker build .

> docker tag <IMAGE_ID> gchen0119/python-openimage:lastest 

* Login my public registry 

> docker  login --username=gchen0119 \-\-email=gchen0119@gmail.com

* Tag the image with `python-openimage` and a subtag `latest` in my public docker repository [gchen0119](https://hub.docker.com/r/gchen0119) 

> docker tag <IMAGE_ID> gchen0119/python-openimage:latest

* Push the image onto my public registry

> docker push gchen0119/python-openimage

# Start Python in the container

* Detach and run the image in the `python-openimage` container

> docker run -v /mnt/disks/openimage:/mnt --name python-openimage -d gchen0119/python-openimage tail -f /dev/null

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
import cv2
```

# TO BE CONTINUED...
--------------------------------------------------------------------------------------
# Random Notes
* Transferring data between my laptop and the VM instance using the save ssh private key

> scp -i ~/.ssh/google_compute_engine /path/to/file gino@<REMOTE_IP>

where `<REMOTE_IP>` is assigned at the start of a VM instance.

* Transferring data between my laptop and google bucket 

> gcloud compute ssh gchen@<LOCAL_IP> --command='gsutil cp /path/to/file gs://mybucket' 

where `<LOCAL_IP>` can be located by `ifconfig` on macbook or `ip addr show` on linux.

* [Push to Container Registry hosted by google](https://cloud.google.com/container-registry/docs/pushing-and-pulling#pushing_an_image_to_a_registry) 
on `gcr.io`. Login to google cloud shell and 

> gcloud config set project openimagechallenge 

where `openimagechallenge` is the `Project ID` shown on the console. 
Then login to get authorization and follow the steps to authenticate.

> gcloud auth login

Then obtain the access-token for docker

> gcloud auth print-access-token

Copy the access-token and go to the COS VM instance

> docker login -u oauth2accesstoken -p "<paste-the-access-token-here\>" https://gcr.io

Finally `push` the image into the google cloud Container Registry (just like Docker Hub)

> docker push HOSTNAME/PROJECT_ID/IMAGENAME:TAG

In my case

> docker push gcr.io/openimageschallenge/openimage:python3.5.6-stretch


