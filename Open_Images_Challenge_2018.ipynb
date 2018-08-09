{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Setting up Google Cloud Platform (GCP)\n",
    "This section is about how I set up the platform to enable the use of docker container for this project. \n",
    "* Create VM instances (with preemptibilty to save cost) with Container-Optimized OS (COS) 66-10452.109.0 stable. \n",
    "* Add an additional Standard Persistent disk (100GB) for multiple VM to share by [mounting](https://cloud.google.com/compute/docs/disks/add-persistent-disk#create_disk), I named the disk `openimage` under the path `/mnt/disks/`.\n",
    "\n",
    "# Downloading data \n",
    "* Download the [Open Image dataset](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/) to the openimage disk by simply using `wget`.\n",
    "\n",
    "* To `unzip` in COS, I run the docker image from GCP [cos-toolbox](gcr.io/google-containers/toolbox) with `-v` to mount local directory in the container.\n",
    "\n",
    "> docker run --name toolbox -v /mnt/disks/openimage:/mnt -d gcr.io/google-containers/toolbox tail -f /dev/null\n",
    "\n",
    "* Run an interactive bash session in the `toolbox` container\n",
    "\n",
    "> docker exec -it toolbox bash\n",
    "\n",
    "* `unzip` the data in the mounted directory `/mnt` in the container, pointing to the 100GB standard persistent disk /mnt/disks/openimage (mounted on my local VM directory).\n",
    "\n",
    "> unzip train00.zip\n",
    "\n",
    "# Setting up python packages using Docker\n",
    "*  Create `Dockerfile` to manage the python environment for this project. The `Dockerfile`\n",
    "\n",
    "```\n",
    "# Use the official Python runtime as a parent image\n",
    "FROM python:2.7-stretch\n",
    "# Install the project related packages\n",
    "RUN pip2 install -r requirements.txt\n",
    "```\n",
    "\n",
    "* For this project, `requirements.txt`\n",
    "\n",
    "```\n",
    "pandas \\ tensorflow \\ numpy \\ matplotlib \\ h5py \\ keras \\ scipy \\ pydot \\ argparse \\ yolo_utils \\ yad2k \\ cv2 \\ fr_utils \\ inception_blocks_v2\n",
    "```\n",
    "\n",
    "* Build the `Dockerfile` to produce an image\n",
    "\n",
    "> docker build .\n",
    "\n",
    "* `push` the image to my registry for future use\n",
    "\n",
    "> docker  login --username=gchen0119 \\-\\-email=gchen0119@gmail.com\n",
    "\n",
    "> docker tag <IMAGE_ID> gchen0119/<REPOSITORY\\>:TAG\n",
    "\n",
    "> docker push gchen0119/<REPOSITORY\\>\n",
    "\n",
    "# Start Python in the container\n",
    "\n",
    "> docker \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Random Notes\n",
    "* Transferring data between my laptop and the VM instance using the save ssh private key\n",
    "\n",
    "> scp -i ~/.ssh/google_compute_engine /path/to/file gino@<REMOTE_IP>\n",
    "\n",
    "where `<REMOTE_IP>` is assigned at the start of a VM instance.\n",
    "\n",
    "* Transferring data between my laptop and google bucket \n",
    "\n",
    "> gcloud compute ssh gchen@<LOCAL_IP> --command='gsutil cp /path/to/file gs://mybucket' \n",
    "\n",
    "where `<LOCAL_IP>` can be located by `ifconfig` on macbook or `ip addr show` on linux."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
