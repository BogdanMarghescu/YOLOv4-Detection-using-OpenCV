# YOLOv4-Detection-using-OpenCV
YOLOv4 detection on COCO dataset using OpenCV DNN module, compiled with CUDA.

### Installing dependencies:
#### For Windows:
1. Check if the GPU of the system supports CUDA by checking if it is in this list: https://developer.nvidia.com/cuda-gpus
2. If the GPU supports CUDA, install it using this guide: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
3. Install the latest Python 3 version from https://www.python.org/downloads/
4. Run this command in command line:
    ```powershell
    PS C:> pip install numpy
    ```
5. Install and compile OpenCV with CUDA support using this tutorial: https://medium.com/analytics-vidhya/build-opencv-from-source-with-cuda-for-gpu-access-on-windows-5cd0ce2b9b37

#### For Ubuntu/Debian:
1. Check if the GPU of the system supports CUDA by checking if it is in this list: https://developer.nvidia.com/cuda-gpus
2. If the GPU supports CUDA, install it using this guide: https://medium.com/@pydoni/how-to-install-cuda-11-4-cudnn-8-2-opencv-4-5-on-ubuntu-20-04-65c4aa415a7b
3. Run these commands:
```console
foo@bar:~$ sudo chmod +x install_dependencies_ubuntu.sh    # make the script executable
foo@bar:~$ sudo ./install_dependencies_ubuntu.sh           # run the script to install the dependencies of the application
```

### Guide for using the Face Mask Detector application:

1. Execute the Python Script from command line like this:
```console
foo@bar:~$ python .\yolov4_detection.py
```
or
```console
foo@bar:~$ python3 .\yolov4_detection.py
```


### Datasets and weights used by the Face Mask Detector application:
- The dataset used for training this model COCO Dataset.
- The trained YOLOv4 weights, together with the configuration file can be found at this link: https://mega.nz/folder/3hgEwR4I#tftUqrtWuIhzOOJk-3gGFA
