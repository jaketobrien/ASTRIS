# Nvidia Jetson Orin NX Setup Instructions

```
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libopenblas-dev libatlas-base-dev liblapack-dev gfortran
sudo apt-get install -y libhdf5-serial-dev graphviz
sudo apt-get install -y python3-dev python3-tk python-imaging-tk
sudo apt-get install -y linux-image-generic linux-image-extra-virtual
sudo apt-get install -y linux-source linux-headers-generic
```

```
sudo apt-get install nvidia-jetpack
```

```
nano ~/.bashrc
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
nvcc --version
```

Create symbolic link to cuDNN libraries and header files

```
sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudnn*.so* /usr/local/cuda/lib64/
sudo ln -sf /usr/include/cudnn*.h /usr/local/cuda/include/
sudo ln -sf /usr/include/aarch64-linux-gnu/cudnn*.h /usr/local/cuda/include/
```

```
nano ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

## Install OpenCV with suitable add-ons

```
sudo apt update 
sudo apt upgrade 
sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall 
sudo apt install libjpeg-dev libpng-dev libtiff-dev 
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev 
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev 
sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev 
sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev 
sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev
```

```
sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd ~
```

```
sudo apt-get install libgtk-3-dev
sudo apt-get install python3-dev python3-pip
sudo -H pip3 install -U pip numpy
sudo apt install python3-testresources
sudo apt-get install libtbb-dev
sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
```

```
pip install ultralytics
pip uninstall -y opencv-python opencv-contrib-python
```

```
cd ~/Downloads
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.2.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.2.zip
unzip opencv.zip
unzip opencv_contrib.zip
```

```
pip install numpy
```

```
cd opencv-4.5.2
mkdir build
cd build
```

```
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_BIN=8.7 \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_GSTREAMER_1_0=ON \
      -D WITH_GTK=ON \
      -D BUILD_EXAMPLES=ON \
      -D BUILD_opencv_python3=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib-4.5.2/modules \
      -D PKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_PC_FILE_NAME=opencv.pc \
      -D WITH_TBB=ON \
      -D WITH_EIGEN=ON \
      -D WITH_JPEG=ON \
      -D WITH_PNG=ON \
      -D WITH_TIFF=ON \
      -D WITH_WEBP=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_HPX=OFF \
      -D WITH_IPP=OFF \
      -D GST_LIB=/usr/lib/aarch64-linux-gnu \
      -D GST_INCLUDE=/usr/include/gstreamer-1.0 \
      -D WITH_GTKGLEW=ON \
      ..
```

```
make -j8
sudo make install
```

## Install PyTorch for GPU

```
cd Downloads
```

Download PyTorch 2.1.0 (or relevant version for your system) https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

```
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev

pip3 install 'Cython<3'

pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
```

Install torchvision

```
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
```

```
cd torchvision

export BUILD_VERSION=0.16.1

python3 setup.py install --user

cd ../
```

## Install TensroFlow for GPU

```
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

sudo apt-get install python3-pip
sudo python3 -m pip install --upgrade pip
sudo pip3 install -U testresources setuptools

sudo pip3 install -U numpy future mock keras_preprocessing keras_applications gast protobuf pybind11 cython pkgconfig packaging h5py
```

Install the TensorFlow version compatable with your system. This can be found on the Nvidia website.

```
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 tensorflow==2.12.0+nv23.06
```
