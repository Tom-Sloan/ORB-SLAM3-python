# Use Ubuntu 22.04 as base
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y gnupg2 curl lsb-core vim wget python3-pip libpng16-16 libjpeg-turbo8 libtiff5

RUN apt-get install -y \
    # Base tools
    cmake \
    build-essential \
    git \
    unzip \
    pkg-config \
    python3-dev \
    # OpenCV dependencies
    python3-numpy \
    # Pangolin dependencies
    libgl1-mesa-dev \
    libglew-dev \
    libpython3-dev \
    libeigen3-dev \
    apt-transport-https \
    ca-certificates\
    software-properties-common \
    '^libxcb.*-dev' \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    x11-apps \ 
    git \
    ffmpeg \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN apt update

# Install OpenCV dependencies
RUN apt-get install -y python3-dev python3-numpy python2-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt-get install -y libgtk-3-dev
RUN apt-get install -y libssl-dev


RUN apt-get update && apt-get install -y \
    libboost-serialization-dev \
    && rm -rf /var/lib/apt/lists/*

# Build OpenCV 4.4.0
RUN cd /tmp && git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib &&\
    cd opencv_contrib && git checkout 4.6.0 && cd .. &&\
    cd opencv && git checkout 4.6.0 && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_EXAMPLES=OFF -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules /tmp/opencv_contrib -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j8 && make install && \
    cd / && rm -rf /tmp/opencv && rm -rf /tmp/opencv_contrib

# Build Pangolin
RUN cd /tmp && git clone https://github.com/stevenlovegrove/Pangolin && \
    cd Pangolin && git checkout v0.9.1 && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++14 -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j8 && make install && \
    cd / && rm -rf /tmp/Pangolin

# Set working directory
WORKDIR /copy
COPY . .

# # Build ORB_SLAM3 dependencies
RUN cd /copy/third_party/ORB_SLAM3/Thirdparty/DBoW2 && \
rm -rf build && mkdir build && cd build && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-maybe-uninitialized" .. && \
make -j8

RUN cd /copy/third_party/ORB_SLAM3/Thirdparty/g2o && \
rm -rf build && mkdir build && cd build && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-maybe-uninitialized" .. && \
make -j8

RUN cd /copy/third_party/ORB_SLAM3/Thirdparty/Sophus && \
rm -rf build && mkdir build && cd build && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-maybe-uninitialized" .. && \
make -j8


# # Install Python bindings
RUN pip3 install .

# Install Python dependencies
RUN pip3 install pillow pyparsing pytz six watchdog aio-pika pika opencv-python-headless
WORKDIR /app

ARG USERNAME
ARG UID
ARG GID
ARG HOME

# Set up .Xdefaults for X11 color customization
# Create a group and user with the specified GID and UID
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

# Grant the user sudo privileges (optional)
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the default user to the new user
USER $USERNAME
RUN echo "*customization: -color" > $HOME/.Xdefaults

CMD ["python3", "./demo/run_rgbd.py"] 