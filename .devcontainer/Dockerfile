# Base image for C++ development on Ubuntu 22.04
FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-22.04

# Argument to control CMake reinstallation
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.31.5"

# Optionally install a specific CMake version for vcpkg compatibility
# Copy the script to reinstall CMake
COPY ./reinstall-cmake.sh /tmp/
# Execute the script if a version is specified, then clean up
RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# [Optional] Placeholder for installing additional vcpkg ports
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Placeholder for installing additional apt packages
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Install essential build dependencies and other required packages
RUN apt-get clean && \
    apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends --fix-missing \
        # Core build tools
        build-essential binutils \
        ca-certificates cmake cmake-qt-gui curl \
        # GUI/Display related
        dbus-x11 \
        # Multimedia
        ffmpeg \
        # Development tools
        gdb gcc g++ gfortran git \
        tar \
        # System utilities
        lsb-release \
        procps \
        manpages-dev \
        unzip \
        zip \
        wget \
        xauth \
        # Python related
        swig \
        python3-pip python3-dev python3-numpy python3-distutils \
        python3-setuptools python3-pyqt5 python3-opencv \
        # Boost libraries
        libboost-python-dev libboost-thread-dev \
        # Math/Linear Algebra libraries
        libatlas-base-dev libeigen3-dev liblapack-dev \
        liblapacke-dev libopenblas-dev \
        # Multimedia/Image/Video libraries
        libavcodec-dev \
        libavformat-dev libavutil-dev libcanberra-gtk3-module \
        libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev libpng-dev libpostproc-dev \
        libswscale-dev libtiff-dev libtiff5-dev libv4l-dev libxvidcore-dev libx264-dev \
        libopencv-dev \
        # Graphics/GUI libraries
        libglew-dev libgl1-mesa-dev libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
        libgtk-3-dev libsm6 libx11-dev libxext6 libxine2-dev \
        libxrender-dev libgtkglext1 libgtkglext1-dev \
        # VTK
        libvtk9-dev \
        # Camera/GStreamer
        libdc1394-dev libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        # OpenEXR
        libopenexr-dev \
        openexr \
        # Other development libraries
        libpq-dev libtbb-dev libtbb2 libtesseract-dev \
        pkg-config \
        # Video4Linux utilities
        qv4l2 \
        v4l-utils \
        # Compression
        zlib1g-dev \
        # *** Add Open3D development library ***
        libopen3d-dev \
        # Locale settings
        locales \
    # Generate locale and set environment variable
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables for locale
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install OpenCV from source
ARG OPENCV_VERSION="4.8.0"

# Create OpenCV directory and download source code
WORKDIR /opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mv opencv-${OPENCV_VERSION} opencv \
    && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

# Create build directory for OpenCV
RUN mkdir /opencv/opencv/build
WORKDIR /opencv/opencv/build

# Configure OpenCV build with CMake
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib/modules \
    # Note: Ensure python3 executable path is correct if changed from default
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D BUILD_EXAMPLES=ON .. \
    # Build and install OpenCV
    && make -j$(nproc) && make install && ldconfig \
    # Clean up OpenCV source directories after install
    && rm -rf /opencv

# Install EVO trajectory evaluation library using pip
RUN pip3 install --no-cache-dir evo --upgrade

# Set display environment variable for GUI forwarding (adjust if needed)
# ENV DISPLAY=host.docker.internal:0
