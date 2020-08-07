# Use tensorflow/tensorflow as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG TF_VERSION=1.15.0-gpu

FROM tensorflow/tensorflow:${TF_VERSION}-py3

# System maintenance
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-tk \
    git \
    libsm6 \
    autoconf \
    automake  \
    libtool  \
    libtiff-dev \
    libcairo2-dev \
    cmake \
    glib2.0 \
    libgdk-pixbuf2.0-dev \
    libxml2-dev \
    sqlite3 \
    libsqlite3-dev \
    pkg-config \
    python3-pip &&\
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/pip install --upgrade pip

# Install openJPEG
WORKDIR /opt
RUN git clone https://github.com/uclouvain/openjpeg.git
WORKDIR /opt/openjpeg
RUN mkdir build
WORKDIR /opt/openjpeg/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
	make && \
	make install && \
	make clean

# Install openslide
WORKDIR /opt
RUN git clone https://github.com/openslide/openslide.git
WORKDIR /opt/openslide
RUN autoreconf -i && \
	 ./configure && \
	 make && \
	 make install

# Copy the setup.py and requirements.txt and install the deepcell-tf dependencies
COPY requirements.txt /opt/TCGA/

# Prevent reinstallation of tensorflow and install all other requirements.
RUN sed -i "/tensorflow/d" /opt/TCGA/requirements.txt && \
    pip3 install -r /opt/TCGA/requirements.txt

# Install deepcell via setup.py
WORKDIR /opt
RUN git clone -b tf2_migration https://github.com/vanvalenlab/deepcell-tf.git
RUN python3 -m pip install /opt/deepcell-tf && \
    cd /opt/deepcell-tf && \
    python3 setup.py build_ext --inplace
WORKDIR /notebooks

# Install the latest version of keras-applications
RUN python3 -m pip install --upgrade git+https://github.com/keras-team/keras-applications.git

# Install tcga-utils
WORKDIR /opt
COPY setup.py requirements.txt /opt/tcga-utils/
COPY tcga_utils /opt/tcga-utils/tcga_utils
COPY manifest /data/manifest
RUN python3 -m pip install /opt/tcga-utils && \
    cd /opt/tcga-utils && \
    python3 setup.py build_ext --inplace
WORKDIR /notebooks

# Copy over tcga notebooks
COPY scripts/ /notebooks/

# Start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
