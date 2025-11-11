FROM cupy/cupy:v13.3.0

RUN apt-get update && apt-get install -y \
    git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN pip3 install --no-cache-dir \
    scikit-build \
    swig \ 
    pywavelets \
    numexpr \
    opencv-python \
    tifffile \
    h5py \ 
    dxchange \
    cmake \
    pyzmq \
    line_profiler \
    zarr \
    dask \
    matplotlib \ 
    notebook


WORKDIR /streamtomocupy
COPY . ./

RUN pip3 install .