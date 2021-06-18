FROM sleak75/conda-mpi4py-haswell:latest
SHELL ["/bin/bash", "-c"]
WORKDIR /app

RUN conda install scipy h5py matplotlib
RUN conda install -c conda-forge phonopy
RUN conda install -c conda-forge phono3py
RUN conda clean -a

RUN /sbin/ldconfig

COPY multiphonon_dsf.py ./
COPY src ./src
RUN mkdir file_io
COPY file_io/__init__.py file_io/
RUN chmod -R a+rX /app
