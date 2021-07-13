FROM sleak75/conda-mpi4py-haswell:latest
SHELL ["/bin/bash", "-c"]
WORKDIR /app

RUN conda install scipy h5py matplotlib
RUN conda install -c conda-forge phonopy
RUN conda install -c conda-forge phono3py
RUN conda install -c conda-forge numba
RUN conda install sympy
RUN conda clean -a

RUN /sbin/ldconfig

COPY multiphonon_dsf.py ./
COPY src ./src
RUN chmod -R a+rX /app
