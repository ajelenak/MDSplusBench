# MDSplusBench
A benchmark reading data from MDSplus using various protocols

This repo contains a python script `benchmark.py`

This contains benchmarks for:

distributed_client
thin_client
getMany over thin_client
hdf5 - 1 file per shot

It contains Multiprocessed benchmarks called
 BENCH_XXXX
which take an array of shots, an array of signal records, and a number of threads.
The these break the list of shots into threads chunks and distribute the work using python multiprocessing.

There is a python function `drop_caches.py` which can be placed in the MDS_PATH of the server being connected to.  With sufficent passwordless sudo priveledges it will cause the server to flush its file system caches.

There is a directory with the HDF files that the benchmark is looking for.

