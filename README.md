# bwtest

A simple bandwidth test for cpu and opencl in linux

# build
mkdir build && cd build && cmake .. && make -j8

# run
## run with cpu
./bin/bw cpu 

## run with gpu(opencl)
./bin/bw gpu
