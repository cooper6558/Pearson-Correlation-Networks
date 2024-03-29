# Pearson Correlation Networks
Compute the Pearson correlation coefficient between every pair of rows in a matrix.

This library provides `float *gpu_correlation(float *, int, int);` which
takes an input matrix and produces a correlation network using a GPU.
The library consists of `correlation.cu`, `correlation.h`, `makefile`,
and `README`, with an example `main.c` also provided.

With these files, run `make` to compile everything, then `./correlation` to
run the example. The current example is set to run correlation analysis on
7 different datasets with 9 samples each (input matrix 7x9) but can handle
up to 1024 by 1024. It will print the resulting correlation network. Tested
on the Clemson University SoC Titan machines, even at maximum matrix size,
this algorithm runs almost instantly due to its incredibly parallel nature.

I submitted this code as my final project for ECE 6780 at Clemson University.
I plan to keep working on it on my own time as there is much that can be
improved. Right now it is more of a demonstration of my kernels than a
library. I need to generalize some of the kernels to expand the input size,
I need to merge some of the kernels into one to reduce global memory access,
and I want to generalize to more hardware, such as older compute capabilities
and possibly use multiple GPUs. I also may be able to use tensor cores on
newer GPUs. Finally, it would be interesting to implement other correlation
metrics than just Pearson.

