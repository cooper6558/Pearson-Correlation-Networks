#define BLOCK_SIZE 1024
#define BLOCK_WIDTH	32
#include <stdio.h>

__global__ void mean(float *, float *);
__global__ void mult(float *, float *, int, int, int);
__global__ void square(float *);
__global__ void sum(float *, float *, int);
__global__ void sqrt(float *);
__global__ void divide(float *, float *, int);


float *gpu_correlation(float *Data, int variables, int samples) {
	puts("Copying Data to GPU...");
	float *d_Data, *Network, *d_Network, *d_Sums;
	cudaMalloc((void **) &d_Data, sizeof(float) * samples * variables);
	cudaMalloc((void **) &d_Network, sizeof(float) * variables * variables);
	cudaMalloc((void **) &d_Sums, sizeof(float) * variables);
	Network = (float *) malloc(sizeof(float) * variables * variables);

	cudaMemcpy(d_Data, Data, sizeof(float) * samples * variables, cudaMemcpyHostToDevice);

	puts("Computing Correlation Network...");

	int sum_threads = 1;
	while (sum_threads < samples)
		sum_threads *= 2;

	int grid_width = variables / BLOCK_WIDTH;
	if (variables % BLOCK_WIDTH)
		grid_width++;
	dim3 threads(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 blocks(grid_width, grid_width);
	int phases = samples / BLOCK_WIDTH;
	if (samples % BLOCK_WIDTH)
		phases++;


	sum<<<variables, sum_threads>>>(d_Data, d_Sums, samples);
	mean<<<variables, samples>>>(d_Data, d_Sums);
	mult<<<blocks, threads>>>(d_Data, d_Network, variables, samples, phases);
	square<<<variables, samples>>>(d_Data);
	sum<<<variables, sum_threads>>>(d_Data, d_Sums, samples);
	sqrt<<<1, variables>>>(d_Sums);
	divide<<<blocks, threads>>>(d_Network, d_Sums, variables);

	// note that d_Sums is a vector of length variables,
	// and d_Network is a square matrix of dimensions
	// variables X variables.


	puts("Copying Correlation Network to CPU...");
	cudaMemcpy(Network, d_Network, sizeof(float) * variables * variables, cudaMemcpyDeviceToHost);
	cudaFree(d_Data);
	cudaFree(d_Network);
	cudaFree(d_Sums);
	return Network;
}

__global__ void divide(float *Data, float *sums, int n) {
	int row = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
	int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

	__shared__ float row_sums[BLOCK_WIDTH];
	__shared__ float col_sums[BLOCK_WIDTH];

	if (threadIdx.x == 0 && row < n)
		col_sums[threadIdx.y] = sums[row];
	if (threadIdx.y == 0 && col < n)
		row_sums[threadIdx.x] = sums[col];
	__syncthreads();
	
	if (row < n && col < n)
		Data[row * n + col] /= row_sums[threadIdx.x] * col_sums[threadIdx.y];
}

// blocks: rows
// threads: smallest power of 2 >= n
// n: elements per row
__global__ void sum(float *Data, float *Output, int n) {
	__shared__ float row[BLOCK_SIZE];

	int global_address = blockIdx.x*n + threadIdx.x;

	if (threadIdx.x < n)
		row[threadIdx.x] = Data[global_address];
	else
		row[threadIdx.x] = 0;
	__syncthreads();

	// parallel reduction
	for (int len=blockDim.x/2; len>0; len/=2) {
		if (threadIdx.x < len)
			row[threadIdx.x] += row[threadIdx.x + len];
		__syncthreads();
	}

	// write sum
	if (threadIdx.x == 0)
		Output[blockIdx.x] = row[0];
}

// element wise square of a matrix
__global__ void square(float *Data) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Data[index] *= Data[index];
}

// element wise square root of a vector
__global__ void sqrt(float *Data) {
	Data[threadIdx.x] = sqrtf(Data[threadIdx.x]);
}

// assume no more than 1024 per row
__global__ void mean(float *A, float *Sums) {
	__shared__ float m;
	if (threadIdx.x == 0)
		m = Sums[blockIdx.x] / blockDim.x;
	__syncthreads();
	A[blockIdx.x*blockDim.x + threadIdx.x] -= m;
}

// multiplies A by its own transpose and stores to B
// the key is to start with matmul and invert coordinates for B.
// this function is the core of the library and is the main contribution.
__global__ void mult(float *A, float *B, int variables, int samples, int phases) {
    __shared__ float m_loc[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float n_loc[BLOCK_WIDTH][BLOCK_WIDTH];
	float p = 0;
	int row = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
	int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

	// iterate on phases
	for (int phase=0; phase<phases; phase++) {
		// step 1: load phase into local memory
		// load phase from A into local memory if phase in bounds
		if (row < variables && phase*BLOCK_WIDTH + threadIdx.x < samples)
			m_loc[threadIdx.y][threadIdx.x] = A[row*samples + phase*BLOCK_WIDTH + threadIdx.x];
			// A[row][phase*BLOCK_WIDTH+threadIdx.x]
		else
			m_loc[threadIdx.y][threadIdx.x] = 0;
		// sync since the diagonal blocks depend on m being loaded
		__syncthreads();
		// load phase from A.T into local memory. This is where coordinates need to be inverted
		// this part is tricky
		// take advantage of m_loc if we're on the diagonal; n_loc=m_loc.T
		if (blockIdx.y == blockIdx.x)
			n_loc[threadIdx.y][threadIdx.x] = m_loc[threadIdx.x][threadIdx.y];
		else if (col < variables && phase*BLOCK_WIDTH + threadIdx.y < samples)
			n_loc[threadIdx.y][threadIdx.x] = A[col*variables + phase*BLOCK_WIDTH + threadIdx.y];
			// B[phase*BLOCK_WIDTH + threadIdx.y][col]
			// A[col][phase*BLOCK_WIDTH + threadIdx.y]
		else 
			n_loc[threadIdx.y][threadIdx.x] = 0;
		__syncthreads();

		// step 2: dot product
		for (int i=0; i<BLOCK_WIDTH; i++)
			p += m_loc[threadIdx.y][i] * n_loc[i][threadIdx.x];
	}

	// if valid thread, write to output matrix
	if (row < variables && col < variables)
		B[row * variables + col] = p;
}
