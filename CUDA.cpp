#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

__global__ void count_kernel(double* res, const double* matrix1, const double* matrix2, size_t n, size_t vector_size, size_t local_n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    double sum_scalar = 0.0, sum1 = 0.0, sum2 = 0.0;
    for (int i=id*local_n; i<(id+1)*local_n; i++)
    {
        for (int j=0; j<vector_size; j++)
        {
                double val1 = matrix1[i * vector_size + j];
                double val2 = matrix2[i * vector_size + j];
                sum_scalar += val1 * val2;
                sum1 += val1 * val1;
                sum2 += val2 * val2;
        }
        res[i]=sum_scalar / (sqrt(sum1) * sqrt(sum2));

        sum_scalar = 0;
        sum1 = 0;
        sum2 = 0;
    }
}


int main() {
    size_t size_in_mb = 20;
    size_t vector_size = 100;
    size_t num_vectors = (size_in_mb * 1024 * 1024) / (2 * vector_size * sizeof(double));
    size_t num_threads = 16;
    if (num_vectors%num_threads!=0) {num_vectors=((num_vectors/num_threads)*num_threads)+num_threads;}
    size_t local_n=num_vectors/num_threads;

    vector<double> matrix1(num_vectors * vector_size);
    vector<double> matrix2(num_vectors * vector_size);
    vector<double> res(num_vectors);

    fstream input_file("in.txt", ios::out);
    srand(static_cast<unsigned>(time(NULL)));

    for (size_t i = 0; i < 2 * num_vectors; i++) {
        for (size_t j = 0; j < vector_size; j++) {
            input_file << static_cast<double>(rand() % 1000 - 500) << " ";
        }
        input_file << endl;
    }
    input_file.close();
    input_file.open("in.txt", ios::in);
    for (size_t i = 0; i < num_vectors; i++) {
        for (size_t j = 0; j < vector_size; j++) {
            input_file >> matrix1[i * vector_size + j];
        }
    }
    for (size_t i = 0; i < num_vectors; i++) {
        for (size_t j = 0; j < vector_size; j++) {
            input_file >> matrix2[i * vector_size + j];
        }
    }
    input_file.close();
    
    double *d_matrix1, *d_matrix2, *d_res;

    cudaMalloc((void **)&d_matrix1, num_vectors * vector_size * sizeof(double));
    cudaMalloc((void **)&d_matrix2, num_vectors * vector_size * sizeof(double));
    cudaMalloc((void **)&d_res, num_vectors * sizeof(double));

    cudaMemcpy(d_matrix1, matrix1.data(), num_vectors * vector_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2.data(), num_vectors * vector_size * sizeof(double), cudaMemcpyHostToDevice);

    int threads_per_block = 1;
    int blocks_per_grid = num_threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    count_kernel<<<blocks_per_grid, threads_per_block>>>(d_res, d_matrix1, d_matrix2, num_vectors, vector_size, local_n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0;

    cudaMemcpy(res.data(), d_res, num_vectors * sizeof(double), cudaMemcpyDeviceToHost);

    fstream output_file("f.txt", ios::out);
    if (!output_file.is_open()) {
        cerr << "Ошибка открытия файла f.txt для записи" << endl;
        return 1;
    }
    for (size_t i = 0; i < num_vectors; i++) {
        output_file << "cos(phi)=" << res[i] << endl;
    }
    output_file << "Время выполнения в секундах: " << seconds << endl;
    output_file.close();

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_res);

    return 0;
}
