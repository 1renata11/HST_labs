#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include </usr/include/mpich-x86_64/mpi.h>

using namespace std;

__global__ void count_kernel(double* res, const double* matrix1, const double* matrix2, size_t n, size_t vector_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Hello from GPU\n");
    double sum_scalar = 0.0, sum1 = 0.0, sum2 = 0.0;
    if (id<n) {
        for (int j = 0; j < vector_size; j++) {
            double val1 = matrix1[id * vector_size + j];
            double val2 = matrix2[id * vector_size + j];
            sum_scalar += val1 * val2;
            sum1 += val1 * val1;
            sum2 += val2 * val2;
        }
        res[id] = sum_scalar / (sqrt(sum1) * sqrt(sum2));

        sum_scalar = 0;
        sum1 = 0;
        sum2 = 0;
    }
    
}

void process_on_cpu(vector<double>& res, const vector<double>& matrix1, const vector<double>& matrix2, size_t start, size_t end, size_t vector_size) {
    //printf("Hello from CPU\n");
    for (size_t i = start; i < end; i++) {
        double sum_scalar = 0.0, sum1 = 0.0, sum2 = 0.0;
        for (size_t j = 0; j < vector_size; j++) {
            double val1 = matrix1[i * vector_size + j];
            double val2 = matrix2[i * vector_size + j];
            sum_scalar += val1 * val2;
            sum1 += val1 * val1;
            sum2 += val2 * val2;
        }
        res[i - start] = sum_scalar / (sqrt(sum1) * sqrt(sum2));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t size_in_mb = 25;
    size_t vector_size = 100;
    size_t num_vectors = (size_in_mb * 1024 * 1024) / (2 * vector_size * sizeof(double));
    if (num_vectors%size!=0) {num_vectors=((num_vectors/size)*size)+size;}

    if (rank == 0) {
        cout << "Running with " << size << " processes (1 GPU, " << size - 1 << " CPUs)." << endl;
    }

    size_t local_n = num_vectors / size;
    vector<double> matrix1(num_vectors * vector_size);
    vector<double> matrix2(num_vectors * vector_size);
    vector<double> local_res(local_n);
    vector<double> all_times(size);
    double local_elapsed_time=0.0;
    if (rank == 0) {
        fstream input_file("in.txt", ios::out);
        if (!input_file.is_open()) {
            cerr << "Ошибка открытия файла in.txt для записи" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        srand(time(NULL));
        for (size_t i = 0; i < 2 * num_vectors; i++) {
            for (size_t j = 0; j < vector_size; j++) {
                input_file << (double)(rand() % 1000 - 500) << " ";
            }
            input_file << endl;
        }
        input_file.close();
        input_file.open("in.txt", ios::in);
        if (!input_file.is_open()) {
            cerr << "Ошибка открытия файла in.txt для чтения" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
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
    }

    MPI_Bcast(matrix1.data(), num_vectors * vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix2.data(), num_vectors * vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double *d_matrix1, *d_matrix2, *d_res;

        cudaMalloc((void **)&d_matrix1, num_vectors * vector_size * sizeof(double));
        cudaMalloc((void **)&d_matrix2, num_vectors * vector_size * sizeof(double));
        cudaMalloc((void **)&d_res, local_n * sizeof(double));

        cudaMemcpy(d_matrix1, matrix1.data(), num_vectors * vector_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix2, matrix2.data(), num_vectors * vector_size * sizeof(double), cudaMemcpyHostToDevice);

        int threads_per_block = 256;
        int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        count_kernel<<<blocks_per_grid, threads_per_block>>>(d_res, d_matrix1, d_matrix2, local_n, vector_size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        float gpu_time=0.0f;
        cudaEventElapsedTime(&gpu_time, start, stop);
        local_elapsed_time = static_cast<double>(gpu_time) / 1000.0;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(local_res.data(), d_res, local_n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_matrix1);
        cudaFree(d_matrix2);
        cudaFree(d_res);
    } else {
        size_t start = rank * local_n;
        size_t end = start + local_n;
        double start_time = MPI_Wtime();
        process_on_cpu(local_res, matrix1, matrix2, start, end, vector_size);
        double end_time = MPI_Wtime();
        local_elapsed_time = end_time - start_time;
    }

    vector<double> final_res;
    if (rank == 0) {
        final_res.resize(num_vectors);
    }
    MPI_Gather(local_res.data(), local_n, MPI_DOUBLE, final_res.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_elapsed_time, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        fstream output_file("f.txt", ios::out);
        if (!output_file.is_open()) {
            cerr << "Ошибка открытия файла" << endl;
            MPI_Finalize();
            return 1;
        }
        for (size_t i = 0; i < num_vectors; i++) {
            output_file << "cos(phi)=" << final_res[i] << endl;
        }
        for (size_t i = 0; i < size; i++) {
            output_file << "Процесс " << i << " потратил " << all_times[i] << " секунд"<< endl;
        }
        output_file.close();
    }

    MPI_Finalize();
    return 0;
}
