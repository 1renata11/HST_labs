#include <iostream>
#include <cmath>
#include </usr/local/Cellar/mpich/4.2.3/include/mpi.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <cstdio>

using namespace std;

void count(vector<double>& res, vector<double> matrix1, vector<double> matrix2, size_t n, size_t vector_size) {
    double sum_scalar, sum1, sum2;
    for (size_t i = 0; i < n; i++) {
        sum_scalar = 0.0;
        sum1 = 0.0;
        sum2 = 0.0;
        for (size_t j = 0; j < vector_size; j++) {
            sum_scalar += matrix1[i * vector_size + j] * matrix2[i * vector_size + j];
            sum1 += matrix1[i * vector_size + j] * matrix1[i * vector_size + j];
            sum2 += matrix2[i * vector_size + j] * matrix2[i * vector_size + j];
        }
        res[i] = sum_scalar / (sqrt(sum1) * sqrt(sum2));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    size_t size_in_mb = 25;
    char output_name[1024] = "f.txt";
    size_t vector_size = 100;
    size_t num_vectors = (size_in_mb * 1024 * 1024) / (2 * vector_size * sizeof(double));
    size_t remain = num_vectors % world_size;
    size_t local_n = num_vectors / world_size + (world_rank < remain ? 1 : 0);
    size_t start_index = world_rank < remain ? world_rank * local_n * vector_size : (remain * (local_n + 1) + (world_rank - remain) * local_n) * vector_size;

    vector<double> data(vector_size * 2 * num_vectors);
    vector<double> matrix1(local_n * vector_size);
    vector<double> matrix2(local_n * vector_size);
    vector<double> res(local_n);
    vector<double> all_times(world_size);
    vector<double> receive(num_vectors);
    vector<int> sendcounts(world_size);
    vector<int> displs(world_size);
    for (int i = 0; i < world_size; i++) {
        sendcounts[i] = (num_vectors / world_size + (i < remain ? 1 : 0)) * vector_size;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    if (world_rank == 0) {
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
        for (size_t i = 0; i < 2 * num_vectors; i++) {
            for (size_t j = 0; j < vector_size; j++) {
                input_file >> data[i * vector_size + j];
            }
        }
        input_file.close();
        int f = remove("in.txt");
    }

    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, matrix1.data(), local_n * vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(data.data() + num_vectors * vector_size, sendcounts.data(), displs.data(), MPI_DOUBLE, matrix2.data(), local_n * vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    count(res, matrix1, matrix2, local_n, vector_size);
    double end_time = MPI_Wtime();
    double local_elapsed_time = end_time - start_time;

    MPI_Gather(res.data(), local_n, MPI_DOUBLE, receive.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_elapsed_time, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        fstream output_file(output_name, ios::out);
        if (!output_file.is_open()) {
            cerr << "Ошибка открытия файла f.txt для записи" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < num_vectors; i++) {
            output_file << "cos(phi)=" << receive[i] << endl;
        }
        for (int i = 0; i < world_size; i++) {
            output_file << "Процесс " << i << " потратил " << all_times[i] << " секунд" << endl;
        }
        output_file.close();
    }

    MPI_Finalize();
    return 0;
}
