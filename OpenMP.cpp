#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <ctype.h>
#include </usr/local/opt/libomp/include/omp.h>

void fill_file(FILE *file, size_t num_vectors, size_t vector_size) {
    srand(time(NULL));
    for (size_t i = 0; i < 2*num_vectors; i++) {
        for (size_t j = 0; j < vector_size; j++) {
            fprintf(file, "%lf ", (double)(rand()%1000-500));
        }
        fprintf(file, "\n");
    }
}

void count(double* res, double* matrix1, double* matrix2, size_t n, size_t vector_size)
{
    #pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i<n; i++)
    {
        double sum_scalar=0.0, sum1=0.0, sum2=0.0;
        for (size_t j=0; j<vector_size; j++)
        {
            sum_scalar += matrix1[i * vector_size + j] * matrix2[i * vector_size + j];
            sum1 += matrix1[i * vector_size + j] * matrix1[i * vector_size + j];
            sum2 += matrix2[i * vector_size + j] * matrix2[i * vector_size + j];
        }
        res[i]=sum_scalar/(sqrt(sum1)*sqrt(sum2));
    }
}

void read_and_write(FILE *input_file, FILE *output_file, size_t n, size_t vector_size) {
    omp_set_num_threads(4);
    char buffer[1024];
    double *res=(double*)malloc(n * sizeof(double));
    double *matrix1=(double*)malloc(n * vector_size * sizeof(double));
    double *matrix2=(double*)malloc(n * vector_size * sizeof(double));
    if (!matrix1 || !matrix2) {
        perror("Failed to allocate memory");
        exit(1);
    }
    size_t j=0;
    while (fgets(buffer, 1024, input_file)) {
        char *ptr = buffer;
        size_t i = 0;
        while (i < vector_size && sscanf(ptr, "%lf", &matrix1[vector_size*j+i]) == 1) {
            i++;
            while (*ptr && *ptr != ' ') ptr++;
            if (*ptr) ptr++;
        }
        j++;
        if (j==n) break;
    }
    j=0;
    while (fgets(buffer, 1024, input_file)) {
        char *ptr = buffer;
        size_t i = 0;
        while (i < vector_size && sscanf(ptr, "%lf", &matrix2[vector_size*j+i]) == 1) {
            i++;
            while (*ptr && *ptr != ' ') ptr++;
            if (*ptr) ptr++;
        }
        j++;
        if (j==n) break;
    }
    struct timeval start, end;
    gettimeofday(&start, NULL);
    count(res, matrix1, matrix2, n, vector_size);
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    for (size_t i=0; i<n; i++)
    {
        fprintf(output_file, "cos(phi) = %lf\n", res[i]);
    }
    fprintf(output_file, "Processing time: %lf", elapsed_time);
    free(matrix1);
    free(matrix2);
    free(res);
}


int main() {
    size_t size_in_mb;
    char output_name[1024];
    scanf("%zu", &size_in_mb);
    scanf("%s", output_name);
    size_t vector_size = 100;
    FILE *input_file = fopen("in.txt", "w");
    if (!input_file) {
        perror("Failed to create input file");
        return 1;
    }
    size_t num_vectors = (size_in_mb * 1024 * 1024) / (2*vector_size * sizeof(double));
    fill_file(input_file, num_vectors, vector_size);
    fclose(input_file);
    input_file=fopen("in.txt", "r");
    if (!input_file) {
        perror("Failed to create input file");
        return 1;
    }
    FILE *output_file = fopen(output_name, "w");
    if (!output_file) {
        perror("Failed to open output file");
        fclose(input_file);
        return 1;
    }
    read_and_write(input_file, output_file, num_vectors, vector_size);
    fclose(input_file);
    fclose(output_file);
    if (remove("in.txt") != 0) {
        perror("Failed to delete input file");
        return 1;
    }
    return 0;
}
