#include <cmath>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <mpi.h>  // MPI Library for the MPI version
#include <omp.h>  // OpenMP library
#include <iomanip>  // For better output formatting>
#include <cstring>  // For memcpy
#include <sstream>
#include <vector>
#include <fstream>

using namespace std;

#define THREADS_PER_BLOCK 256
#define OPENMP_THREADS 8  // Default number of threads for OpenMP

/* ----------------------- CUDA Helper Functions ----------------------- */
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void norm_kernel(double* x, double* result, int length) {
    __shared__ double shared_sum[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    shared_sum[threadIdx.x] = 0.0;

    for (int i = tid; i < length; i += blockDim.x * gridDim.x) {
        shared_sum[threadIdx.x] += x[i] * x[i];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAddDouble(result, shared_sum[0]);
    }
}

__global__ void scalar_div_kernel(double* x, double r, int length, double* y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < length; i += blockDim.x * gridDim.x) {
        y[i] = x[i] / r;
    }
}

__global__ void scalar_sub_kernel(double* x, double r, int length, double* y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < length; i += blockDim.x * gridDim.x) {
        y[i] -= r * x[i];
    }
}

__global__ void dot_product_kernel(double* x, double* y, double* result, int length) {
    __shared__ double shared_sum[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    shared_sum[threadIdx.x] = 0.0;

    for (int i = tid; i < length; i += blockDim.x * gridDim.x) {
        shared_sum[threadIdx.x] += x[i] * y[i];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAddDouble(result, shared_sum[0]);
    }
}

/* --------------------- Progress Tracker Function --------------------- */
void track_progress(int current_step, int total_steps, const string& version) {
    int progress = (100 * current_step) / total_steps;
    static int last_progress = -1;  // Keep track of the last reported progress

    if (progress > last_progress) {  // Only report progress when it changes
        cout << "[" << version << "] Progress: " << progress << "%\r";
        cout.flush();  // Ensure the output is printed immediately
        last_progress = progress;
    }
}

/* ----------------------- OpenMP Functions ----------------------- */
// Set the number of threads based on the matrix size for OpenMP
void adjust_threads(int matrix_size) {
    if (matrix_size < 1000) {
        omp_set_num_threads(2);  // Use fewer threads for small matrices
    } else if (matrix_size < 5000) {
        omp_set_num_threads(4);  // Moderate number of threads for medium matrices
    } else {
        omp_set_num_threads(OPENMP_THREADS);  // Max threads for larger matrices
    }
}

// OpenMP Norm Calculation
double norm_openmp(double* x, int length) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < length; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

// OpenMP Scalar Division
void scalar_div_openmp(double* x, double r, int length, double* y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        y[i] = x[i] / r;
    }
}

// OpenMP Scalar Subtraction
void scalar_sub_openmp(double* x, double r, int length, double* y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        y[i] -= r * x[i];
    }
}

// OpenMP Dot Product
double dot_product_openmp(double* x, double* y, int length) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < length; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

/* --------------------- OpenMP Version of Householder QR --------------------- */
void householder_openmp(double** A, double** Q, double** R, int m, int n) {
    adjust_threads(m);  // Adjust thread usage based on matrix size

    // Initialize R to A
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            R[i][j] = A[i][j];

    // Initialize Q to identity
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            Q[i][j] = (i == j) ? 1.0 : 0.0;

    for (int k = 0; k < n; k++) {
        // Compute the norm of the k-th column below row k
        double norm_x = 0.0;
        #pragma omp parallel for reduction(+:norm_x)
        for (int i = k; i < m; i++) {
            norm_x += R[i][k] * R[i][k];
        }
        norm_x = sqrt(norm_x);

        if (norm_x == 0.0) continue;

        // Adjust sign to ensure positive diagonal in R
        double s = (R[k][k] >= 0) ? 1.0 : -1.0;
        double alpha = s * norm_x;

        // Compute the Householder vector v
        double* v = new double[m];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            if (i < k)
                v[i] = 0.0;
            else if (i == k)
                v[i] = R[i][k] + alpha;
            else
                v[i] = R[i][k];
        }

        // Normalize v
        double norm_v = 0.0;
        #pragma omp parallel for reduction(+:norm_v)
        for (int i = k; i < m; i++) {
            norm_v += v[i] * v[i];
        }
        norm_v = sqrt(norm_v);
        #pragma omp parallel for schedule(static)
        for (int i = k; i < m; i++) {
            v[i] /= norm_v;
        }

        // Apply the Householder reflection to R
        #pragma omp parallel for schedule(static)
        for (int j = k; j < n; j++) {
            double s = 0.0;
            for (int i = k; i < m; i++) {
                s += v[i] * R[i][j];
            }
            s *= 2.0;
            for (int i = k; i < m; i++) {
                R[i][j] -= s * v[i];
            }
        }

        // Apply the Householder reflection to Q
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < m; j++) {
            double s = 0.0;
            for (int i = k; i < m; i++) {
                s += v[i] * Q[i][j];
            }
            s *= 2.0;
            for (int i = k; i < m; i++) {
                Q[i][j] -= s * v[i];
            }
        }

        delete[] v;

        track_progress(k, n, "OpenMP");
    }

    // Transpose Q to get the correct orthogonal matrix
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            double temp = Q[i][j];
            Q[i][j] = Q[j][i];
            Q[j][i] = temp;
        }
    }
}

/* --------------------- Matrix Magnitude Calculation using OpenMP --------------------- */
double matrix_magnitude_openmp(double* A, int m, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < m * n; i++) {
        sum += A[i] * A[i];
    }
    return sqrt(sum);
}

/* ----------------------- CPU Version of Householder QR ----------------------- */
void householder_cpu(double** A, double** Q, double** R, int m, int n) {
    // Initialize R to A
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            R[i][j] = A[i][j];

    // Initialize Q to identity matrix
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            Q[i][j] = (i == j) ? 1.0 : 0.0;

    for (int k = 0; k < n; k++) {
        // Compute the norm of the k-th column below row k
        double norm_x = 0.0;
        for (int i = k; i < m; i++) {
            norm_x += R[i][k] * R[i][k];
        }
        norm_x = sqrt(norm_x);

        if (norm_x == 0.0) continue;  // Skip if the column is all zeros

        // Adjust sign to ensure positive diagonal in R
        double s = (R[k][k] >= 0) ? 1.0 : -1.0;
        double alpha = s * norm_x;

        // Compute the Householder vector v
        double* v = new double[m];
        for (int i = 0; i < m; i++) {
            if (i < k)
                v[i] = 0.0;
            else if (i == k)
                v[i] = R[i][k] + alpha;
            else
                v[i] = R[i][k];
        }

        // Normalize v
        double norm_v = 0.0;
        for (int i = k; i < m; i++) {
            norm_v += v[i] * v[i];
        }
        norm_v = sqrt(norm_v);
        for (int i = k; i < m; i++) {
            v[i] /= norm_v;
        }

        // Apply the Householder reflection to R
        for (int j = k; j < n; j++) {
            double s = 0.0;
            for (int i = k; i < m; i++) {
                s += v[i] * R[i][j];
            }
            s *= 2.0;
            for (int i = k; i < m; i++) {
                R[i][j] -= s * v[i];
            }
        }

        // Apply the Householder reflection to Q
        for (int j = 0; j < m; j++) {
            double s = 0.0;
            for (int i = k; i < m; i++) {
                s += v[i] * Q[i][j];
            }
            s *= 2.0;
            for (int i = k; i < m; i++) {
                Q[i][j] -= s * v[i];
            }
        }

        delete[] v;
    }

    // Transpose Q to get the correct orthogonal matrix
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            double temp = Q[i][j];
            Q[i][j] = Q[j][i];
            Q[j][i] = temp;
        }
    }

    // Ensure full progress is displayed at the end
    track_progress(n, n, "CPU");
}

/* --------------------- Compute Q from Householder Vectors --------------------- */
void compute_Q(double** v_list, int m, int n, double** Q) {
    // Initialize Q as identity matrix
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            Q[i][j] = (i == j) ? 1.0 : 0.0;

    // Apply Householder transformations in reverse order
    for (int k = n - 1; k >= 0; k--) {
        double* v = v_list[k]; // Length m

        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (int j = k; j < m; j++) {
                s += v[j] * Q[j][i];
            }
            s *= 2.0;
            for (int j = k; j < m; j++) {
                Q[j][i] -= s * v[j];
            }
        }
    }
}

/* --------------------- Extract R from Transformed A --------------------- */
void extract_R(double** R_full, int m, int n, double** R) {
    // Copy upper triangular part of R_full into R
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            R[i][j] = R_full[i][j];
        }
        for (int j = 0; j < i; j++) {
            R[i][j] = 0.0;
        }
    }
}


/* --------------------- Matrix Printing Function --------------------- */
void print_matrix(const char* name, double** mat, int rows, int cols) {
    cout << name << " = \n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << setw(10) << setprecision(4) << mat[i][j] << " ";
        }
        cout << endl;
    }
}

/* --------------------- Optimized MPI Version of Householder QR --------------------- */
void householder_mpi(double* a, int m, int n, int rank, int size) {
    int i;

    // Calculate rows per process (handling uneven division)
    int rows_per_proc = (m + size - 1) / size;  // Ceiling division
    int start_row = rank * rows_per_proc;
    int local_rows = std::min(rows_per_proc, m - start_row);

    // Allocate local matrix chunk and vectors
    double* local_a = new double[local_rows * n];  // Local matrix chunk for each process
    double* local_v = new double[local_rows];      // Local portion of the vector 'v'
    double* local_x = new double[local_rows];      // Local portion of the vector 'x'

    // Compute sendcounts and displacements for scattering/gathering
    int* sendcounts = new int[size];
    int* displs = new int[size];

    for (i = 0; i < size; i++) {
        int rows = std::min(rows_per_proc, m - i * rows_per_proc);
        sendcounts[i] = rows * n;
        displs[i] = i * rows_per_proc * n;
    }

    // Scatter the matrix 'a' to all processes
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE, local_a, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Set the number of OpenMP threads per process
    int total_cores = omp_get_num_procs(); // Total cores available
    int cores_per_process = total_cores / size; // Number of cores per MPI process
    if (cores_per_process < 1) cores_per_process = 1;
    omp_set_num_threads(cores_per_process);

    double* local_vTa_array = new double[n];
    double* global_vTa_array = new double[n];

    for (i = 0; i < n; i++) {
        // Compute norm_x and local_x0
        double local_sum_squares = 0.0;
        double local_x0 = 0.0;

        #pragma omp parallel for reduction(+:local_sum_squares) reduction(+:local_x0)
        for (int row = 0; row < local_rows; row++) {
            int global_row = start_row + row;
            if (global_row >= i) {
                double x = local_a[row * n + i];
                local_x[row] = x;
                local_sum_squares += x * x;
                if (global_row == i) {
                    local_x0 = x;
                }
            } else {
                local_x[row] = 0.0;
            }
        }

        // Compute global norm_x and x0
        double global_sum_squares = 0.0;
        double global_x0 = 0.0;
        double local_data[2] = { local_sum_squares, local_x0 };
        double global_data[2];

        MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_sum_squares = global_data[0];
        global_x0 = global_data[1];

        double norm_x = sqrt(global_sum_squares);

        if (norm_x == 0.0) continue;

        // Compute beta
        double s = (global_x0 >= 0) ? 1.0 : -1.0;
        double alpha = s * norm_x;
        double beta = global_x0 + alpha;

        // Compute local_v and vnorm_sq
        double local_vnorm_sq = 0.0;
        #pragma omp parallel for reduction(+:local_vnorm_sq)
        for (int row = 0; row < local_rows; row++) {
            int global_row = start_row + row;
            if (global_row >= i) {
                local_v[row] = (global_row == i) ? beta : local_x[row];
                local_vnorm_sq += local_v[row] * local_v[row];
            } else {
                local_v[row] = 0.0;
            }
        }

        // Compute global vnorm_sq
        double global_vnorm_sq = 0.0;
        MPI_Allreduce(&local_vnorm_sq, &global_vnorm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double vnorm = sqrt(global_vnorm_sq);

        // Normalize local_v
        #pragma omp parallel for
        for (int row = 0; row < local_rows; row++) {
            local_v[row] /= vnorm;
        }

        int num_cols = n - i;

        // Compute local vTa for each column
        #pragma omp parallel for
        for (int k = 0; k < num_cols; k++) {
            int j = i + k;
            double temp = 0.0;
            for (int row = 0; row < local_rows; row++) {
                temp += local_v[row] * local_a[row * n + j];
            }
            local_vTa_array[k] = temp;
        }

        // Compute global vTa
        MPI_Allreduce(local_vTa_array, global_vTa_array, num_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Update local_a
        #pragma omp parallel for
        for (int row = 0; row < local_rows; row++) {
            for (int k = 0; k < num_cols; k++) {
                int j = i + k;
                double vTa = 2 * global_vTa_array[k];
                local_a[row * n + j] -= vTa * local_v[row];
            }
        }

        if (rank == 0) {
            track_progress(i, n, "MPI");
        }
    }

    // Gather the updated local_a back to a on root process
    MPI_Gatherv(local_a, local_rows * n, MPI_DOUBLE, a, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Clean up
    delete[] local_a;
    delete[] local_v;
    delete[] local_x;
    delete[] sendcounts;
    delete[] displs;
    delete[] local_vTa_array;
    delete[] global_vTa_array;
}

/* --------------------- CUDA Version of Householder QR --------------------- */
void householder_cuda(double* h_A, double* d_A, double* d_V, int m, int n) {
    for (int i = 0; i < n; i++) {
        cudaMemcpy(d_V, d_A + i * n + i, (m - i) * sizeof(double), cudaMemcpyDeviceToDevice);

        double vnorm = 0;
        norm_kernel<<<(m - i + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_V, &vnorm, m - i);
        cudaDeviceSynchronize();

        if (vnorm != 0) {
            vnorm = sqrt(vnorm);
            scalar_div_kernel<<<(m - i + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_V, vnorm, m - i, d_V);
            cudaDeviceSynchronize();
        }

        for (int j = i; j < n; j++) {
            double vTa = 0;
            dot_product_kernel<<<(m - i + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_A + j * n + i, d_V, &vTa, m - i);
            cudaDeviceSynchronize();

            vTa *= 2;
            scalar_sub_kernel<<<(m - i + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_V, vTa, m - i, d_A + j * n + i);
            cudaDeviceSynchronize();
        }

        track_progress(i, n, "CUDA");
    }
    cudaMemcpy(h_A, d_A, m * n * sizeof(double), cudaMemcpyDeviceToHost);
}

/* --------------------- Matrix Multiplication Function --------------------- */
double** multiply_matrices(double** A, double** B, int m, int n, int p) {
    // A is m x n
    // B is n x p
    // Result is m x p
    double** C = new double*[m];
    for (int i = 0; i < m; i++) {
        C[i] = new double[p];
        for (int j = 0; j < p; j++) {
            C[i][j] = 0.0;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            double Aik = A[i][k];
            for (int j = 0; j < p; j++) {
                C[i][j] += Aik * B[k][j];
            }
        }
    }
    return C;
}

/* --------------------- Matrix Difference Norm Function --------------------- */
double compute_matrix_difference(double** A, double** B, int m, int n) {
    double diff_norm = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = A[i][j] - B[i][j];
            diff_norm += diff * diff;
        }
    }
    return sqrt(diff_norm);
}

/* --------------------- Matrix Export to CSV Function --------------------- */
void export_matrices_to_csv(const std::string& filename, double** original_A, double** Q, double** R, double** reconstructed_A, int m, int n) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Unable to open file " << filename << " for writing.\n";
        return;
    }

    // Write Original Matrix A
    file << "Original Matrix A\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            file << original_A[i][j];
            if (j != n - 1) file << ",";
        }
        file << "\n";
    }
    file << "\n";  // Add an empty line for separation

    // Write Matrix Q
    file << "Matrix Q\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            file << Q[i][j];
            if (j != m - 1) file << ",";
        }
        file << "\n";
    }
    file << "\n";  // Add an empty line for separation

    // Write Matrix R
    file << "Matrix R\n";
    for (int i = 0; i < m; ++i) {  // R is m x n
        for (int j = 0; j < n; ++j) {
            file << R[i][j];
            if (j != n - 1) file << ",";
        }
        file << "\n";
    }
    file << "\n";  // Add an empty line for separation

    // Write Reconstructed Matrix A
    file << "Reconstructed Matrix A\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            file << reconstructed_A[i][j];
            if (j != n - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    cout << "Matrices exported to " << filename << " in sections: 'Original Matrix A', 'Matrix Q', 'Matrix R', and 'Reconstructed Matrix A'.\n";
}

// Function to read matrix from CSV file
bool read_matrix_from_csv(const char* filename, double**& matrix, int& m, int& n) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Unable to open file " << filename << " for reading.\n";
        return false;
    }

    vector<vector<double>> temp_matrix;
    string line;
    int cols = -1;
    while (getline(file, line)) {
        vector<double> row_values;
        stringstream ss(line);
        string cell;
        int current_cols = 0;
        while (getline(ss, cell, ',')) {
            try {
                row_values.push_back(stod(cell));
            } catch (const invalid_argument& ia) {
                cout << "Invalid number found in CSV file.\n";
                file.close();
                return false;
            }
            current_cols++;
        }
        if (cols == -1) {
            cols = current_cols;
        } else if (current_cols != cols) {
            cout << "Inconsistent number of columns in CSV file.\n";
            file.close();
            return false;
        }
        temp_matrix.push_back(row_values);
    }
    file.close();

    m = temp_matrix.size();
    n = cols;

    // Allocate matrix
    matrix = new double*[m];
    for (int i = 0; i < m; i++) {
        matrix[i] = new double[n];
        for (int j = 0; j < n; j++) {
            matrix[i][j] = temp_matrix[i][j];
        }
    }

    return true;
}

/* --------------------- Performance Display Function --------------------- */
double** show_performance(const chrono::duration<double>& cpu_time,
    const chrono::duration<double>& cuda_time,
    const chrono::duration<double>& mpi_time,
    const chrono::duration<double>& omp_time,
    double magnitude_cpu,
    double magnitude_cuda,
    double magnitude_mpi,
    double magnitude_omp,
    double** Q, int m, int n,
    double** R,
    double** a_original) {

    cout << fixed << setprecision(6);

    // Show times in seconds (for consistency)
    double cpu_sec = cpu_time.count();
    double cuda_sec = cuda_time.count();
    double mpi_sec = mpi_time.count();
    double omp_sec = omp_time.count();

    cout.imbue(std::locale("")); // Add commas to numbers
    cout << "\nPerformance Summary:\n";
    cout << "----------------------\n";
    cout << "Matrix Magnitude (CPU): " << magnitude_cpu << "\n";
    cout << "Matrix Magnitude (CUDA): " << magnitude_cuda << "\n";
    cout << "Matrix Magnitude (MPI): " << magnitude_mpi << "\n";
    cout << "Matrix Magnitude (OpenMP): " << magnitude_omp << "\n";

    cout << "\nTime taken by CPU version: " << cpu_sec << " seconds\n";
    cout << "Time taken by CUDA version: " << cuda_sec << " seconds\n";
    cout << "Time taken by MPI version: " << mpi_sec << " seconds\n";
    cout << "Time taken by OpenMP version: " << omp_sec << " seconds\n";

    double speedup_cuda = cpu_sec / cuda_sec;
    double speedup_mpi = cpu_sec / mpi_sec;
    double speedup_omp = cpu_sec / omp_sec;

    cout << "\nSpeedup (CUDA vs CPU): " << speedup_cuda << "x faster\n";
    cout << "Speedup (MPI vs CPU): " << speedup_mpi << "x faster\n";
    cout << "Speedup (OpenMP vs CPU): " << speedup_omp << "x faster\n";

    double percent_improvement_cuda = ((cpu_sec - cuda_sec) / cpu_sec) * 100;
    double percent_improvement_mpi = ((cpu_sec - mpi_sec) / cpu_sec) * 100;
    double percent_improvement_omp = ((cpu_sec - omp_sec) / cpu_sec) * 100;

    cout << "Percentage improvement (CUDA): " << setprecision(2) << percent_improvement_cuda << "%\n";
    cout << "Percentage improvement (MPI): " << setprecision(2) << percent_improvement_mpi << "%\n";
    cout << "Percentage improvement (OpenMP): " << setprecision(2) << percent_improvement_omp << "%\n";

    // Compare accuracies
    double magnitude_diff_cpu_cuda = fabs(magnitude_cpu - magnitude_cuda);
    double magnitude_diff_cpu_mpi = fabs(magnitude_cpu - magnitude_mpi);
    double magnitude_diff_cpu_omp = fabs(magnitude_cpu - magnitude_omp);

    double relative_accuracy_cuda = (magnitude_diff_cpu_cuda / magnitude_cpu) * 100;
    double relative_accuracy_mpi = (magnitude_diff_cpu_mpi / magnitude_cpu) * 100;
    double relative_accuracy_omp = (magnitude_diff_cpu_omp / magnitude_cpu) * 100;

    cout << "\nAccuracy of CUDA version: " << 100.0 - relative_accuracy_cuda << "%\n";
    cout << "Accuracy of MPI version: " << 100.0 - relative_accuracy_mpi << "%\n";
    cout << "Accuracy of OpenMP version: " << 100.0 - relative_accuracy_omp << "%\n";
    cout << "----------------------\n";

    // Create copies of Q and R for display and reconstruction
    double** Q_display = new double*[m];
    double** R_display = new double*[m];
    for (int i = 0; i < m; i++) {
        Q_display[i] = new double[m];
        R_display[i] = new double[n];
        for (int j = 0; j < m; j++) {
            Q_display[i][j] = Q[i][j]; // Copy Q
        }
        for (int j = 0; j < n; j++) {
            R_display[i][j] = R[i][j]; // Copy R
        }
    }

    // Verification of QR Decomposition
    cout << "\nVerification of QR Decomposition:\n";
    // Reconstruct A = Q * R using the copied matrices
    double** reconstructed_A = multiply_matrices(Q_display, R_display, m, m, n); // Q_display is m x m, R_display is m x n
    //Displaye the reconstructed Matrix A
    if (m <= 20 && n <= 20) {
        cout << "\nReconstructed A after multiplying Q and R:\n";
        print_matrix("A", reconstructed_A, m, n);  
    }else {
        cout << "A matrix are too large to display.\n";
    }
    // Compute difference between original A and reconstructed A
    double reconstruction_error = compute_matrix_difference(a_original, reconstructed_A, m, n);
    cout << "Reconstruction error (norm of difference between A and Q*R): " << reconstruction_error << "\n";

    // Display Q and R matrices at the bottom
    if (m <= 20 && n <= 20) {
        cout << "\nQ matrix after QR decomposition:\n";
        print_matrix("Q", Q_display, m, m); // Use Q_display
        cout << "\nR matrix after QR decomposition:\n";
        print_matrix("R", R_display, m, n); // Use R_display
    } else {
        cout << "Q and R matrices are too large to display.\n";
    }

    // Clean up reconstructed_A, Q_display, and R_display
    for (int i = 0; i < m; i++) {
        delete[] Q_display[i];
        delete[] R_display[i];
    }
    delete[] Q_display;
    delete[] R_display;

    return reconstructed_A;
}


/* --------------------- Main Program --------------------- */
int main(int argc, char* argv[]) {
    int m, n;
    int rank, size;
    string filename;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double** a_cpu = nullptr;
    double** a_omp = nullptr;
    double** Q = nullptr;
    double** R = nullptr;
    double* h_A = nullptr;
    double* d_A = nullptr;
    double* d_V = nullptr;
    double** temp_matrix = nullptr;

    double finalMagnitude_cpu = 0, finalMagnitude_cuda = 0, finalMagnitude_mpi = 0, finalMagnitude_omp = 0;
    chrono::duration<double> elapsed_cpu, elapsed_cuda, elapsed_mpi, elapsed_omp;

    int input_choice = 0; // Declare input_choice outside the rank == 0 block

    if (rank == 0) {
        cout << "Choose the way to input the matrix A:\n";
        cout << "1) Input the matrix elements manually\n";
        cout << "2) Read the matrix from a CSV file\n";
        cout << "3) Use default values\n";
        cout << "Enter your choice (1/2/3): ";
        cin >> input_choice;

        if (input_choice == 1) {
            // Manually input the matrix dimensions
            cout << "Enter the dimension m (rows) where A is a m by n matrix: ";
            cin >> m;
            cout << "Enter the dimension n (columns) where A is a m by n matrix: ";
            cin >> n;
        } else if (input_choice == 2) {
            // Read from CSV file
            cout << "Enter the filename of the CSV file containing the matrix A: ";
            cin >> filename;
            if (!read_matrix_from_csv(filename.c_str(), temp_matrix, m, n)) {
                cout << "Failed to read the matrix from the CSV file. Terminating program.\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Display the matrix read from the CSV file
            cout << "\nMatrix A read from " << filename << ":\n";
            if (m <= 20 && n <= 20) {
                print_matrix("A", temp_matrix, m, n);
            } else {
                cout << "Matrix is too large to display.\n";
            }
        } else if (input_choice == 3) {
            // Use default values and input dimensions
            cout << "Enter the dimension m (rows) where A is a m by n matrix: ";
            cin >> m;
            cout << "Enter the dimension n (columns) where A is a m by n matrix: ";
            cin >> n;
        } else {
            cout << "Invalid choice. Terminating program.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (m < n) {
            cout << "For a successful factorization, this implementation requires n <= m.\nTerminating program.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Broadcast matrix size to all MPI processes */
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double** a_original = nullptr;

    if (rank == 0) {
        // Allocate memory for matrices
        a_cpu = new double*[m];
        a_omp = new double*[m];
        Q = new double*[m];
        R = new double*[m];
        h_A = new double[m * n];

        a_original = new double*[m]; // For verification

        for (int i = 0; i < m; i++) {
            a_cpu[i] = new double[n];
            a_omp[i] = new double[n];
            Q[i] = new double[m];
            R[i] = new double[n];
            a_original[i] = new double[n]; // For verification
        }

        if (input_choice == 2) {
            // Copy temp_matrix into a_cpu
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    a_cpu[i][j] = temp_matrix[i][j];
                }
            }
            // Free temp_matrix
            for (int i = 0; i < m; i++) {
                delete[] temp_matrix[i];
            }
            delete[] temp_matrix;
            temp_matrix = nullptr;
        } else {
            // Input or default values
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (input_choice == 1) {
                        cout << "Enter element A[" << i << "][" << j << "]: ";
                        cin >> a_cpu[i][j];
                    } else {
                        a_cpu[i][j] = j - i + 1;  // Default value
                    }
                }
            }
        }

        // Copy data to a_omp and a_original
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_omp[i][j] = a_cpu[i][j];
                a_original[i][j] = a_cpu[i][j];
            }
        }

        // Flatten the matrix for CUDA and MPI versions
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                h_A[i * n + j] = a_cpu[i][j];
            }
        }

        // Allocate memory for CUDA version with error checking
        cudaError_t err = cudaMalloc(&d_A, m * n * sizeof(double));
        if (err != cudaSuccess) {
            cout << "CUDA error: " << cudaGetErrorString(err) << endl;
            return -1;
        }
        cudaMalloc(&d_V, m * sizeof(double));
        cudaMemcpy(d_A, h_A, m * n * sizeof(double), cudaMemcpyHostToDevice);

        // Measure CPU time and magnitude
        auto start_cpu = chrono::high_resolution_clock::now();
        householder_cpu(a_cpu, Q, R, m, n);
        auto end_cpu = chrono::high_resolution_clock::now();
        elapsed_cpu = end_cpu - start_cpu;

        // Flatten the result for magnitude calculation
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                h_A[i * n + j] = R[i][j];
            }
        }
        finalMagnitude_cpu = norm_openmp(h_A, m * n);
        cout << "CPU version completed.\n";

        // Reset h_A to the initial matrix values before starting CUDA version
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                h_A[i * n + j] = a_cpu[i][j];
            }
        }

        // Measure CUDA time and magnitude (Assuming CUDA function is correctly implemented)
        auto start_cuda = chrono::high_resolution_clock::now();
        householder_cuda(h_A, d_A, d_V, m, n);
        auto end_cuda = chrono::high_resolution_clock::now();
        elapsed_cuda = end_cuda - start_cuda;
        finalMagnitude_cuda = norm_openmp(h_A, m * n);
        cout << "CUDA version completed.\n";
        cout.flush();  // Ensure the output is flushed

        // Measure OpenMP time and magnitude
        auto start_omp = chrono::high_resolution_clock::now();
        householder_openmp(a_omp, Q, R, m, n);
        auto end_omp = chrono::high_resolution_clock::now();
        elapsed_omp = end_omp - start_omp;

        // Flatten the result of OpenMP QR decomposition into h_A for magnitude calculation
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                h_A[i * n + j] = R[i][j];
            }
        }

        finalMagnitude_omp = norm_openmp(h_A, m * n);
        cout << "OpenMP version completed.\n";
        cout.flush();  // Ensure the output is flushed

        // Free CUDA memory
        cudaFree(d_A);
        cudaFree(d_V);
    } else {
        h_A = new double[m * n]; // Non-root processes need to allocate h_A
    }

    // Broadcast h_A to all processes
    MPI_Bcast(h_A, m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Measure MPI version (distributed across all processes)
    double* h_A_mpi = new double[m * n]; // Local copy for MPI
    memcpy(h_A_mpi, h_A, m * n * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before starting timing
    auto start_mpi = chrono::high_resolution_clock::now();
    householder_mpi(h_A_mpi, m, n, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize after computation
    auto end_mpi = chrono::high_resolution_clock::now();

    if (rank == 0) {
        elapsed_mpi = end_mpi - start_mpi;
        finalMagnitude_mpi = norm_openmp(h_A_mpi, m * n);
        cout << "MPI version completed.\n";
        cout.flush();  // Ensure the output is flushed
    }

    // Display performance comparison (only rank 0 will print results)
    if (rank == 0) {
        // Create copies of Q and R for display, reconstruction, and export
        double** Q_display = new double*[m];
        double** R_display = new double*[m];
        for (int i = 0; i < m; i++) {
            Q_display[i] = new double[m];
            R_display[i] = new double[n];
            for (int j = 0; j < m; j++) {
                Q_display[i][j] = Q[i][j]; // Copy Q
            }
            for (int j = 0; j < n; j++) {
                R_display[i][j] = R[i][j]; // Copy R
            }
        }

        // Now pass Q_display and R_display to show_performance
        // Now pass Q_display and R_display to show_performance
        double** reconstructed_A = show_performance(elapsed_cpu, elapsed_cuda, elapsed_mpi, elapsed_omp,
            finalMagnitude_cpu, finalMagnitude_cuda, finalMagnitude_mpi, finalMagnitude_omp,
            Q_display, m, n, R_display, a_original);

        // Export matrices to CSV files
        cout << "Do you want to export the matrices A, Q, and R to CSV files? (y/n): ";
        char export_choice;
        cin >> export_choice;
        if (export_choice == 'y' || export_choice == 'Y') {
            string output_filename;
            if (input_choice == 2) {
                // If the matrix was read from a CSV file, create the output filename based on the input filename
                size_t lastindex = filename.find_last_of(".");
                if (lastindex == string::npos) {
                    // No extension found, use the entire filename
                    output_filename = filename + "_output.csv";
                } else {
                    string rawname = filename.substr(0, lastindex);
                    output_filename = rawname + "_output.csv";
                }
            } else {
                // Otherwise, ask for a filename
                cout << "Enter the filename to save the matrices (e.g., output.csv): ";
                cin >> output_filename;
            }
        
            export_matrices_to_csv(output_filename, a_original, Q, R, reconstructed_A, m, n);
            cout << "Matrices exported to " << output_filename << "\n";
        }

        // After exporting or after we're done with reconstructed_A
        if (reconstructed_A != nullptr) {
            for (int i = 0; i < m; i++) {
                delete[] reconstructed_A[i];
            }
            delete[] reconstructed_A;
        }
    }

    // Clean up
    if (rank == 0) {
        for (int i = 0; i < m; i++) {
            delete[] a_cpu[i];
            delete[] a_omp[i];
            delete[] Q[i];
            delete[] R[i];
            delete[] a_original[i]; // Clean up original A
        }
        delete[] a_cpu;
        delete[] a_omp;
        delete[] Q;
        delete[] R;
        delete[] a_original; // Clean up original A
        delete[] h_A;
    }
    delete[] h_A_mpi;

    // Finalize MPI
    MPI_Finalize();

    if (rank == 0) {
        cout << "\nPress Enter to exit...";
        cin.ignore();  // Wait for the user to press Enter
        cin.get();
    }

    return 0;
}