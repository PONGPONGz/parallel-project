#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to generate random integers
void generate_random_numbers(int* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = rand() % 100000;  // Random number between 0 and 99999
    }
}

// Merge function to combine two halves
void merge(int arr[], int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = left; // Initial index of merged subarray

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

// Merge sort function
void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Sort first and second halves
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);

        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);  // Seed the random number generator

    // Array of data sizes to be tested
    int data_sizes[] = {10000, 100000, 500000, 1000000, 2000000};
    int threshold = 1000;

    for (int s = 0; s < 5; s++) {
        int n = data_sizes[s];
        int* data = NULL;

        if (rank == 0) {
            // Generate random data on the root process
            data = (int*)malloc(n * sizeof(int));
            generate_random_numbers(data, n);
        }

        // Broadcast the data size to all nodes
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Divide data among nodes
        int local_n = n / size;
        int* local_data = (int*)malloc(local_n * sizeof(int));

        // Scatter data from root to all processes
        MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        // Start timing the local sort
        double start_time = MPI_Wtime();

        // Sort the local data using merge sort
        merge_sort(local_data, 0, local_n - 1);

        // End timing
        double end_time = MPI_Wtime();
        double local_sort_time = end_time - start_time;

        // Gather sorted data back to the root process
        MPI_Gather(local_data, local_n, MPI_INT, data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        // On the root, merge all sorted parts and calculate the final sort time
        double total_sort_time = 0.0;
        if (rank == 0) {
            // Start timing the final merge on the root process
            double final_merge_start = MPI_Wtime();

            // Final merge of the sorted data
            merge_sort(data, 0, n - 1);

            // End timing the final merge
            double final_merge_end = MPI_Wtime();
            total_sort_time = (final_merge_end - final_merge_start) + local_sort_time;
        }

        // Gather the timing results on the root process
        double max_local_sort_time;
        MPI_Reduce(&local_sort_time, &max_local_sort_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Data size: %d, Nodes: %d, Max Local Sort Time: %f, Total Sort Time: %f\n", n, size, max_local_sort_time, total_sort_time);
        }

        free(local_data);
        if (rank == 0) {
            free(data);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
