#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>   
#include "utility.h"

//csr requires a row grouping, so it converts the coo grouping
int create_sparse_csr_from_coo(int nrows, int ncols, int nnz, int *coo_rows, int *coo_cols, double *coo_vals, sparse_csr *csr){
    csr->nrows = nrows;   //store matrix dimensions
    csr->ncols = ncols;
    csr->nnz   = nnz;

    csr->row_ptr = calloc(nrows + 1, sizeof(size_t)); //allocate CSR arrays
    csr->col_ind = malloc(nnz * sizeof(size_t));
    csr->val     = malloc(nnz * sizeof(double));

    if ( !csr->row_ptr || !csr->col_ind || !csr->val ) { //check failures
        return EXIT_FAILURE; 
    }

    int *row_count = calloc(nrows, sizeof(int));   //temporary array to count nnz per row

    for (int k = 0; k < nnz; k++) {   //counts each non-zero row element
        row_count[coo_rows[k]]++;
    }

    csr->row_ptr[0] = 0;     //build the row_ptr array           
    for (int i = 0; i < nrows; i++) {
        csr->row_ptr[i + 1] = csr->row_ptr[i] + row_count[i];
    }

    for (int i = 0; i < nrows; i++) {  //reset row_count
        row_count[i] = 0;
    }

    for (int k = 0; k < nnz; k++) {    //fill col ind e val in CSR
        int row = coo_rows[k];
        int dest = csr->row_ptr[row] + row_count[row];
        csr->col_ind[dest] = coo_cols[k];
        csr->val[dest] = coo_vals[k];
        row_count[row]++;
    }

    free(row_count);
    return EXIT_SUCCESS;
}

//prints the matrix values
void print_sparse_csr(const sparse_csr* csr){
    printf("row\tcol\tval\n");
    printf("---\n");
    for (size_t i = 0; i < csr->nrows; ++i) {
        size_t nz_start = csr->row_ptr[i];
        size_t nz_end = csr->row_ptr[i + 1];
        for(size_t nz_id = nz_start; nz_id < nz_end; ++nz_id){
            size_t j = csr->col_ind[nz_id];
            double val = csr->val[nz_id];
            printf("%d\t%d\t%02.2f\n", i, j, val);
        }
    }
}

// linear spmv multiplies eache nnz with each v element
void matrix_vector_sparse_csr(const sparse_csr* csr, const double* v, double* res){
    for (size_t i = 0; i < csr->nrows; ++i) {
        res[i] = 0.0;
        size_t nz_start = csr->row_ptr[i];
        size_t nz_end = csr->row_ptr[i + 1];
        for(size_t nz_id = nz_start; nz_id < nz_end; ++nz_id){
            size_t j = csr->col_ind[nz_id];
            double val = csr->val[nz_id];
            res[i] = res[i] + val * v[j];
        }
    }
}

//deallocates the arrays of csr struct
int free_sparse_csr(sparse_csr* csr) {
    free(csr->row_ptr);
    free(csr->col_ind);
    free(csr->val);

    return EXIT_SUCCESS;
}

//-----------------------------------------------------MPI FUNCTIONS---------------------------------
//identifing ghost in global columns
int check_ghost_entries(const sparse_csr* csr, int rank, int size, int** ghost_cols, int* nghost){
    int max_possible = csr->nnz;  
    int* temp = malloc(max_possible * sizeof(int));
    if (!temp) {
        return EXIT_FAILURE;
    }

    int count = 0;
    for (int k = 0; k < csr->nnz; k++) {
        int j = csr->col_ind[k];    //global column

        //checking if the column is remote
        if (j % size != rank) {
            int found = 0;
            //checking doubled values
            for (int t = 0; t < count; t++) {
                if (temp[t] == j) {
                    found = 1;
                    break;
                }
            }

            if (!found) {
                temp[count++] = j;
            }
        }
    }

    *ghost_cols = malloc(count * sizeof(int));
    if (!(*ghost_cols)) {
        free(temp);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < count; i++)
        (*ghost_cols)[i] = temp[i];

    *nghost = count;
    free(temp);

    return EXIT_SUCCESS;
}

// ghosts are grouped by owning rank and exchanged using collectives, with a detrministic order, avoiding deadlock
int exchange_ghosts(const double* x_local, double** x_ghost, int ncols_local, const int* ghost_cols, int nghost, int rank, int size, MPI_Comm comm)
{
    *x_ghost = NULL;
    //no ghosts --> no communication
    if (nghost == 0) return EXIT_SUCCESS;

    double *ghost_vals = calloc(nghost, sizeof(double));

    int *send_counts = calloc(size, sizeof(int));
    int *recv_counts = calloc(size, sizeof(int));

    //counts how many ghosts we need from each rank
    for (int i = 0; i < nghost; i++) {
        int owner = ghost_cols[i] % size;
        
        if (owner != rank) {
            send_counts[owner]++;
        }
    }

    //exchanging the counts
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);

    int *send_displs = calloc(size, sizeof(int));
    int *recv_displs = calloc(size, sizeof(int));

    //offsets for data
    for (int r = 1; r < size; r++) {
        send_displs[r] = send_displs[r-1] + send_counts[r-1];
        recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
    }

    int send_tot = send_displs[size-1] + send_counts[size-1];
    int recv_tot = recv_displs[size-1] + recv_counts[size-1];

    //allocating the globa list of indexes
    int *send_idx = malloc(send_tot * sizeof(int));
    int *pos = malloc(size * sizeof(int));
    memcpy(pos, send_displs, size * sizeof(int));

    for (int i = 0; i < nghost; i++) {
        int owner = ghost_cols[i] % size;
        if (owner != rank) {
            send_idx[pos[owner]++] = ghost_cols[i];
        }
    }

    int *recv_idx = malloc(recv_tot * sizeof(int));
    MPI_Alltoallv(send_idx, send_counts, send_displs, MPI_INT, recv_idx, recv_counts, recv_displs, MPI_INT, comm);

    //allocating the values
    double *send_vals = malloc(recv_tot * sizeof(double));
    for (int i = 0; i < recv_tot; i++) {
        send_vals[i] = x_local[ recv_idx[i] / size ];
    }
        
    //exchanging the values
    double *recv_vals = malloc(send_tot * sizeof(double));
    MPI_Alltoallv(send_vals, recv_counts, recv_displs, MPI_DOUBLE, recv_vals, send_counts, send_displs, MPI_DOUBLE, comm);

    //ordering values back to ghost order
    int k = 0;
    for (int r = 0; r < size; r++) {
        for (int i = 0; i < nghost; i++) {
            if (ghost_cols[i] % size == r) {
                ghost_vals[i] = recv_vals[k++];
            }
        }
    }

    free(send_counts); free(recv_counts);
    free(send_displs); free(recv_displs);
    free(send_idx); free(recv_idx);
    free(send_vals); free(recv_vals);
    free(pos);

    *x_ghost = ghost_vals;
    return EXIT_SUCCESS;
}


static inline double get_x_value(int j_global, const double* x_local, const int* ghost_pos, const double* x_ghost, int rank, int size){
    if (j_global % size == rank) {
        // local column
        int j_loc = j_global / size;
        return x_local[j_loc];
    } else {
        // ghost column: O(1) lookup
        int pos = ghost_pos[j_global];
        return (pos >= 0) ? x_ghost[pos] : 0.0;
    }
}


//mpi version of spmv
void spmv_csr_mpi(const sparse_csr* csr, const double* x_local, const int* ghost_pos, const double* x_ghost, int rank, int size, double* y_local){
    for (int i = 0; i < (int)csr->nrows; i++) {
        double sum = 0.0;

        for (size_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            int j = (int)csr->col_ind[k];
            double a = csr->val[k];

            double x = get_x_value(j, x_local, ghost_pos, x_ghost, rank, size);
            sum += a * x;
        }

        y_local[i] = sum;
    }
}


//checks the result of the sequential calulation and parallel version
int check_results(const double *res_seq, const double *res_par, size_t n) {
    const double eps = 1e-8;
    for (size_t i = 0; i < n; i++) {
        if (fabs(res_seq[i] - res_par[i]) > eps) {
            printf("Error: mismatch at index %zu : seq = %f, par = %f\n", i, res_seq[i], res_par[i]);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}


