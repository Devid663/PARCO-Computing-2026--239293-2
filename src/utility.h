#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>

//struct for the matrix in csr format
typedef struct sparse_CSR {
    size_t nrows; //number of rows
    size_t ncols; //number of columns
    size_t nnz; //number of no zero entries
    size_t* row_ptr; //array of row pointers
    size_t* col_ind; //array of column indices
    double* val; //array of values
} sparse_csr;

//--------------------------------------FUNCTIONS--------------------------------------
//function for buolding the csr
int create_sparse_csr_from_coo(int nrows, int ncols, int nnz, int *coo_rows, int *coo_cols, double *coo_vals, sparse_csr *csr);
//function for printing the matrix
void print_sparse_csr(const sparse_csr* csr);
//sequential spmv operation
void matrix_vector_sparse_csr(const sparse_csr* csr, const double* v, double* res);
//csr deallocation
int free_sparse_csr(sparse_csr* csr);
//function for checking ghost entries (values required from a rank that doesn't have it)
int check_ghost_entries(const sparse_csr* csr, int rank, int size, int** ghost_cols, int* nghost);
//function for exchanging ghost entries (ranks exchange the ghost that they require)
int exchange_ghosts(const double* x_local, double** x_ghost, int ncols_local, const int* ghost_cols, int nghost, int rank, int size, MPI_Comm comm);
//function for retrieving the x values based on the type
static inline double get_x_value(int j_global, const double* x_local, const int* ghost_pos, const double* x_ghost, int rank, int size);
//mpi version of spmv
void spmv_csr_mpi(const sparse_csr* csr, const double* x_local, const int* ghost_pos, const double* x_ghost, int rank, int size, double* y_local);
//function for cheching the results
int check_results(const double *res_seq, const double *res_par, size_t n);



#endif