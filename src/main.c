#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "mmio.h"
#include "utility.h"

int main(int argc, char** argv) {
    srand(time(NULL));

    //benchmark metrics variables
    double t_start, t_end;
    double t_local, t_global;

    int ret_code;

    MPI_Init(&argc, &argv);

    int rank, size; //rank of the process and number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nrows_global = 0;   //matrix global row dimension
    int ncols_global = 0;   //matrix global column dimension
    int nz_global;   //matriz global non-zero dimension

    //COO dimensions (rank 0 only)
    int    *I_global = NULL;    //global coo rows 
    int    *J_global = NULL;    //global coo columns    
    double *VAL_global = NULL;  //global coo values

    //COO dimension (locals)
    int *I_local = NULL;    //local rows
    int *J_local = NULL;    //local columns
    double *VAL_local = NULL;   //local values

    if (argc < 2) {
        printf("Usage: %s matrix_file.mtx\n", argv[0]);
        return EXIT_FAILURE;
    }

    //rank 0 reads the Matrix Market file
    if(rank == 0){
        FILE *f;
        MM_typecode matcode;
        mm_initialize_typecode(&matcode);

        //opens the matrix.mtx file
        if ((f = fopen(argv[1], "r")) == NULL) {
            perror("Cannot open file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Opened file successfully!\n");


        //reading the headers of the matrix.mtx
        if (mm_read_banner(f, &matcode) != 0) {
            printf("Could not process Matrix Market banner.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        //matrix pattern types
        int is_pattern = mm_is_pattern(matcode);  
        int is_real    = mm_is_real(matcode);
        int is_integer = mm_is_integer(matcode);
        int is_complex = mm_is_complex(matcode);

        //accepts only real/integer/pattern type
        if (!mm_is_coordinate(matcode)) {
            fprintf(stderr, "Error: this code only supports COO (coordinate) format.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (is_complex) {
            fprintf(stderr, "Error: complex matrices are not supported.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!mm_is_general(matcode)) {
            fprintf(stderr, "Error: symmetric/skew/hermitian matrices not supported.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        //read matrix dimensions
        if ((ret_code = mm_read_mtx_crd_size(f, &nrows_global, &ncols_global, &nz_global)) != 0)  
            return ret_code;
        
        printf("Matrix dimensions: %d x %d with %d non-zero entries\n", nrows_global, ncols_global, nz_global);

        //allocation of coo rapresentation
        I_global = malloc(nz_global * sizeof(int));  
        J_global = malloc(nz_global * sizeof(int));
        VAL_global = malloc(nz_global * sizeof(double));

        //read the coo entries from the file
        for (int k = 0; k < nz_global; k++) {  
            int row, col;
            double value = 1.0;  
        
            if (is_pattern) {  //if is in pattern type values are implicitly 1
                if (fscanf(f, "%d %d", &row, &col) != 2) {
                    fprintf(stderr, "Error reading pattern entry %d\n", k);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else {
                if (fscanf(f, "%d %d %lf", &row, &col, &value) != 3) {
                    fprintf(stderr, "Error reading valued entry %d\n", k);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            
            I_global[k]   = row - 1;  //converts the Matrix indices
            J_global[k]   = col - 1;
            VAL_global[k] = value;
        }
        fclose(f);
    }

    //broadcast dimension
    MPI_Bcast(&nrows_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ncols_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    //calulate the numbero of local rows  
    int nrows_local = 0;
    for (int i = rank; i < nrows_global; i += size) {
        nrows_local++;
    }

    //rank 0 counts and distrivure the non-zero entries base on the owner: owner(row) = row % size
    //counting local nnzs (rank 0)
    int nz_local = 0;
    int *nz_rank = NULL;
    if (rank == 0) {
        nz_rank = (int*)calloc((size_t)size, sizeof(int));

        for (int k = 0; k < nz_global; k++){
            int owner = I_global[k] % size;
            nz_rank[owner]++;
        }
        /*
        printf("Rank 0 nz_rank: ");
        for (int r = 0; r < size; r++) printf("%d ", nz_rank[r]);
        printf("\n");
        fflush(stdout);
        */

        nz_local = nz_rank[0];
        //sending nz_rank
        for (int dest = 1; dest < size; dest++){
            MPI_Send(&nz_rank[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
        //free(nz_rank);

    } else if (rank != 0) {        
        MPI_Status status;
        //receiving nz_rank from rank 0
        MPI_Recv(&nz_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        //printf("Rank %d: nz_local = %d\n", rank, nz_local);
    }

    I_local  = malloc(nz_local * sizeof(int));
    J_local  = malloc(nz_local * sizeof(int));
    VAL_local = malloc(nz_local * sizeof(double));

    //prepares coo and sends coo buffers
    if (rank == 0) {
        int *offset = calloc(size, sizeof(int)); 
        int** Ibuf = calloc(size, sizeof(int*));
        int** Jbuf = calloc(size, sizeof(int*));
        double** Vbuf = calloc(size, sizeof(double*));


        for (int r = 0; r < size; r++) {
            int count = nz_rank[r];
            if (count > 0) {
                Ibuf[r] = malloc(count * sizeof(int));
                Jbuf[r] = malloc(count * sizeof(int));
                Vbuf[r] = malloc(count * sizeof(double));
            }
        }

        // fills the buffer
        for (int k = 0; k < nz_global; k++) {
            int owner = I_global[k] % size;
            int pos = offset[owner]++;
            Ibuf[owner][pos] = I_global[k];
            Jbuf[owner][pos] = J_global[k];
            Vbuf[owner][pos] = VAL_global[k];
        }

        if (nz_local > 0) {
            memcpy(I_local,   Ibuf[0], (size_t)nz_local * sizeof(int));
            memcpy(J_local,   Jbuf[0], (size_t)nz_local * sizeof(int));
            memcpy(VAL_local, Vbuf[0], (size_t)nz_local * sizeof(double));
        }

        //send to the other ranks
        for (int r = 1; r < size; r++) {
            int cnt = nz_rank[r];
            if (cnt == 0) continue;
            MPI_Send(Ibuf[r], cnt, MPI_INT,    r, 1, MPI_COMM_WORLD);
            MPI_Send(Jbuf[r], cnt, MPI_INT,    r, 2, MPI_COMM_WORLD);
            MPI_Send(Vbuf[r], cnt, MPI_DOUBLE, r, 3, MPI_COMM_WORLD);
        }

        for (int r = 0; r < size; r++) {
            free(Ibuf[r]); free(Jbuf[r]); free(Vbuf[r]);
        }

        free(Ibuf); free(Jbuf); free(Vbuf);
        free(offset);
        free(nz_rank);

    }else if (rank != 0){
        MPI_Status st;
        if (nz_local > 0) {
            MPI_Recv(I_local,   nz_local, MPI_INT,    0, 1, MPI_COMM_WORLD, &st);
            MPI_Recv(J_local,   nz_local, MPI_INT,    0, 2, MPI_COMM_WORLD, &st);
            MPI_Recv(VAL_local, nz_local, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &st);
        } 
    }
    /*
    printf("Rank %d: nrows_local=%d nz_local=%d\n", rank, nrows_local, nz_local);
    if (nz_local > 0) {
        int show = (nz_local < 3) ? nz_local : 3;
        for (int t = 0; t < show; t++) {
            printf("  Rank %d COO[%d] = (%d, %d) %.3f\n", rank, t, I_local[t], J_local[t], VAL_local[t]);
        }
    }
    */
    //converting global row indices into local row indexes
    for (int k = 0; k < nz_local; k++) {
        I_local[k] = I_local[k] / size;
    }

    //local csr 
    sparse_csr csr;
    create_sparse_csr_from_coo(nrows_local, ncols_global, nz_local, I_local, J_local, VAL_local,&csr);
    //if(rank == 0) print_sparse_csr(&csr);
    free(I_local);
    free(J_local);
    free(VAL_local);

    //ghost identification(entries required by the rank that are not avaible)
    int* ghost_cols = NULL;
    int nghost = 0;
    check_ghost_entries(&csr, rank, size, &ghost_cols, &nghost);
    
    int *ghost_pos = malloc(ncols_global * sizeof(int));  
    for (int j = 0; j < ncols_global; j++) ghost_pos[j] = -1;
    for (int k = 0; k < nghost; k++) ghost_pos[ ghost_cols[k] ] = k;

    //building the local vector
    int ncols_local = 0;
    for (int j = rank; j < ncols_global; j += size) {
        ncols_local++;
    }

    double* x_local = malloc(ncols_local * sizeof(double));
    for (int i = 0; i < ncols_local; i++){
        x_local[i] = (double) rand() / RAND_MAX;
    }

    double* x_ghost = NULL;
    double* y_local = malloc(nrows_local * sizeof(double));

    //-------------------------WARM-UP RUN---------------------------
    //printf("Rank %d entering exchange (nghost=%d)\n", rank, nghost);
    //fflush(stdout);
    //exchanging ghosts
    exchange_ghosts(x_local, &x_ghost, ncols_local, ghost_cols, nghost, rank, size, MPI_COMM_WORLD);
    /*
    int gmin, gmax, gsum;
    MPI_Reduce(&nghost, &gmin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nghost, &gmax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nghost, &gsum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("GHOST stats: min=%d avg=%.2f max=%d\n", gmin, (double)gsum/size, gmax);
    }
    */
    //printf("Rank %d leaving exchange\n", rank);
    //fflush(stdout);
    //performin the spmvb
    spmv_csr_mpi(&csr, x_local, ghost_pos, x_ghost, rank, size, y_local);

    free(x_ghost);
    x_ghost = NULL;

    //-------------------------BENCHMARK RUNS---------------------------
    for (int i = 0; i < 5; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
    
        exchange_ghosts(x_local, &x_ghost, ncols_local, ghost_cols, nghost, rank, size, MPI_COMM_WORLD);
    
        double t1 = MPI_Wtime();
    
        spmv_csr_mpi(&csr, x_local, ghost_pos, x_ghost, rank, size, y_local);
    
        double t2 = MPI_Wtime();
    
        double comm = t1 - t0;
        double comp = t2 - t1;
    
        double comm_g, comp_g;
        MPI_Reduce(&comm, &comm_g, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp, &comp_g, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
        if (rank == 0) {
            printf("run=%d comm=%e comp=%e total=%e\n",
                   i, comm_g, comp_g, comm_g + comp_g);
        }
    
        free(x_ghost);
        x_ghost = NULL;
    }


    /*
    double *v = malloc(ncols_global * sizeof(double));  //vector
    for (int i = 0; i < ncols_global; i++) {  //random vector generation
        v[i] = (double) rand() / RAND_MAX;
        //    printf("v[%d] = %g\n", i, v[i]);
    }
    */

    //deallocation
    free(x_local);
    free(y_local);
    free_sparse_csr(&csr);
    free(ghost_pos);
    free(ghost_cols);

    if (rank == 0) {
        free(I_global);
        free(J_global);
        free(VAL_global);
    }

    MPI_Finalize();

    return 0;
}