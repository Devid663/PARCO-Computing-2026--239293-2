SpMV operation using CSR matrix format with MPI and 1D row partitioning
# Setup
SSH Client: Use MobaXterm for Windows. For macOS and Linux, the built-in SSH client is sufficient. VPN: To access the University network from an external network, establish a secure connection using a VPN.

!To correctly run the experiments, it is **mandatory to manually download and place a matrix inside the `data/` directory**. The matrix must be **regular (i.e., not symmetric, not Hermitian, etc.)**. This requirement is due to **GitHub file size limitations**, which prevent large matrices from being included directly in the repository.


# Compiling
1. Access the cluster:
  <pre>ssh username@hpc.unitn.it</pre>
2. Reserve a Node and Enter an Interactive Session: on a node with the wanted specification. 
3. Clone the repo
<pre>git clone https://github.com/Devid663/PARCO-Computing-2026--239293.git</pre>
4. Move to head node run to submit a request using the **spmv.pbs** in the /scripts directory. It will automatically run all the experiment executions (Pay attention to the note above). Submit the job to the queue for compile and execution: 
<pre>qsub commands.pbs</pre>
5. View the results in the /results directory.
If you want to modify the file, access the /src directory and re-compile the code:
<pre>mpicc -std=c99 -O3 -fopenmp main.c utility.c mmio.c -o spmv_mpi_exec</pre>
after loaded the modules:
<pre>module load gcc91
module load mpich-3.2.1--gcc-9.1.0</pre>

# Directory

<pre>
├── README.md               
├── data/
│   ├── matrix1.mtx
│   ├── matrix2.mtx
│   └── ...
├── src/
│   ├── main.c
│   ├── utility.c
│   ├── mmio.c
|   ├── mmio.h
│   └── utility.h
├── scripts/
│   ├── spmv.pbs             
├── results/
│   ├── logs.txt
│   ├── spmv_mpi_exec
│   ├── ...
│   └── spmv_strong_results.csv
└── report.pdf

</pre>
