# source inside the interactive allocation (job 47841126) via: srun --jobid=47841126 --overlap bash -lc "source tools/dev_env.sh; ..."
module purge; module load gcc; module load openmpi/5.0.10; module load cuda/12.8.1; module load cmake
export PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/bin:$PATH
export LD_LIBRARY_PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/lib:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ucx UCX_TLS=^ib
export R=/blue/ypchen/ntaormina/nep/cac_kokkos
export LMP=${LMP:-/blue/ypchen/ntaormina/nep/cac_kokkos/build_cuda/lmp}
export PY=/blue/ypchen/ntaormina/bin/miniconda38/envs/neptools_env/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OMP_PROC_BIND=${OMP_PROC_BIND:-true} OMP_PLACES=${OMP_PLACES:-cores}
