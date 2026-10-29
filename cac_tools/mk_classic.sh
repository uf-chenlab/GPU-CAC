cd /blue/ypchen/ntaormina/nep/cac_kokkos
mkdir -p models_classic test_classic
cat > test_classic/in.classic <<"EOT"
# variables: data avec pstyle pcoeff nsteps rebuild
units metal
boundary s s s
atom_style ${avec}
comm_style cac
newton off
read_data ${data}
pair_style ${pstyle}
pair_coeff ${pcoeff}
timestep 0.001
compute Ntemp all cac/nodal/temp
velocity all create 300 4928 dist gaussian
fix NVE all cac/nve
thermo_style custom step c_Ntemp pe press
thermo_modify format float %24.16e
thermo ${nsteps}
if "${rebuild} == 1" then "neigh_modify every 1 delay 0 check no"
run ${nsteps}
EOT
source tools/dev_env.sh
T=tools/make_cac_model.py; cd models_classic
$PY ../$T --lattice fcc --a 3.615 --masses 63.546 --blocks 4 4 4 --block 4 --core 1.0 -o cu_atoms.data | tail -1
$PY ../$T --lattice fcc --a 3.615 --masses 63.546 --blocks 4 4 4 --block 4 --core 0.0 -o cu_elem4.data | tail -1
$PY ../$T --lattice fcc --a 3.615 --masses 63.546 --blocks 2 2 2 --block 8 --core 0.0 -o cu_elem8.data | tail -1
$PY ../$T --lattice fcc --a 3.615 --masses 63.546 --blocks 4 4 4 --block 4 --core 0.34 -o cu_mix4.data | tail -1
$PY ../$T --lattice diamond --a 5.431 --masses 28.0855 --blocks 3 3 3 --block 4 --core 1.0 -o si_atoms.data | tail -1
$PY ../$T --lattice diamond --a 5.431 --masses 28.0855 --blocks 3 3 3 --block 4 --core 0.0 -o si_elem4.data | tail -1
$PY ../$T --lattice diamond --a 5.431 --masses 28.0855 --blocks 2 2 2 --block 6 --core 0.0 -o si_elem6.data | tail -1
$PY ../$T --lattice diamond --a 5.431 --masses 28.0855 --blocks 3 3 3 --block 4 --core 0.34 -o si_mix4.data | tail -1
cp ../bench/inputs/Cu_u3.eam ../bench/inputs/Cu_mishin1.eam.alloy ../bench/inputs/CuZr_mm.eam.fs ../bench/inputs/Si.sw ../bench/inputs/Si.tersoff .
