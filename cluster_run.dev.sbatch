#!/bin/bash

#SBATCH -J osimrl
#SBATCH -e slurm.err
#SBATCH -N 2 # number of nodes
#SBATCH -n 20 # number of cores
#SBATCH --tasks-per-node=20
#SBATCH --mem 18G # memory pool for all cores
#SBATCH --mail-user=florin.dzeladini@epfl.ch
#SBATCH --mail-type=fail
#SBATCH --time=02:00:00

echo "running on hosts: $SLURM_NODELIST"
echo "running on $SLURM_NNODES nodes."
echo "running $SLURM_NTASKS tasks."
echo "current working directory is `pwd`"

# setup simbody path (Please change SIMBODY_HOME accordingly!)
export SIMBODY_HOME=/shared-dev/$USER/opensim-core/dependencies/simbody
export Simbody_DIR=$SIMBODY_HOME/lib/cmake/simbody
export PATH=$SIMBODY_HOME/libexec/simbody:$PATH
export LD_LIBRARY_PATH=$SIMBODY_DIR/lib:$LD_LIBRARY_PATH

# setup opensim-core path (Please change OPENSIM_HOME accordingly!)
export OPENSIM_HOME=/shared-dev/$USER/opensim-core/
export OpenSim_DIR=$OPENSIM_HOME/lib/cmake/OpenSim
export OpenSim_DIR=$OPENSIM_HOME
export LD_LIBRARY_PATH=$OPENSIM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$OPENSIM_HOME/bin:$PATH

# activate opensim-evn (Please change PATH accordingly!)
source /shared-dev/$USER/venv/bin/activate
cd /shared-dev/$USER/osimrl_biorob
# run script and create SLURM_NTASKS subprocesses
mpirun -np $SLURM_NTASKS python optimization_spinalCtrl.py -n $((SLURM_NTASKS-1)) -t CMAES -g 100 --duration=10.0


echo "Program finished"
