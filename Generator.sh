#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:15:00

module load python/3.9

cd $SCRATCH
cp $SCRATCH/PERM.inc $SCRATCH/PORO.inc $SCRATCH/RESTART.DATA $SCRATCH/RESTARTCOPY.DATA $SLURM_TMPDIR
cd $HOME

source OPMGen/bin/activate

export PATH=$HOME/opm-simulators/install/bin:$PATH

python Train_main.py

cd $SLURM_TMPDIR
mv $SLURM_TMPDIR/RESTARTCOPY.DATA $SCRATCH