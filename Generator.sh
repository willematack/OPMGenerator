#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00
#SBATCH --mail-user=elijah.french@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.9

cd $SCRATCH
cp $SCRATCH/PERM.inc $SCRATCH/PORO.inc $SCRATCH/RESTART.DATA $SCRATCH/RESTARTCOPY.DATA $SCRATCH/Transitions $SLURM_TMPDIR
cd $HOME

source OPMGen/bin/activate

export PATH=$HOME/opm-simulators/install/bin:$PATH

python Train_main.py

cd $SLURM_TMPDIR
mv $SLURM_TMPDIR/RESTART.DATA  $SCRATCH