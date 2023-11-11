#!/bin/bash
#SBATCH --account=elijahf
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=1:00:00
#SBATCH --mail-user=elijah.french@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.9
source OPMGen/bin/activate

cd $SCRATCH
cp $SCRATCH/PERM.inc $SCRATCH/PORO.inc $SCRATCH/RESTART.DATA $SCRATCH/RESTARTCOPY.DATA $SLURM_TMPDIR
cd $HOME

export PATH=$HOME/opm-simulators/install/bin:$PATH

python Train_main.py

cd $SLURM_TMPDIR
mv $SLURM_TMPDIR/mem_cntr $SLURM_TMPDIR/actions $SLURM_TMPDIR/states_ $SLURM_TMPDIR/states $SCRATCH
