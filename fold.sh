#!/bin/bash 
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --time=11:59:00
#SBATCH --job-name folding
#SBATCH --output=B0329_ex_%j.txt
#SBATCH --mail-type=ALL

#cd /home/p/pen/h`siuhsil/codes/
# EXECUTION COMMAND; -np = nodes*ppn

#mpirun -np 16 python findgp.py

cd /scratch/p/pen/hsiuhsil/vdif_fold

#mpirun -np 32 python mushroom_test.py

#mpirun -np 64 python vdif_fold.py -site 'stk' -T0 '2021-02-05T00:03:31.0000' -duration 383
#mpirun -np 128 python vdif_fold.py -site 'stk' -T0 '2021-02-05T02:08:20.0000' -duration 3603
#mpirun -np 128 python vdif_fold.py -site 'stk' -T0 '2021-02-05T03:08:20.0000' -duration 3603
#mpirun -np 128 python vdif_fold.py -site 'stk' -T0 '2021-02-05T04:08:20.0000' -duration 3603
#mpirun -np 128 python vdif_fold.py -site 'stk' -T0 '2021-02-05T05:08:20.0000' -duration 3603
#mpirun -np 128 python vdif_fold.py -site 'stk' -T0 '2021-02-05T06:08:20.0000' -duration 3603
#mpirun -np 128 python vdif_fold.py -site 'stk' -T0 '2021-02-05T07:08:20.0000' -duration 3603
mpirun -np 128 python vdif_fold.py -site 'stk' -T0 '2021-02-05T08:08:20.0000' -duration 3603
mpirun -np 128 python vdif_fold.py -site 'stk' -T0 '2021-02-05T09:08:20.0000' -duration 3603
#mpirun -np 64 python vdif_fold.py -site 'stk' -T0 '2021-02-05T10:08:20.0000' -duration 3603
#mpirun -np 64 python vdif_fold.py -site 'stk' -T0 '2021-02-05T11:08:20.0000' -duration 2503




