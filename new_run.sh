#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --time=2:00:00
#SBATCH --mem=24GB
#SBATCH -o assignment_3-%j
#SBATCH --export=ALL

export seed=100


export WORKDIR="$HOME/WORK/NLP-with-Deep-Learning/assignment_3"
export SCRDIR="/scratch/general/vast/$USER/2_{$seed}job_{$lr}_nl_{$num_layer}_$batch_size"
export models="Models"

mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR
mkdir $models



source ~/miniconda3/etc/profile.d/conda.sh
conda activate envKD
python ./my_proj.py --seed $seed > output_file

