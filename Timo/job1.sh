#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120GB
#SBATCH --job-name=test_job_1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=timostrijbis@gmail.com
# ensure that there are now left-over modules loaded from previous jobs


#load the python module
module load Python/3.8.6-GCCcore-10.2.0

# load your previously created virtual environment
source /home4/$USER/.envs/my_env/bin/activate


# make sure to load all needed modules after activating your virtual environment now
# this ensures that all the dependencies are loaded correctly inside the environment
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
pip install transformers datasets evaluate accalerate
pip install -U datasets transformers
pip install accelerate>=0.26.0

# move cached datasets to the /scratch directory
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

# move downloaded models and tokenizers to the /scratch directory
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/hub"

# start your program
python3 train.py
