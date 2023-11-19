#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=detectron2_keypoints
#SBATCH --output=out.slurm
#SBATCH --time=0-12:00:00
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
source ~/.profile
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total of GPUS: $CUDA_VISIBLE_DEVICES"
source ~/.profile
export BASE_DATASET_DIR=/cluster/work/haakohu/datasets
export BASE_OUTPUT_DIR=/cluster/work/haakohu/outputs/fba_new
export FBA_METRICS_CACHE=/cluster/work/haakohu/outputs/fba_new/fba_metrics_cache
export PRETRAINED_CHECKPOINTS_PATH=/cluster/work/haakohu/pretrained_checkpoints
module purge
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Anaconda3/2020.07
# >>> conda initialize >>>
__conda_setup="$('/cluster/apps/eb/software/Anaconda3/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
   if [ -f "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh" ]; then
      . "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
   else
      export PATH="/cluster/apps/eb/software/Anaconda3/2020.07/bin:$PATH"
   fi
fi
unset __conda_setup
conda activate bdd100k-mmdet
nvidia-smi
nvidia-smi nvlink -s
nvidia-smi topo -m
