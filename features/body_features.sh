#!/bin/bash

for i in {0..255}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=bodyfeatures_$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH -G1
#SBATCH --cpus-per-task=1
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out

# loading uv virtual environment
source /home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate

python body_features.py --index "$i" \
                          --batch_size 50 \
                          --pose_path /path/to/pose/file/directory/ \
                          --pose_features_path /path/to/pose/features/file/directory/ \
                          --time_limit 257890

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


