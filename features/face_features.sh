#!/bin/bash

for i in {0..255}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=facefeatures_$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH -G1
#SBATCH --cpus-per-task=1
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out

# loading uv virtual environment
source /home/kan/Research/SHuBERT/.venv-dino/bin/activate

python dinov2_features.py --index "$i" \
                          --batch_size 5000 \
                          --files_list /path/to/files/list/file/files_list.list \
                          --output_folder /path/to/output/folder/ \
                          --dino_path /path/to/dino/model/dino_faces.pt \
                          --time_limit 257890

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


