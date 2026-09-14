#!/bin/bash

for i in {0..255}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=cropface_$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH --cpus-per-task=1
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out

# loading uv virtual environment
source /home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate

python crop_face.py --index "$i" \
                          --batch_size 50 \
                          --files_list /path/to/files/list/file/files_list.list \
                          --problem_file_path /path/to/problem/file/directory/problem.txt \
                          --pose_path /path/to/pose/file/directory/ \
                          --face_path /path/to/face/file/directory/ \
                          --time_limit 257890

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


