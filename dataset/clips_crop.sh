#!/bin/bash

for i in {0..255}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=crop_$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH -G1
#SBATCH --cpus-per-task=1
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out

# loading uv virtual environment
source /home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate

python clips_bbox.py --index "$i" \
    --batch_size 50 \
    --files_list /path_to_files_list_file/files_list.list \
    --output_clips_directory /path_to_output_clips_directory \
    --problem_file_path /path_to_problem_file_directory/problem.txt \
    --yolo_model_path /path_to_yolo_model_file/yolov8n.pt

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


