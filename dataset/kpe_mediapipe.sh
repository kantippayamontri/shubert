#!/bin/bash

for i in {0..255}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=kpe_$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH -G1
#SBATCH --cpus-per-task=1
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out

# loading uv virtual environment
source /home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate

# remember to download hand kpe model : wget https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
# remember to download face kpe model : wget https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models

python kpe_mediapipe.py --index "$i" \
                          --batch_size 1 \
                          --pose_path /path/to/pose/file/folder/ \
                          --stats_path /path/to/stats/file/folder/ \
                          --time_limit 257890 \
                          --files_list /path/to/files/list/file/files_list.list \
                          --problem_file_path /path/to/problem/file/directory/problem.txt \
                          --face_model_path /path/to/face/model/file/mediapipe_face_model.task \
                          --hand_model_path /path/to/hand/model/file/mediapipe_hand_model.task

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


