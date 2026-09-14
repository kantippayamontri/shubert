#!/bin/bash

for i in {0..255}; do
    cat >tempGr_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=shubertinference$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH -G1
#SBATCH --constraint=
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out


# loading uv virtual environment
source /home/kan/Research/SHuBERT/.venv-shubert/bin/activate



fairseq_root=/home/kan/Research/SHuBERT/fairseq



# Ensure fairseq_root is in PYTHONPATH
export PYTHONPATH=\$PYTHONPATH:\$fairseq_root

python shubert_inference.py \\
    --index "$i" \\
    --csv_path /path/to/csv/file/csv_file.csv \\
    --checkpoint_path /path/to/checkpoint/file/checkpoint_file.pt \\
    --output_dir /path/to/output/folder/ \\
    --batch_size 1000

EOL

    # Submit the temporary slurm script
    sbatch tempGr_slurm_script_$i.sh

    # Remove the temporary slurm script
    rm tempGr_slurm_script_$i.sh
done
