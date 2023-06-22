#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Setting up default!"
  checkpoint_root_path="/mnt/shared_storage/hephaestus_checkpoints"
  exit 1
fi
checkpoint_root_path=$1
checkpoints=($(ls $checkpoint_root_path))
checkpoint_pattern="checkpoint_"
for ((index=0;index<${#checkpoints[@]}; index++))
do
    file_path="$checkpoint_root_path"/"${checkpoints[index]}"
    files=($(ls $file_path))

    filtered_files=$(printf '%s\n' "${files[@]}"| egrep "${checkpoint_pattern}" | sort)
    last_checkpoint=$(echo "$filtered_files" | tail -n 1)
    echo "Initializing experiment from checkpoint folder: " $file_path
    config=$file_path/"config.json"
    echo "SSL Encoder: " $last_checkpoint " and configuration file: " $config

    last_checkpoint_full_path=$file_path/$last_checkpoint
    for img_size in 224 448 560
    do
       python main.py --supervised=True --ssl_encoder_path=$last_checkpoint_full_path --ssl_config_path=$config --supervised_img_size=$img_size
    done
done