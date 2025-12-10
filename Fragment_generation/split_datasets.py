# Issues/crashes with large datasets and unidock mean we just split the work into multiple runs.
# Writes a shell script to the working directory. Run it to generate dataset chunks.
# Output chunks are combined using combine_dataset_chunks.py

import pandas
import argparse
import glob
import os

parser = argparse.ArgumentParser(description="Dataset splitting")
parser.add_argument('--target_file', type=str, required=True, help="Path to input CSV file with SMILES strings")
parser.add_argument("--fragmentation_mode", type=str, required=True, help="valid options are brics, recap, medchemfrag, macfrag, and fragpredict")
parser.add_argument("--protein", type=str, required=True, help="Name of target to dock against. Working direcory should contain a docking configuration file named $protein$_docking_config.json.")
parser.add_argument("--model_path", type=str, required=False, help="Path to trained FragPredict model if using FragPredict as the fragmentation mode")
args = parser.parse_args()

# target_file = "./mTORcanonical.csv"
target_file = args.target_file
chunk_size = 2000
fragmentation_method = args.fragmentation_mode
protein = args.protein

bash_script_file = f"gendata_run_{protein}_{fragmentation_method}.sh"
base_command = f"python3.11 gen_data_main.py --docking_config_path {protein}_docking_config.json --fragmentation_mode {fragmentation_method} "
if args.model_path: base_command += f"--model_path {args.model_path} "
base_dataset_file_name = "chunked_validation_dataset/canonical_dataset_chunk_"
base_output_file_name = f"gendata_{protein}_{fragmentation_method}_output_chunks/{protein}_{fragmentation_method}_final_"

for file in glob.glob(f"./gendata_{protein}_{fragmentation_method}_output_chunks/*.csv"):
    os.remove(file)

i = 0
with pandas.read_csv(target_file, chunksize=chunk_size) as reader:
    with open(bash_script_file, "w") as script_file:
        for chunk in reader:
            fname = base_dataset_file_name+f"part{i}.csv"
            chunk.to_csv(fname, header=True, mode="w")
            script_file.write(base_command + f" --input_csv {fname} --output_path {base_output_file_name}part{i}.csv \n")
            i+=1
