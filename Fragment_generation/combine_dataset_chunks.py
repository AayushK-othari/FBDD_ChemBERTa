# Combine dataset chunks created by shell command from split_datasets.py

import pandas
import glob
import argparse
from rdkit import Chem

parser = argparse.ArgumentParser(description="Dataset splitting")
parser.add_argument("--fragmentation_mode", type=str, required=True, help="valid options are brics, recap, medchemfrag, macfrag, and fragpredict")
parser.add_argument("--protein", type=str, required=True)
parser.add_argument("--clean_fragments", type=bool, required=True, help="writes an additional file with stereochemistry and implicit hydrgoens removed and fragments canonicalized")
args = parser.parse_args()
filenames = glob.glob(f"gendata_{args.protein}_{args.fragmentation_mode}_output_chunks/*.csv")
outname = f"{args.protein}_{args.fragmentation_mode}_final"

dataframe = pandas.concat((pandas.read_csv(f) for f in filenames), ignore_index=True)
dataframe.to_csv(outname+".csv")

nrows=0
avg_LE = 0
avg_SAscore = 0
avg_logS = 0
avg_BA = 0
avg_score = 0
for _, row in pandas.read_csv(outname+".csv").iterrows():
    nrows+=1
    avg_LE += row["ligand_efficiency"]
    avg_SAscore += row["sascore"]
    avg_logS += row["predicted_log_s"]
    avg_BA += row["docking_score"]
    avg_score += row["score"]

with open(f"{args.fragmentation_mode}_{args.protein}_summary.txt", "w") as f:
    f.write(f"Mean ligand efficiency : {avg_LE/nrows} \n")
    f.write(f"Mean SAscore : {avg_SAscore/nrows}\n")
    f.write(f"Mean log(S) : {avg_logS/nrows}\n")
    f.write(f"Mean binding affinity : {avg_BA/nrows}\n")
    f.write(f"Mean fragment score : {avg_score/nrows}\n")

if (args.clean_fragments):
    data = pandas.read_csv(outname+".csv")

    new_rows = []

    for index, row in data.iterrows():
        frag = Chem.MolFromSmiles(row["fragment"])
        Chem.RemoveHs(frag)
        # Chem.RemoveStereochemistry(frag)
        # Chem.RemoveHs(frag)
        canonicalized_frag = Chem.MolToSmiles(frag, isomericSmiles=False, canonical=True)
        row["fragment"] = canonicalized_frag
        new_rows.append(row)

    results_df = pandas.DataFrame(list(new_rows))
    results_df.to_csv(outname+"_canonicalized.csv", index=False)
