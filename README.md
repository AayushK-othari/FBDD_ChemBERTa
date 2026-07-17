# FBDD_ChemBERTa
Using transfer learning to help in Fragment Based Drug Discovery.


- This project builds on bert-loves-chemistry by Seyone Chithrananda et al.
Original license: MIT

Data generation instructions:

Dependencies:
- Python 3.11
- RDKit, Pandas
- UniDock 1.1.3
- OBabel 3.1.1


Example generation:

python3.11 split_datasets.py --target_file ./mTORcanonical.csv --fragmentation_mode recap --protein mtor
chmod +x gendata_run_mtor_recap.sh
./gendata_run_mtor_recap.sh
python3.11 combine_dataset_chunks.py --fragmentation_mode recap --protein mtor --clean_fragments true


This repository contains code and data for our research on applying transformer models to chemical SMILES data. It builds upon the excellent work from [bert-loves-chemistry](https://github.com/seyonechithrananda/bert-loves-chemistry) by Seyone Chithrananda et al.

We adapted components from the ChemBERTa implementation to explore masked language modeling and property prediction tasks in cheminformatics. The original ChemBERTa models were trained on datasets like ZINC and PubChem using RoBERTa-style architectures.


All DeepBERTa models and fragment prediction models are available in HuggingFace under aakothari

If you use DeepBERTa or any models derived from it in your research, please cite:

```
@article{kothari2026transformer,
  title={Transformer-Based Molecular Fragment Prediction Using SMILES and DeepSMILES Representations in Fragment-Based Drug Discovery},
  author={Kothari, Aayush and Gupta, Amish and Shah, Nisarg and Reed, Thomas and Nattuva, Anvita and Nazirudeen, Rahima and Brah, Harman and Akl, Marx},
  year={2026}
}
```

Paper: https://www.researchsquare.com/article/rs-9614513/v1
