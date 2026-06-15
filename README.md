# FBDD_ChemBERTa
Using transfer learning to help in Fragment Based Drug Discovery.


- This project builds on bert-loves-chemistry by Seyone Chithrananda et al.
Original license: MIT


This repository contains code and data for our research on applying transformer models to chemical SMILES data. It builds upon the excellent work from [bert-loves-chemistry](https://github.com/seyonechithrananda/bert-loves-chemistry) by Seyone Chithrananda et al.

We adapted components from the ChemBERTa implementation to explore masked language modeling and property prediction tasks in cheminformatics. The original ChemBERTa models were trained on datasets like ZINC and PubChem using RoBERTa-style architectures.


All DeepBERTa models and fragment prediction models are available in HuggingFace under aakothari

If you use DeepBERTa or any models derived from it in your research, please cite:

```
@article{kothari2026deepberta,
  title={DeepBERTa-RL: Reinforcement Learning for Fragment Prediction in Fragment-Based Drug Discovery Using DeepSMILES Molecular Representations},
  author={Kothari, Aayush and others},
  journal={Research Square},
  year={2026},
  doi={10.21203/rs.3.rs-9614513/v1}
}
```

Paper: https://www.researchsquare.com/article/rs-9614513/v1
