# annotation-generation-and-expert-level-testing
# Honest and Reliable Evaluation and Expert Equivalence Testing of Automated Neonatal Seizure Detection
## Overview
This repository contains the source code for the **expert-level tests** and the **annotation generation methods (A and B)** used in:

> *Honest and Reliable Evaluation and Expert Equivalence Testing of Automated Neonatal Seizure Detection*  
> Jovana Kljajic, John M. O’Toole, Robert Hogan, Tamara Skorić  
> arXiv:2508.04899 :contentReference[oaicite:0]{index=0}

All implementation details, motivations, and results are available in the paper which is available at https://arxiv.org/abs/2508.04899.


## Repository Structure

- **methods/** – Functions for annotation generation methods A and B
  - `method_A.py`
  - `method_B.py`

- **expert-level-tests/** – Expert-level test functions
  - `IRA_vs_AI_Consensus_Agreement.py`
  - `Multi_Rater_Agreement_Statistical_Turing_Tests.py`
  - `Pairwise_Metric_Statistical_Non_Inferiority_Tests.py`

- **examples/** – Example scripts showing usage
  - `example_method_A.py`
  - `example_method_B.py`

- `utils.py` – Shared utility functions  
- `README.md` 
- `LICENSE` 
- `requirements.txt` – Python dependencies  
- `.gitignore` 
---
## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2. (Optional) Create a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
This will install the following packages with the versions used in this project:
- numpy==1.24.4
- pandas==1.5.3
- scipy==1.10.1
- scikit-learn==1.3.2
- statsmodels==0.14.1
- tqdm==4.67.1
- irrcac==0.4.0

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code, please cite the paper associated with this repository.

