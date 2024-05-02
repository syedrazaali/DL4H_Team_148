# Improving Medical Predictions with LSTM Models

## Description
This repository contains the implementation of an LSTM-based model for predicting medical outcomes using data derived from the MIMIC-III database. The project focuses on handling and integrating irregular time series data and clinical notes for predictions in intensive care units.

## Installation
### Prerequisites:

- Python 3.8+
- PostgreSQL (latest version)
- Access to the MIMIC-III dataset
- Google Colab for running the notebook
### Setup

1. **Clone the repository:**\
```"git clone https://github.com/syedrazaali/DL4H_Team_148/"```
2. **Install required Python packages:**
```"pip install -r requirements.txt"```
## Data Preparation
### PostgreSQL Database Setup

1. **Download the MIMIC-III dataset**, specifically the CHARTEVENTS.csv, LABEVENTS.csv, and NOTEEVENTS.csv files.
2. **Set up your PostgreSQL database** and create the necessary tables by executing the SQL commands provided in the sql_setup.sql file in this repository.
3. **Import the CSV files into your PostgreSQL database:**
```
"COPY chartevents FROM 'path/to/CHARTEVENTS.csv' DELIMITER ',' CSV HEADER;
COPY labevents FROM 'path/to/LABEVENTS.csv' DELIMITER ',' CSV HEADER;
COPY noteevents FROM 'path/to/NOTEEVENTS.csv' DELIMITER ',' CSV HEADER;"
```
4. **Execute preprocessing SQL queries** to filter and aggregate necessary features. Ensure you create relevant output files that can be downloaded.
### Using Google Colab
After preprocessing the data and creating necessary outputs:

- **Upload the resulting files to your Google Drive.**
- **Open the Google Colab notebook provided in this repository (DL4H_Final_Team148.ipynb).**
- **Connect your Google Drive to the Colab session using:**
``` 
from google.colab import drive
drive.mount('/content/drive')
```
- **Run the notebook** to perform further data processing, model training, and evaluation.

## License
Distributed under the MIT License. See LICENSE for more information.

## Citation and References
If you use this model or the methodology in your research, please cite the original paper and this repository:
```
"@article{zhang2023improving,
title={Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling},
author={Zhang, X. and Li, S. and Chen, Z. and Yan, X. and Petzold, L.},
journal={Proceedings of the 40th International Conference on Machine Learning},
year={2023},
volume={202},
doi={https://doi.org/10.48550/arXiv.2210.12156}
}"
```

[1]: Zhang, X., Li, S., Chen, Z., Yan, X., Petzold, L., Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling, Proceedings of the 40th International Conference on Machine Learning, Honolulu, Hawaii, USA, PMLR 202, 2023, doi: [https://doi.org/10.48550/arXiv.2210.12156]
