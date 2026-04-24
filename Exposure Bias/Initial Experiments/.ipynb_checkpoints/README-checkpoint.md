# Movie Recommendation System using Neural Collaborative Filtering (NCF)

## Overview
This project implements a **Movie Recommendation System** using **Neural Collaborative Filtering (NCF)** on the **MovieLens dataset**.  
The model captures **user–item interactions** using embeddings and a **Multi-Layer Perceptron (MLP)**, outperforming baseline collaborative filtering models.

## Features
- Preprocesses MovieLens ratings data  
- Encodes users and movies into embedding vectors  
- NCF model with:
  - Embedding layers for users and movies
  - Dense layers (256 → 128 → 64) with Dropout & BatchNormalization
- EarlyStopping and ModelCheckpoint callbacks for stable training  
- Evaluation metrics: **MSE, MAE, RMSE**  
- Generates **Top-N movie recommendations** for any user  

## Repository Structure
Movie-Recommendation-NCF/
│── data/ # dataset instructions
│── notebooks/ # Jupyter notebooks
│── src/ # modular Python code
│── results/ # training results and plots
│── requirements.txt # dependencies
│── README.md # project documentation

## Dataset
We use the **MovieLens dataset** given in data folder.  

## Installation
Clone the repository and install dependencies:

```
git clone https://github.com/gopal092003/Movie-Recommendation-NCF.git
cd Movie-Recommendation-NCF
pip install -r requirements.txt
```

## Results
Evaluation metrics are saved in results/metrics.txt.
Training loss plot saved as results/loss_plot.png.

## Future Improvements
Experiment with NeuMF (GMF + MLP hybrid) for better accuracy
Incorporate attention mechanisms in the recommender
Build an API for serving recommendations