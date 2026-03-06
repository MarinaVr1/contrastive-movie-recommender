# Contrastive Learning for Implicit Recommender Systems

This project implements a recommender system based on **contrastive learning** techniques.

The model learns **latent embeddings of users and movies** from implicit interactions using the **MovieLens 100k dataset**.  
The goal of the project is to compare different learning objectives for recommendation tasks.

The implementation includes approaches such as:

- Bayesian Personalized Ranking (BPR)
- InfoNCE contrastive loss
- different negative sampling strategies for training

For a detailed description of the methodology, experiments, and results see the project documentation.

📄 **Project specification in Serbian:**  
[Specification](docs/project_specification.pdf)

---
## Implementation Details

All models and training procedures in this project are implemented from scratch using basic scientific Python libraries.

No high-level machine learning frameworks (such as PyTorch or TensorFlow) were used for building the recommendation models.

The implementation includes:

- manual matrix factorization model
- custom loss functions (BPR, InfoNCE variants)
- negative sampling strategies
- training and evaluation pipelines

---
## Dataset

Experiments are conducted on the **MovieLens 100k dataset**, which contains:

- 100,000 ratings
- 943 users
- 1682 movies

Ratings are converted into **implicit interactions** representing positive user preferences.

Dataset source:  
https://grouplens.org/datasets/movielens/

---

## Project Structure
```text
project/
│
├── data/
│
├── src/
│   ├── data_utils.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   └── evaluation.py        # Core implementation
│
├── experiments/
│   ├── runs/                # Training runs
│   └── analysis/            # Hyperparameter experiments and behaviour analysis
│
├── results/
│
├── predictions.ipynb        # Example recommendations
│
└── README.md
```
---

## Running the Project
Install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tqdm jupyter
```

Run a training experiment:

```bash
python3 -m experiments.runs.run_infonce
```

Other experiments can be executed in the same way, for example:

```bash
python3 -m experiments.runs.run_bpr
python3 -m experiments.runs.run_symmetric
```
For analysis experiments:
```bash
python3 -m experiments.analysis.epoch_stability
```

Model evaluation and analysis can also be performed using the `predictions.ipynb` notebook.
Specify the desired movies in the `recommend_from_multiple_movies()` function and run all cells to generate the top-K recommendations.

## Example Recommendation

To illustrate the model behavior, consider a user who likes the following movies:

- Star Wars (1977)
- The Godfather (1972)
- The Terminator (1984)

The system generates the following recommendations:

- Return of the Jedi (1983)
- Raiders of the Lost Ark (1981)
- The Empire Strikes Back (1980)
- Terminator 2: Judgment Day (1991)
- Indiana Jones and the Last Crusade (1989)
- Alien (1979)
- The Silence of the Lambs (1991)

This example shows how the model identifies thematically similar movies in the learned latent space.

---

## Author

Marina Vračarić  
Faculty of Mathematics  
University of Belgrade  

Course: **Computational Intelligence**


