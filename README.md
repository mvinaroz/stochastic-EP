# DP-SEP
code for differentially private stochastic expectation propagation (DP-SEP)

### Dependencies
Versions numbers are based on our system and may not need to be exact matches. 

    python 3.8
    pystan 3.2
    scikit-learn 0.24.2
    scipy 1.6.2
    seaborn 0.11.2
    theano 1.0.5
    autodp  0.2
    matplotlib 3.4.3
    numpy 1.21.2
    pandas 1.3.3

## Repository Structure

### MoG code

Contains the code for computing the DP-SEP experiments for MoG clustering.

- For plotting Figure 1 run the following command: `python plot_cluster.py --is-private --num-data 1000 --num-iter 100 --c 1 --gamma 1 --delta 1e-5`
- For obtaining Table 1 errors run:
    - SEP: `python demo_dpsep.py --num-data 1000 --dimension 4 --num_group 4 --num-iter 100 --gamma 1`
    - DP-SEP: `python demo_dpsep.py --num-data 1000 --dimension 4 --num_group 4 --num-iter 100 --gamma 1 --clip --is-private --epsilon 1 --delta 1e-5` 
      for different epsilon values.

 
### pbp_code

Contains pbp experiments for regression datasets using EP, SEP, clipped SEP and DP-SEP. 
- To obtain Table 2 and Table 3 different seed errors with:
    - EP: `python pbp_regression_ep.py --seed 1 --data-name naval --n-hidden 50 --epochs 40`
    - SEP: `python pbp_regression_sep.py --seed 1 --data-name naval --n-hidden 50 --epochs 40`
    - clipping SEP: `python pbp_regression_dpsep.py --seed 1 --data-name naval --n-hidden 50 --epochs 40 --clip --c 1`
    - DP-SEP: `python pbp_regression_dpsep.py --seed 1 --data-name naval --n-hidden 50 --epochs 40 --clip --c 1 --is-private --epsilon 1 --delta 1e-5`
