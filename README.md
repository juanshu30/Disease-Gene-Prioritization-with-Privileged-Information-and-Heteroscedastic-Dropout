# Disease-Gene-Prioritization-with-Privileged-Information-and-Heteroscedastic-Dropout

About
====

This directory contains the code and resources of the following paper:

"Disease Gene Prediction with Privileged Information and Heteroscedastic Dropout". Under review.

- LUPI_RGCN is a relational GNN-based classification algorithm for graphs. It takes disease gene network as inputs and predict the possible associations.  
- Our experimental results are based on gene disease network data developed in [1]. The data can be obtained here [data](https://drive.google.com/drive/folders/1y5ZSxHq6psjfVE2OreyjJQ7xsZlIq4kL?usp=sharing)

Overview of the Model
====
We introduce LUPI_RGCN algorithm to address the gene disease prioritization problem. To achieve this goal, we develop a Heteroscedastic Gaussian Dropout algorithm, where the dropout probability of the GNN model is determined by another GNN model with a mirrored GNN architecture. The model is trained under a VAE framework with reparameterization trick. 

<img width="400" height="600" src="https://github.com/juanshu30/Disease-Gene-Prioritization-with-Privileged-Information-and-Heteroscedastic-Dropout/blob/main/figures/model.png"/>

Sub-directories
====
-[src] contains impelmentation of the rationale model used for the beer review data. main.m is the main function to run NBS2.0.
-[data] contains the pre-processed TCGA data which can be used to reproduce our results.
-[output] contains the KM-plot and clustering assignments.
