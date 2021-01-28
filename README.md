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
- [figures] contains the figures that are used in the paper.
- [LUPI_RGCN] contains implementation of our model for the disease gene network.

Data
====
There are three datasets in the link. 

### genes_phenes.mat

GeneGene_Hs: The HumanNet gene interaction network of size 12331 x 12331.
GenePhene: a cell array containing Gene-Phenotype networks of 9 species.
GP_SPECIES: The names of the species corresponding to the networks in 'GenePhene' variable.
geneIds: The entrezdb ids of genes, corresponding to the rows of the matrix 'GeneGene_Hs' (or 'GenePhene' matrices).
pheneIds: a cell array containing OMIM ids for phenotypes of 9 species.
PhenotypeSimilaritiesLog: Similarity network between OMIM diseases.

Code Usage
====


License
====

