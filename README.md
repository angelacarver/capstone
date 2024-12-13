# Team 1 COMP 7980 Capstone Fall 2024
This repository contains the code for the Team 1 Capstone Project, "Comparing Gene Regulatory Network Construction Methods for hiPSC 4R Taupathy Model scRNA-seq Data."

The Jupyter Notebook, `updated_pipeline.ipynb` contains the exploratory data analysis in scanpy, data preprocessing, Cell Marker annotation in scanpy, Pearson correlation coefficient gene regulatory network creation, and node2vec gene regulatory network creation. The python file `grn_n2v.py` defines supporting methods that are used in the Jupyter notebook.

Due to the memory constraints of Jupyter Notebooks, the variational graph autoencoder gene regulatory network was constructed using `memory_control_vgae_script.py` rather than a Jupyter notebook.

Finally, the statistical analysis of the networks is performed using the `network_metrics.py` script.

To run these scripts, you will need the raw data. To get this data as well as the saved network embeddings, please contact the authors.
