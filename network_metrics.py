import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from networkx.algorithms.community import greedy_modularity_communities
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean
import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score
from sklearn.neighbors import NearestNeighbors

"""
This python file contains methods for computing GRN network similarity metrics.
"""

def degree_distribution(network1, network2, network3, network1_name: str, network2_name: str, network3_name:str):
    """
    Calculates and plots degree distribution
    Compares distributions using Kolmogorov-Smirnov test

    Args:
        networkx objects and their names (str)
    Returns:
        ANOVA f_stat and p value
    """
    degrees_network1 = [d for n, d in nx.degree(network1)]
    plt.hist(degrees_network1, bins=20, alpha=0.7, label=f"{network1_name} Degree Distribution")
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()

    degrees_network2 = [d for n, d in nx.degree(network2)]
    plt.hist(degrees_network2, bins=20, alpha=0.7, label=f"{network2_name} Degree Distribution")
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()

    degrees_network3 = [d for n, d in nx.degree(network2)]
    plt.hist(degrees_network3, bins=20, alpha=0.7, label=f"{network3_name} Degree Distribution")
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()

    f_stat, p_value = f_oneway(degrees_network1, degrees_network2, degrees_network3)
    print(f"ANOVA: F-statistic={f_stat}, p-value={p_value}")
    return f_stat, p_value


def clustering_coefficients(network1, network2, network3):
    """
    Calculates and compares the clustering coefficients for 3 networks

    Args:
        three networkx objects representing the networks to be compared
    Returns:
        clustering coefficients
        f statistic and p value
    """
    i = 1
    clustering_coefficients = []
    for network in [network1, network2, network3]:
        clustering_coeffs = nx.clustering(network)
        average_clustering = sum(clustering_coeffs.values()) / len(clustering_coeffs)
        clustering_coefficients.append(average_clustering)
        print(f"Average Clustering Coefficient for Network {i}: {average_clustering}")

    f_stat, p_value = f_oneway(clustering_coefficients[0], clustering_coefficients[1], clustering_coefficients[2])
    print(f"ANOVA: F-statistic={f_stat}, p-value={p_value}")
    return f_stat, p_value

def modularity_score(network1, network2, network3):
    """
    Computes and returns the modularity scores from 3 networks
    """
    modularity_scores = []
    for network in [network1, network2, network3]:
        # Detect communities and compute modularity
        communities = list(greedy_modularity_communities(network))
        modularity_score = nx.algorithms.community.modularity(network, communities)
        print(f"Modularity Score: {modularity_score}")
        modularity_scores.append(modularity_score)
    return modularity_scores

def node_centrality(network1, network2, network3, network1_name, network2_name, network3_name):
    """
    Calculates and returns the top 5 nodes from each
    network by degree centrality
    """
    for network in [network1, network2, network3]:
        degree_centrality = nx.degree_centrality
        print(f"Top 5 Nodes by Degree Centrality: {sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]}")
        betweenness_centrality = nx.betweenness_centrality(network)
        print(f"Top 5 Nodes by Betweenness Centrality: {sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    #compare centrality distributions
    betweenness1 = list(nx.betweenness_centrality(network1).values())
    betweenness2 = list(nx.betweenness_centrality(network2).values())

    f_stat, p_value = f_oneway(clustering_coefficients[0], clustering_coefficients[1], clustering_coefficients[2])
    print(f"ANOVA: F-statistic={f_stat}, p-value={p_value}")

"""
The following section is designed to compare embeddings between node2vec and the VGAE.
"""


def procrustes_disparity(embeddings1, embeddings2):
    """
    Measures embedding space alignment
    """
    m1, m2, disparity = procrustes(embeddings1, embeddings2)
    print(f"Procrustes Disparity: {disparity}")
    return m1, m2, disparity

def embedding_cosine_similarity(embeddings1, embeddings2):
    cos_sim = np.mean([1 - cosine(e1, e2) for e1, e2 in zip(embeddings1, embeddings2)])
    print(f"Average cosine similarity of embeddings: {cos_sim}")
    return cos_sim

def plot_embeddings(embedding1, embedding2):
    # Combine embeddings
    combined_embeddings = np.vstack([embedding1, embedding2])
    labels = ['Treatment1'] * len(embedding1) + ['Treatment2'] * len(embedding2)

    # Reduce dimensionality
    pca = PCA(n_components=2).fit_transform(combined_embeddings)

    # Plot
    plt.scatter(pca[:len(embedding1), 0], pca[:len(embedding1), 1], label='Treatment1', alpha=0.6)
    plt.scatter(pca[len(embedding1):, 0], pca[len(embedding1):, 1], label='Treatment2', alpha=0.6)
    plt.legend()
    plt.show()

def jaccard_similarity(embedding1, embedding2):
    # Find nearest neighbors
    nn1 = NearestNeighbors(n_neighbors=5).fit(embedding1)
    nn2 = NearestNeighbors(n_neighbors=5).fit(embedding2)

    neighbors1 = nn1.kneighbors(embedding1, return_distance=False)
    neighbors2 = nn2.kneighbors(embedding1, return_distance=False)

    # Jaccard similarity for node 0
    node_index = 0
    jaccard = jaccard_score(neighbors1[node_index], neighbors2[node_index], average='micro')
    print(f"Jaccard Similarity for Node {node_index}: {jaccard}")

"""
Visualizations
"""

def plot_network(network, title):
    plt.figure(figsize=(8,6))
    nx.draw(network, node_size=50, node_color='blue', with_labels=True)
    plt.title(title)
    plt.show()




if __name__ == '__main__':

    """
    Comparing models
    """    
    #read in data
    

    #convert to networkx objects


    #run metrics & visualizations
    degree_distribution(network1, network2, network3, network1_name: str, network2_name: str, network3_name:str)


    """
    Comparing Treatments
    """
