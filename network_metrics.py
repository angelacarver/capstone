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
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

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

def embeddings_to_networkx(network_embeddings):
    reconstructed_adj = network_embeddings @ network_embeddings.T  # Dot product to get similarity scores

    # Threshold to create edges (e.g., values > 0.5 are considered connected)
    threshold = 0.5
    reconstructed_adj_binary = (reconstructed_adj > threshold).astype(int)

    # Convert adjacency matrix to a NetworkX graph
    G = nx.from_numpy_array(reconstructed_adj_binary)
    return G

def corr_to_networkx(corr_matrix, threshold=0.5):
    adj_matrix = np.where(np.abs(corr_matrix) > threshold, 1, 0)
    np.fill_diagonal(adj_matrix, 0)
    G = nx.from_numpy_array(adj_matrix)
    #assign correlation values as edge weights
    for i in tqdm(range(corr_matrix.shape[0])):
        for j in range(i+1, corr_matrix.shape[1]):
            if np.abs(corr_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=corr_matrix[i, j])
    
    return G

def node2vec_to_networkx(network: dict, threshold=0.5):
    # Convert the embeddings dictionary to a matrix
    nodes = list(network.keys())
    vectors = np.array(list(network.values()))

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(vectors)

    # Create a graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(nodes)

    # Add edges based on similarity threshold
    for i, node1 in tqdm(enumerate(nodes)):
        for j, node2 in enumerate(nodes):
            if i < j and similarity_matrix[i, j] > threshold:
                G.add_edge(node1, node2, weight=similarity_matrix[i, j])

    return G


if __name__ == '__main__':

    """
    Comparing models
    """    
    #read in data
    #the Pearson Correlation coefficient networks are already networkx objects
    with open('corr_control_graph.pickle', 'rb') as f:
        corr_control = pickle.load(f)
    with open('corr_group2_graph.pickle', 'rb') as f:
        corr_treat1 = pickle.load(f)
    with open('corr_group3_graph.pickle', 'rb') as f:
        corr_treat2 = pickle.load(f)

    with open('node2vec_control_embeddings.pickle', 'rb') as f:
        node2vec_control = pickle.load(f)
    with open('node2vec_group2_embeddings.pickle','rb') as f:
        node2vec_treat1 = pickle.load(f)
    with open('node2vec_group3_embeddings.pickle', 'rb') as f:
        node2vec_treat2 = pickle.load(f)

    vgae_control_embeddings = np.load("vgae_control_node_embeddings.npy", allow_pickle=True)
    vgae_treat1_embeddings = np.load("vgae_group2_node_embeddings.npy", allow_pickle=True)
    vgae_treat2_embeddings = np.load("vgae_group3_node_embeddings.npy", allow_pickle=True)

    #convert to networkx objects
    node2vec_control_network = node2vec_to_networkx(node2vec_control)
    node2vec_treat1_network = node2vec_to_networkx(node2vec_treat1)
    node2vec_treat2_network = node2vec_to_networkx(node2vec_treat2)

    vgae_control_network = embeddings_to_networkx(vgae_control_embeddings)
    vgae_treat1_network = embeddings_to_networkx(vgae_treat1_embeddings)
    vgae_treat2_network = embeddings_to_networkx(vgae_treat2_embeddings)


    #compare networks with same treatment
    f_stat, p_value = degree_distribution(corr_control, node2vec_control_network, vgae_control_network, "Control Pearson correlation", "Control node2vec", "Control VGAE")
    f_stat, p_value = degree_distribution(corr_treat1, node2vec_treat1_network, vgae_treat1_network, "Treatment 1 Pearson correlation", "Treatment 1 node2vec", "Treatment 1 VGAE")
    f_stat, p_value = degree_distribution(corr_treat2, node2vec_treat2_network, vgae_treat2_network, "Treatment 2 Pearson correlation", "Treatment 2 node2vec", "Treatment 2 VGAE")

    f_stat, p_value = clustering_coefficients(corr_control, node2vec_control_network, vgae_control_network)
    f_stat, p_value = clustering_coefficients(corr_treat1, node2vec_treat1_network, vgae_treat1_network)
    f_stat, p_value = clustering_coefficients(corr_treat2, node2vec_treat2_network, vgae_treat2_network)

    modularity_scores = modularity_score(corr_control, node2vec_control_network, vgae_control_network)
    modularity_scores = modularity_score(corr_treat1, node2vec_treat1_network, vgae_treat1_network)
    modularity_scores = modularity_score(corr_treat2, node2vec_treat2_network, vgae_treat2_network)

    print("Node Centrality Between Model Types")
    node_centrality(corr_control, node2vec_control_network, vgae_control_network, "Control Pearson Correlation", "Control Node2Vec", "Control VGAE")
    node_centrality(corr_treat1, node2vec_treat1_network, vgae_treat1_network, "Treatment 1 Pearson Correlation", "Treatment 1 Node2Vec", "Treatment 1 VGAE")
    node_centrality(corr_treat2, node2vec_treat2_network, vgae_treat2_network, "Treatment 2 Pearson Correlation", "Treatment 2 Node2Vec", "Treatment 2 VGAE")

    print("Node Centrality Between Treatments")
    node_centrality(corr_control, corr_treat1, corr_treat1, "Control Pearson Correlation", "Treatment 1 Pearson Correlation", "Treatment 2 Pearson Correlation")
    node_centrality(node2vec_control_network, node2vec_treat1_network, node2vec_treat2_network, "Control Node2Vec", "Treament 1 Node2Vec", "Treatment 2 Node2Vec")
    node_centrality(vgae_control_network, vgae_treat1_network, vgae_treat2_network, "Control VGAE", "Treament 1 VGAE", "Treatment 2 VGAE")

    print('Procrusted Disparity Between Node2Vec and VGAE Embeddings')
    print('control')
    procrustes_disparity(node2vec_control, vgae_control_embeddings)
    print('treat1')
    procrustes_disparity(node2vec_treat1, vgae_treat1_embeddings)
    print('treat2')
    procrustes_disparity(node2vec_treat2, vgae_treat2_embeddings)

    print('Cosine similarity of embeddings')
    print('control')
    embedding_cosine_similarity(node2vec_control, vgae_control_embeddings)
    print('treat1')
    embedding_cosine_similarity(node2vec_treat1, vgae_treat1_embeddings)
    print('treat2')
    embedding_cosine_similarity(node2vec_treat2, vgae_treat2_embeddings)

    print('Jaccard Similarity Between Node2Vec and VGAE')
    print('control')
    jaccard_similarity(node2vec_control, vgae_control_embeddings)
    print('treat1')
    jaccard_similarity(node2vec_treat1, vgae_treat1_embeddings)
    print('treat2')
    jaccard_similarity(node2vec_treat2, vgae_treat2_embeddings)

    print('Plotting Networks')
    plot_network(corr_control, 'Control Pearson Correlation')
    plot_network(corr_treat1, 'Treatment 1 Pearson Correlation')
    plot_network(corr_treat2, 'Treatment 2 Pearson Correlation')

    plot_network(node2vec_control_network, 'Control Node2Vec')
    plot_network(node2vec_treat1_network, 'Treatment 1 Node2Vec')
    plot_network(node2vec_treat2_network, 'Treatment 2 Node2Vec')

    plot_network(vgae_control_network, 'Control VGAE')
    plot_network(vgae_treat1_network, 'Treatment 1 VGAE')
    plot_network(vgae_treat2_network, 'Treatment 2 VGAE')




    










    """
    Comparing Treatments
    """
