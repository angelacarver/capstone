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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import jaccard_score

"""
This python file contains methods for computing GRN network similarity metrics.
"""

def degree_distribution(network1, network2, network3, network1_name: str, network2_name: str, network3_name:str):
    """
    Calculates and plots degree distribution
    Compares distributions using ANOVA

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
    i = 0
    clustering_coefficients = []
    for network in [network1, network2, network3]:
        clustering_coeffs = nx.clustering(network)
        average_clustering = sum(clustering_coeffs.values()) / len(clustering_coeffs)
        #clustering_coefficients[i] = clustering_coeffs
        print(f"Average Clustering Coefficient for Network {i}: {average_clustering}")
        i += 1

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
    i = 1
    for network in [network1, network2, network3]:
        deg_centrality = nx.degree_centrality(network)
        print(f"Top 5 Nodes by Degree Centrality for network {i}: {sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:5]}")
        betweenness_centrality = nx.betweenness_centrality(network)
        print(f"Top 5 Nodes by Betweenness Centrality for network {i}: {sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]}")
        i += 1
    #compare centrality distributions
    betweenness1 = list(nx.betweenness_centrality(network1).values())
    betweenness2 = list(nx.betweenness_centrality(network2).values())

    f_stat, p_value = f_oneway(clustering_coefficients[0], clustering_coefficients[1], clustering_coefficients[2])
    print(f"ANOVA: F-statistic={f_stat}, p-value={p_value}")


#The following section is designed to compare embeddings between node2vec and the VGAE.



def procrustes_disparity(embeddings1, embeddings2):
    """
    Measures embedding space alignment
    """
    keys1 = embeddings1.keys()

    
    matrix1 = np.array([np.array(embeddings1[key]) for key in keys1])
    print(matrix1.shape)
    m1, m2, disparity = procrustes(matrix1, embeddings2)
    print(f"Procrustes Disparity: {disparity}")
    return m1, m2, disparity

def embedding_cosine_similarity(embeddings1, embeddings2):
    keys1 = embeddings1.keys()
    
    matrix1 = np.array([embeddings1[key] for key in keys1])

    cos_sim = np.mean([1 - cosine(e1, e2) for e1, e2 in zip(matrix1, embeddings2)])
    print(f"Average cosine similarity of embeddings: {cos_sim}")
    return cos_sim

def plot_embeddings(embedding1, embedding2):
    # Ensure the embeddings are sorted by node keys
    keys1 = sorted(embedding1.keys())
    keys2 = sorted(embedding2.keys())
    
    # Extract embedding values
    matrix1 = np.array([embedding1[key] for key in keys1])
    matrix2 = np.array([embedding2[key] for key in keys2])
    
    # Combine embeddings
    combined_embeddings = np.vstack([matrix1, matrix2])
    labels = ['Treatment1'] * len(matrix1) + ['Treatment2'] * len(matrix2)

    # Reduce dimensionality
    pca = PCA(n_components=2).fit_transform(combined_embeddings)

    # Plot
    plt.scatter(pca[:len(embedding1), 0], pca[:len(embedding1), 1], label='Treatment1', alpha=0.6)
    plt.scatter(pca[len(embedding1):, 0], pca[len(embedding1):, 1], label='Treatment2', alpha=0.6)
    plt.legend()
    plt.show()

def jaccard_similarity(embedding1, embedding2):

    # Ensure the embeddings are sorted by node keys
    keys1 = embedding1.keys()
    #keys2 = sorted(embedding2.keys())

    #if keys1 != keys2:
     #   raise ValueError("The embeddings do not have the same nodes.")
    
    # Extract embedding values
    matrix1 = np.array([embedding1[key] for key in keys1])
    #matrix2 = np.array([embedding2[key] for key in keys2])

    # Find nearest neighbors
    nn1 = NearestNeighbors(n_neighbors=5).fit(matrix1)
    nn2 = NearestNeighbors(n_neighbors=5).fit(embedding2)

    neighbors1 = nn1.kneighbors(matrix1, return_distance=False)
    neighbors2 = nn2.kneighbors(embedding2, return_distance=False)

    # Jaccard similarity for node 0
    node_index = 0
    jaccard = jaccard_score(neighbors1[node_index], neighbors2[node_index], average='micro')
    print(f"Jaccard Similarity for Node {node_index}: {jaccard}")

"""
Visualizations
"""

def plot_network(network, title):
    #degree_centrality = nx.degree_centrality(network)

    # Sort nodes by degree centrality and pick the top
    #top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:20]

    # Subgraph with the top nodes
    #subgraph = network.subgraph(top_nodes)
    plt.figure(figsize=(20, 20))
    nx.draw(network, node_size=9, alpha=0.5, edge_color= 'gray', node_color='blue', with_labels=True)
    plt.title(title)
    name = title + '.png'
    plt.savefig(name)

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

def scale_network(network, scaler):
    """
    Scaler that is already fitted to first network used to transform network

    Returns:
        scaled network
    """
    edge_weights = [data['weight'] for _, _, data in network.edges(data=True)]

    # For edge weights
    scaled_edge_weights = scaler.fit_transform([[w] for w in edge_weights])
    for i, (u, v, data) in enumerate(corr_control.edges(data=True)):
        data['weight'] = scaled_edge_weights[i][0]
    
    return network

def jaccard_similarity_on_networkx(graph1, graph2):
    common_nodes = set(graph1.nodes()).intersection(set(graph2.nodes()))
    
    jaccard_similarities = {}
    for node in common_nodes:
        # Get the neighbors for the node in both graphs
        neighbors1 = set(graph1.neighbors(node))
        neighbors2 = set(graph2.neighbors(node))
        
        # Compute Jaccard similarity
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        jaccard_similarities[node] = intersection / float(union)
    
    print(f"Average Jaccard similarity: {np.mean(jaccard_similarities.values())}")
    return jaccard_similarities



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
    #node2vec_control_network = node2vec_to_networkx(node2vec_control)
    #node2vec_treat1_network = node2vec_to_networkx(node2vec_treat1)
    #node2vec_treat2_network = node2vec_to_networkx(node2vec_treat2)

    #with open ('node2vec_control_networkx.pickle', 'wb') as handle:
     #   pickle.dump(node2vec_control_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('node2vec_treat1_networkx.pickle', 'wb') as handle:
     #   pickle.dump(node2vec_treat1_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('node2vec_treat2_networkx.pickle', 'wb') as handle:
    #    pickle.dump(node2vec_treat2_network, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #vgae_control_network = embeddings_to_networkx(vgae_control_embeddings)
    #vgae_treat1_network = embeddings_to_networkx(vgae_treat1_embeddings)
    #vgae_treat2_network = embeddings_to_networkx(vgae_treat2_embeddings)

   # with open ('vgae_control_networkx.pickle', 'wb') as handle:
    #    pickle.dump(vgae_control_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('vgae_treat1_networkx.pickle', 'wb') as handle:
     #   pickle.dump(vgae_treat1_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('vgae_treat2_networkx.pickle', 'wb') as handle:
     #   pickle.dump(vgae_treat2_network, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #open networkx objects
    with open ('node2vec_control_networkx.pickle', 'rb') as handle:
        node2vec_control_network = pickle.load(handle)
    with open('node2vec_treat1_networkx.pickle', 'rb') as handle:
        node2vec_treat1_network = pickle.load(handle)
    with open('node2vec_treat2_networkx.pickle', 'rb') as handle:
        node2vec_treat2_network = pickle.load(handle)
    
    with open ('vgae_control_networkx.pickle', 'rb') as handle:
        vgae_control_network = pickle.load(handle)
    with open('vgae_treat1_networkx.pickle', 'rb') as handle:
        vgae_treat1_network = pickle.load(handle)
    with open('vgae_treat2_networkx.pickle', 'rb') as handle:
        vgae_treat2_network = pickle.load(handle)
    
    #compare networks with same treatment
    print('degree distribution')
    print('control')
    f_stat, p_value = degree_distribution(corr_control, node2vec_control_network, vgae_control_network, "Control Pearson correlation", "Control node2vec", "Control VGAE")
    print('treatment 1')
    f_stat, p_value = degree_distribution(corr_treat1, node2vec_treat1_network, vgae_treat1_network, "Treatment 1 Pearson correlation", "Treatment 1 node2vec", "Treatment 1 VGAE")
    print('treatment 2')
    f_stat, p_value = degree_distribution(corr_treat2, node2vec_treat2_network, vgae_treat2_network, "Treatment 2 Pearson correlation", "Treatment 2 node2vec", "Treatment 2 VGAE")

    print('clustering coefficient')
    print('control')
    clustering_coefficients(corr_control, node2vec_control_network, vgae_control_network)
    print('treatment 1')
    clustering_coefficients(corr_treat1, node2vec_treat1_network, vgae_treat1_network)
    print('treatment 2')
    clustering_coefficients(corr_treat2, node2vec_treat2_network, vgae_treat2_network)

    print('modularity scores')
    print('control')
    modularity_scores = modularity_score(corr_control, node2vec_control_network, vgae_control_network)
    print('treatment 1')
    modularity_scores = modularity_score(corr_treat1, node2vec_treat1_network, vgae_treat1_network)
    print('treatment 2')
    modularity_scores = modularity_score(corr_treat2, node2vec_treat2_network, vgae_treat2_network)

    print("Node Centrality Between Model Types")
    print('control')
    node_centrality(corr_control, node2vec_control_network, vgae_control_network, "Control Pearson Correlation", "Control Node2Vec", "Control VGAE")
    print('treatment 1')
    node_centrality(corr_treat1, node2vec_treat1_network, vgae_treat1_network, "Treatment 1 Pearson Correlation", "Treatment 1 Node2Vec", "Treatment 1 VGAE")
    print('treatment 2')
    node_centrality(corr_treat2, node2vec_treat2_network, vgae_treat2_network, "Treatment 2 Pearson Correlation", "Treatment 2 Node2Vec", "Treatment 2 VGAE")

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
