import scanpy as sc
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import gseapy as gp
from scipy.spatial.distance import cosine
from tqdm import tqdm

def load_and_preprocess_data():
    #load in mtx file
    adata = sc.read_10x_mtx('C:\\Users\\a303\\Downloads\\alzheimers_data\\', var_names='gene_symbols', cache=True)

    #accessing the matrix, gene names, and barcodes
    expression_matrix = adata.X #sparse matrix (cell x gene)
    genes = adata.var_names #list of gene names
    barcodes = adata.obs_names #list of cell barcodes

    print('expression matrix shape:\n', expression_matrix.shape)
    print('first gene:\n', expression_matrix[0])

    #filter cells and genes, normalize data
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(adata) #print filtered, normalized expression matrix
    return adata

def construct_grn(adata) -> nx.Graph:
    expression_matrix = adata.X.T.toarray() #extract expression matrix (genes x cells) and transpose it (cells x genes)
    correlation_matrix = np.corrcoef(expression_matrix) #compute correlation matrix (gene-gene correlations)
    threshold = 0.5 #threshold for correlations to build edges in the network
    correlation_matrix[np.abs(correlation_matrix) < threshold] = 0

    #create graph
    genes = adata.var_names
    G = nx.Graph()
    G.add_nodes_from(genes) #add gene nodes

    #add edges (gene-gene correlations above threshold)
    for i, gene1 in enumerate(genes):
        for j, gene2 in enumerate(genes):
            if i != j and correlation_matrix[i, j] != 0:
                G.add_edge(gene1, gene2, weight=correlation_matrix[i, j])
    
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    return G

def apply_node2vec(G: nx.Graph) -> dict:
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4) #apply Node@Vec on constructed graph
    model = node2vec.fit(window=10, min_count=1, batch_words=4) #fit the model to get node embeddings
    embeddings = {node: model.wv[node] for node in G.nodes()} #get embeddings as a dictionary {gene: embedding_vector}
    print(embeddings)
    return embeddings

def visualize(embeddings: dict, G):
    #use PCA or UMAP to reduce the dimensionality of embeddings for visualization (2D)
    embedding_matrix = np.array([embeddings[gene] for gene in G.nodes()])
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(embedding_matrix)

    #create a layout using the embeddings (2D coordinates)
    pos = {gene: embedding_2d[i] for i, gene in enumerate(G.nodes())}

    #plot the graph
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=20, node_color='blue', edge_color='gray')
    plt.show()

def cluster(embeddings: dict, G):
    """
    Checks for gene clusters and co-expression
    """
    embedding_matrix = np.array([embeddings[gene] for gene in G.nodes()])
    #k-means clustering
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embedding_matrix)
    clusters = kmeans.labels_
    #visualize clusters using dimensionality reduction
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(embedding_matrix)

    plt.figure(figsize=(10,7))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=clusters, palette='viridis', legend='full')
    plt.title('Gene Clusters based on Node2Vec Embeddings')
    plt.show()

    return clusters

def gsea(G, clusters):
    genes_in_cluster = [gene for gene, label in zip(G.nodes(), clusters) if label == 0]
    gsea_results = gp.enrichr(gene_list=genes_in_cluster, gene_sets='KEGG_2016')
    #visualize GSEA results
    gsea_results.results.head(10)

def predict_regulatory_relationships(G, embeddings):
    """
    predict novel gene-gene regulatory interactions
    can predict genes that are structurally similar
    genes close to each other in the embedding space may not be directly connected in the graph
    these could represent novel regulatory interactions
    """
    missing_edges = []
    for gene1 in tqdm(G.nodes()):
        for gene2 in G.nodes():
            if gene1 != gene2 and not G.has_edge(gene1, gene2):
                similarity = 1 - cosine(embeddings[gene1], embeddings[gene2])
                if similarity > 0.9: #threshold for high similarity
                    missing_edges.append((gene1, gene2, similarity))

    #sort missing edges by similarity and inspect potential new regulatory interactions
    missing_edges_sorted = sorted(missing_edges, key=lambda x: x[2], reverse=True)
    print(missing_edges_sorted[:10])




if __name__=='__main__':
    data = load_and_preprocess_data()
    graph = construct_grn(data)
    embeddings = apply_node2vec(graph)
    visualize(embeddings, graph)
    clusters = cluster(embeddings, graph)
    gsea(graph, clusters)
    predict_regulatory_relationships(graph, embeddings)



