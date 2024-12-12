import scanpy as sc
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import gseapy as gps
from scipy.spatial.distance import cosine
from tqdm import tqdm
import harmonypy as hm
import math
import anndata as ad
from scipy.stats import fisher_exact
from networkx.algorithms import community
import sklearn
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

def load_and_preprocess_data(filepaths: list, batch_labels: list):
    #load in mtx file
    adata_objects = []
    for filepath in filepaths:
        adata = sc.read_10x_mtx(filepath, var_names='gene_symbols', cache=True)#('C:\\Users\\a303\\Downloads\\alzheimers_data\\', var_names='gene_symbols', cache=True)
        adata_objects.append(adata)
    adata = ad.concat(adata_objects, join='outer', label='batch', keys=batch_labels, index_unique='-') 
    #accessing the matrix, gene names, and barcodes
    expression_matrix = adata.X #sparse matrix (cell x gene)
    genes = adata.var_names #list of gene names
    barcodes = adata.obs_names #list of cell barcodes

    #quality control
    #mitochondrial genes, "MT-"
    adata.var["mt"] = adata.var_names.str.startswith("MT")
    #ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    #hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True) #calculate QC metrics
    sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"], jitter=0.4, multi_panel=True,) #violin plot of QC
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

    # Combine filters: Keep only genes not in the above categories
    genes_to_keep = ~(
        adata.var['mt'] |
        adata.var['hb'] |
        adata.var['ribo']
    )

    # Filter the data
    adata = adata[:, genes_to_keep].copy()
    print('adata mt: ', adata.var['mt'])

    #filter cells and genes, normalize data
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    #plot the highest expression genes
    sc.pl.highest_expr_genes(adata, n_top=20)

    print('expression matrix shape:\n', expression_matrix.shape)
    print('first gene:\n', expression_matrix[0])

    #Doublet Detection
    sc.pp.scrublet(adata)

    #save count data
    adata.layers["counts"] = adata.X.copy()
    #normalize total counts per cell to 10,000
    sc.pp.normalize_total(adata)
    #log transformation to stabilize variance
    sc.pp.log1p(adata)

    #feature selection
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pl.highly_variable_genes(adata)

    #dimensionality reduction
    sc.tl.pca(adata)
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
    sc.pl.pca(adata, #color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"], #look into color issues
              dimensions=[(0, 1), (2,3), (0,1), (2,3)],
              ncols=2,
              size=2
              )

    #nearest neighbors
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        #color="sample",
        size=2,
    )

    #clustering
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
    sc.pl.umap(adata, color=["leiden"])

    #reassess quality control and cell filtering
    sc.pl.umap(
        adata,
        color=["leiden", "predicted_doublet", "doublet_score"],
        wspace=0.5,
        size=3
    )
    sc.pl.umap(
        adata,
        color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
        wspace=0.5,
        ncols=2,
    )

    return adata
    #TODO need to convert to expression matrix using adata.X

#TODO check for and filter out the noncoding genes

def cell_type_annotation(adata, filepath):
    #how do you choose res? Since there is only one cell type in the control, should it just be one cluster?
    for res in [0.02, 0.5, 2.0]:
        #generate clusters
        sc.tl.leiden(adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph")
    #plot it
    sc.pl.umap(adata, color=["leiden_res_0.02", "leiden_res_0.50", "leiden_res_2.00"],
    legend_loc="on data",)
    #need list of neuron-related markers to filter out housekeeping genes, noise, etc.
    neuronal_markers = pd.read_csv(filepath)
    neuronal_markers = neuronal_markers["marker"].to_list()
    print('Typical neuronal markers: ', neuronal_markers)
    available_markers = [gene for gene in neuronal_markers if gene in adata.var_names]
    print("Available neuronal markers: ", available_markers)
    #plot expression of markers on UMAP
    markers_per_plot = 4
    n_markers = len(available_markers)
    n_plots = math.ceil(n_markers/markers_per_plot)
    for i in range(n_plots):
        subset_markers = available_markers[i * markers_per_plot : (i+1) * markers_per_plot]
        titles = [f"Expression of {marker}" for marker in subset_markers]
        sc.pl.umap(adata, color=subset_markers, title=titles)

    #calculate the mean expression of each marker per cluster
    sc.tl.rank_genes_groups(adata, groupby='leiden_res_0.50', method='t-test') #number of clusters in the ranking doesn't match number of clusters in leiden .5 graph
    #annotate clusters based on known markers
    sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)

    #rename clusters manually
    cluster_labels = {
        '0': "Mature/Differentiating Neurons",
        '1': "Neuronal Progenitors",
        '2': "Mature Neurons",
        '3': "Differentiating Neurons",
        '4': "Early Differentiating Neurons",
        '5': "Stress-Responsive/Signaling Neurons",
        '6': "Transitioning/Active Neurons",
        '7': "Astrocytes/Glia"
    }
    
    adata.obs['cell_type'] = adata.obs['leiden_res_0.50'].map(cluster_labels)
    sc.pl.umap(adata, color='cell_type', legend_loc='on data')

    clusters_of_interest = ['0', '2']
    print(adata[adata.obs['batch']== 'control'].X.toarray())
    print(adata[adata.obs['leiden_res_0.50'].isin(clusters_of_interest)].X.toarray())

    control_data = adata[(adata.obs['batch'] == 'control') &
                    (adata.obs['leiden_res_0.50'].isin(clusters_of_interest))]

    group2_data = adata[(adata.obs['batch'] == 'treatment1') &
                    (adata.obs['leiden_res_0.50'].isin(clusters_of_interest))]
    
    group3_data = adata[(adata.obs['batch'] == 'treatment2') &
                    (adata.obs['leiden_res_0.50'].isin(clusters_of_interest))]

    return control_data, group2_data, group3_data
    
def construct_grn(adata, threshold=0.5) -> nx.Graph:
    # Extract sparse expression matrix (genes x cells)
    expression_matrix = adata.X.T  # Transpose to (cells x genes)
    
    # Compute correlation matrix efficiently for sparse matrices
    # Only calculate correlations for upper triangle
    correlation_matrix = np.corrcoef(expression_matrix.toarray())
    
    # Apply threshold directly on the correlation matrix
    mask = np.abs(correlation_matrix) >= threshold
    np.fill_diagonal(mask, False)  # Remove self-loops
    
    # Create graph
    genes = adata.var_names
    G = nx.Graph()
    G.add_nodes_from(genes)  # Add gene nodes
    
    # Use numpy to extract edges
    edges = np.column_stack(np.where(mask))
    weights = correlation_matrix[mask]
    
    # Add edges with weights
    for (i, j), weight in zip(edges, weights):
        G.add_edge(genes[i], genes[j], weight=weight)
    
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
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embedding_matrix) #how determine number of clusters? --> need to look into how to determine
    #may need to group the cells with particular marker first, then investigate different groups of cells' networks
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

def visualize_graph(node_embeddings):
    tsne = sklearn.manifold.TSNE(n_components=2)
    node_embeddings_tsne = tsne.fit_transform(node_embeddings)
    alpha = 0.7
    plt.figure(figsize=(10, 8))
    plt.scatter(
        node_embeddings_tsne[:, 0],
        node_embeddings_tsne[:, 1],
        cmap="jet",
        alpha=alpha
    )

def ANOVA(control_embeddings, treatment1_embeddings, treatment2_embeddings):
    # Assume embeddings is a NumPy array (genes x dimensions), labels is an array of conditions
    p_values = []
    for dim in range(control_embeddings.shape[1]):
        control = control_embeddings[dim]
        treatment1 = treatment1_embeddings[dim]
        treatment2 = treatment2_embeddings[dim]
        _, p = f_oneway(control, treatment1, treatment2)
        p_values.append(p)

    # Multiple testing correction
    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1] #control for false discovery rate

    # Identify significant dimensions
    significant_dims = np.where(adjusted_p_values < 0.05)[0]
    print(f"Significant embedding dimensions: {significant_dims}")
    return significant_dims

def extract_top_genes(significant_dims, embeddings):
    overall_top_genes = []
    for dim in significant_dims:
        top_genes = np.argsort(-np.abs(embeddings[:, dim]))[:10]  # Top 10 genes by absolute value
        print(f"Significant dimension {dim}: Top contributing genes {adata.var_names[top_genes]}")
        overall_top_genes.append(top_genes)
    return overall_top_genes

def calculate_network_metrics(G):
    metrics = {
        "average degree": np.mean([d for _, d in G.degree()]),
        "clustering coefficient": nx.average_clustering(G),
        "density": nx.density(G),
        "avg_path_length": nx.average_shortest_path_length(G),
    }
    return metrics

def calculate_grn_metrics(grn_control, grn_treatment1):
    # Overlap of edges between control and treatment
    edges_control = set(grn_control.edges())
    edges_treatment = set(grn_treatment1.edges())
    overlap = len(edges_control & edges_treatment)

    # Fisher's test for overlap
    contingency_table = [
        [overlap, len(edges_control) - overlap],
        [len(edges_treatment) - overlap, len(genes)**2 - len(edges_control) - len(edges_treatment) + overlap]
    ]
    odds_ratio, p_value = fisher_exact(contingency_table)
    print(f"Edge overlap p-value: {p_value}")

#Differential Connectivity Analysis

def calculate_node_metrics(G):
    metrics = {
        "degree": dict(G.degree()),
        "betweenness": nx.betweenness_centrality(G),
        "clustering": nx.clustering(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
    }
    return metrics

def louvain_subnetwork_analysis():
    # Detect communities in GRNs
    communities_control = list(community.louvain_communities(grn_control))
    communities_treatment = list(community.louvain_communities(grn_treatment1))

    # Compare average degrees within communities
    community_degrees_control = [np.mean([grn_control.degree[n] for n in c]) for c in communities_control]
    community_degrees_treatment = [np.mean([grn_treatment1.degree[n] for n in c]) for c in communities_treatment]

    print("Control community degrees:", community_degrees_control)
    print("Treatment community degrees:", community_degrees_treatment)
    return community_degrees_control, community_degrees_treatment

def visualize_node_degree_differences():
    # Degree differences
    degree_diff = {gene: metrics_treatment1["degree"].get(gene, 0) - metrics_control["degree"].get(gene, 0) 
               for gene in grn_control.nodes}

    # Plot network with degree differences
    pos = nx.spring_layout(grn_control)  # Layout for visualization
    nx.draw(grn_control, pos, node_color=list(degree_diff.values()), cmap=plt.cm.coolwarm, with_labels=True)
    plt.title("Differential Node Degrees (Treatment 1 - Control)")
    plt.show()


#comparing networks constructed using different methods

def edge_jaccard_similarity(G1, G2):
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    return intersection / union

# Assume `clusters_method1` and `clusters_method2` are lists of sets (clusters of nodes)
def cluster_similarity(clusters1, clusters2):
    """
    Used to compare network subcommunities from networks created using different methods.
    """
    # Flatten clusters to node-to-cluster mappings
    node_to_cluster1 = {node: i for i, cluster in enumerate(clusters1) for node in cluster}
    node_to_cluster2 = {node: i for i, cluster in enumerate(clusters2) for node in cluster}

    # Align node sets
    common_nodes = set(node_to_cluster1.keys()) & set(node_to_cluster2.keys())
    labels1 = [node_to_cluster1[node] for node in common_nodes]
    labels2 = [node_to_cluster2[node] for node in common_nodes]

    return adjusted_rand_score(labels1, labels2)

def node_level_comparison(network1, network2):
    degree_method1 = dict(network_method1.degree())
    degree_method2 = dict(network_method2.degree())

    # Only compare nodes that are present in both networks
    common_nodes = set(degree_method1.keys()) & set(degree_method2.keys())
    degrees1 = [degree_method1[node] for node in common_nodes]
    degrees2 = [degree_method2[node] for node in common_nodes]

    correlation, pval = scipy.stats.pearsonr(degrees1, degrees2)
    print("Degree Correlation:", correlation)

def embeddings_level_comparison():
    # Assume embeddings_method1 and embeddings_method2 are numpy arrays (nodes x dimensions)
    similarity_matrix = cosine_similarity(embeddings_method1, embeddings_method2)
    average_similarity = similarity_matrix.mean()
    print("Average Cosine Similarity:", average_similarity)
    return similarity_matrix, average_similarity





if __name__=='__main__':
    control_filepath = "C:\\Users\\a303\\Downloads\\alzheimers_data\\data\\control"
    group2_filepath = "C:\\Users\\a303\\Downloads\\alzheimers_data\\data\\group2"
    group3_filepath = "C:\\Users\\a303\\Downloads\\alzheimers_data\\data\\group3"
    #combined group
    control_data = load_and_preprocess_data([control_filepath], batch_labels=["control"])
    group2_data = load_and_preprocess_data([group2_filepath], batch_labels=["treatment1"])
    group3_data = load_and_preprocess_data([group3_filepath], batch_labels=["treatment2"])
    #cell_type_annotation(data)
    control_graph = construct_grn(control_data)
    visualize_graph(control_graph)
    group2_graph = construct_grn(group2_data)
    visualize_graph(group2_graph)
    group3_graph = construct_grn(group3_data)
    visualize_graph(group3_graph)
    
    #control only
    #control_data = load_and_preprocess_data(control_filepath)
    #cell_type_annotation(control_data)
    #graph = construct_grn(control_data)
    #group2 only
    #group2_data = load_and_preprocess_data(group2_filepath)
    #cell_type_annotation(group2_data)
    #graph = construct_grn(group2_data)
    #group3
    #group3_data = load_and_preprocess_data(group3_filepath)
    #cell_type_annotation(group3_data)
    #graph = construct_grn(group3_data)
    #embeddings = apply_node2vec(graph)
    #visualize(embeddings, graph)
    #clusters = cluster(embeddings, graph)
    #gsea(graph, clusters)
    #predict_regulatory_relationships(graph, embeddings)
    #calculate AUC and metrics
    #create networks for patient vs healthy
    #think about how this compares to scorpion, need way to compare with other algorithms
    #can compare to scorpion's coregulatory network
    #can see if embedding improves existing methods like scorpion, panda
    #string protein-protein interaction network database for knowledge-driven (our approach is data-driven); string database has API, can use their datasets  #maybe try to run node2vec to create embeddings for protein-protein interactions
    #look into GNN and other machine learning 
    #look at scorpion's code, see if can translate some of it to python
    #scorpion uses a correlation coefficient network, compare it to this


    #better indicate pipeline, use workflow, step by step
    #provide slides for future work, indicate paper you want to read to advance discovery/fill knowledge gap
    #compare algorithms' edges; which algorithms create which types of edges?

    #use scanpy's standard latent space using their standard pipeline, they use biomarker to distinguish the cell type, can also overlay any spatial data on the projection of the UMAP
    #use comparisons between the cell types for comparison, generate the diffentially expressed genes (this is the most important)
        #can use these to check the GRN and also discover new functions
    #next week goal: run scanpy, generate basic picture
    #set up synapse account and get data from ssREAD
    #machine learning methods for comparing and integrating multiple networks?
    #GCNN with node2vec embeddings??



