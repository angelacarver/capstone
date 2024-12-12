import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import VGAE, GCNConv

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 4 * out_channels)
        self.conv2 = GCNConv(4 * out_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)
    
    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index).relu()
        x2 = self.conv2(x1, edge_index).relu()
        mu = self.conv_mu(x2, edge_index)
        logstd = self.conv_logstd(x2, edge_index)
        return mu, logstd
    
    def build_adjacency_matrix(adata, threshold=0.3):
        expression_matrix = adata.X.toarray()
        correlation_matrix = torch.tensor(np.corrcoef(expression_matrix), dtype=torch.float32)
        correlation_matrix[torch.abs(correlation_matrix) < threshold] = 0
        adjacency_matrix = coo_matrix(correlation_matrix)
        return from_scipy_sparse_matrix(adjacency_matrix)

    def train_vgae(data, in_channels, out_channels, beta, epochs=100, lr=0.01):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VGAE(GCNEncoder(in_channels, out_channels)).to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            loss = model.recon_loss(data.edge_index, z) + (1 / data.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
        
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
        return model, z

    def evaluate_vgae(model, data):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
            reconstructed_adj = model.decode(z, edge_index=data.edge_index)
        return reconstructed_adj



# Function to create adjacency matrix with memory efficiency
def create_sparse_adjacency(num_nodes, density=0.5):
    print("Creating sparse adjacency matrix...")
    graph = nx.erdos_renyi_graph(num_nodes, density)
    adjacency_matrix = nx.adjacency_matrix(graph)  # Sparse matrix representation
    return from_scipy_sparse_matrix(adjacency_matrix)



def build_sparse_adjacency_from_embeddings(embeddings, threshold=0.25):
    """
    Build a sparse adjacency matrix using node2vec embeddings.
    Args:
        embeddings (torch.Tensor): Node2Vec embeddings (nodes x features).
        threshold (float): Similarity threshold to create edges.
    Returns:
        scipy.sparse.coo_matrix: Sparse adjacency matrix.
    """
    # Convert embeddings to numpy array for processing
    embeddings = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Retain only edges above the threshold
    mask = similarity_matrix >= threshold
    row, col = np.where(mask)
    data = similarity_matrix[row, col]

    # Create sparse adjacency matrix
    adjacency_matrix = coo_matrix((data, (row, col)), shape=(embeddings.shape[0], embeddings.shape[0]))
    return adjacency_matrix


def custom_recon_loss(z, edge_index, num_nodes):
    """
    Custom reconstruction loss using binary cross-entropy.

    Args:
        z (torch.Tensor): Latent embeddings of shape [num_nodes, latent_dim].
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        num_nodes (int): Total number of nodes in the graph.
    
    Returns:
        torch.Tensor: The custom loss.
    """
    # Reconstruct adjacency matrix
    logits = (z @ z.t()).view(-1)  # Shape: [num_nodes * num_nodes]
    
    # Create labels for existing edges (1) and non-edges (0)
    labels = torch.zeros(num_nodes, num_nodes, device=z.device)
    labels[edge_index[0], edge_index[1]] = 1
    labels = labels.view(-1)  # Shape: [num_nodes * num_nodes]
    
    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss


def weighted_recon_loss(z, edge_index, edge_weight):
    """
    Custom reconstruction loss using binary cross-entropy with weighted edges.

    Args:
        z (torch.Tensor): Latent embeddings of shape [num_nodes, latent_dim].
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        edge_weight (torch.Tensor): Weights for each edge.
        num_nodes (int): Total number of nodes in the graph.
    
    Returns:
        torch.Tensor: The custom loss.
    """
    # Calculate the dot product of the latent vectors of connected nodes
    row, col = edge_index
    logits = (z[row] * z[col]).sum(dim=1)  # Shape: [num_edges]

    # Binary cross-entropy loss (weighted by edge weights)
    loss = F.binary_cross_entropy_with_logits(logits, edge_weight)
    return loss



def train_vgae(data, model, beta, epochs=100, lr=0.01):
    """
    Train VGAE without batching.
    
    Args:
        data (torch_geometric.data.Data): Input data containing features and edge index.
        model (torch_geometric.nn.VGAE): VGAE model.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        
    Returns:
        model: Trained VGAE model.
        z: Latent embeddings of nodes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Encode and calculate loss
        z = model.encode(data.x, data.edge_index)
        recon_loss = custom_recon_loss(z, data.edge_index, data.num_nodes)#data.edge_index, z)
        kl_loss = (1 / data.num_nodes) * model.kl_loss()
        loss = recon_loss + beta * kl_loss

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
    print(f"Epoch {epoch}/{epochs}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
    return model, z

def visualize_reconstructed_adjacency_matrix(adj_reconstructed, name):
    weights = adj_reconstructed.flatten()
    plt.hist(weights, bins=50)
    plt.xlabel("Reconstructed Edge Weight")
    plt.ylabel("Frequency")
    name = name + '.png'
    plt.savefig(name)


# Main pipeline
if __name__ == "__main__":
    # Load embeddings efficiently
    print("Loading data...")

    #load in adata object
    with open('control_data.pickle', 'rb') as f:
        control_data = pickle.load(f)

    with open('group2_data.pickle', 'rb') as f:
        group2_data = pickle.load(f)

    with open('group3_data.pickle', 'rb') as f:
        group3_data = pickle.load(f)

    print("Control Data Network")
    beta = 100
    X = control_data.X.toarray()

    similarity_matrix = cosine_similarity(X)

    threshold = 0.7
    adjacency_matrix = np.where(similarity_matrix > threshold, similarity_matrix, 0)
    sparse_adj = coo_matrix(adjacency_matrix)
    row, col = sparse_adj.row, sparse_adj.col
    weights = sparse_adj.data #similarity values

    #control_embeddings_list = torch.tensor(list(control_embeddings.values()), dtype=torch.float32)
    control_embeddings_list = torch.tensor(X, dtype=torch.float32)

    # Initialize sparse adjacency matrix
    #sparse_adj = build_sparse_adjacency_from_embeddings(control_embeddings_list)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_index = edge_index.to(torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    # Prepare data for PyTorch Geometric
    print("Preparing data for PyTorch Geometric...")
    data = Data(x=control_embeddings_list, edge_index=edge_index)

    #initialize model
    # Determine input and output dimensions
    in_channels = control_embeddings_list.size(1)  # Number of features in the embeddings
    out_channels = 16  # Latent dimension size (can be adjusted)

    # Initialize VGAE with the GCN encoder
    encoder = GCNEncoder(in_channels, out_channels)
    model = VGAE(encoder)



    # Train the VGAE in batches
    print("Training model...")
    model, z = train_vgae(data, model, beta, epochs=150, lr=0.001)

    # Evaluate the GRN reconstruction
    print("Reconstructing adjacency matrix...")
    reconstructed_adjacency = GCNEncoder.evaluate_vgae(model, data)
    np.save('control_reconstructed_adjacency_matrix.npy', reconstructed_adjacency.cpu().detach().numpy())
    print("Reconstructed adjacency matrix:")
    print(reconstructed_adjacency)
    visualize_reconstructed_adjacency_matrix(reconstructed_adjacency.detach().cpu().numpy(), 'control')

    # Save the embeddings efficiently
    print("Saving embeddings...")
    np.save("vgae_control_node_embeddings.npy", z.cpu().detach().numpy())

    print('Treatment Group 1 Network')
    X = group2_data.X.toarray()

    similarity_matrix = cosine_similarity(X)

    adjacency_matrix = np.where(similarity_matrix > threshold, similarity_matrix, 0)
    sparse_adj = coo_matrix(adjacency_matrix)
    row, col = sparse_adj.row, sparse_adj.col
    weights = sparse_adj.data #similarity values

    control_embeddings_list = torch.tensor(X, dtype=torch.float32)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_index = edge_index.to(torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    # Prepare data for PyTorch Geometric
    print("Preparing data for PyTorch Geometric...")
    data = Data(x=control_embeddings_list, edge_index=edge_index)

    #initialize model
    # Determine input and output dimensions
    in_channels = control_embeddings_list.size(1)  # Number of features in the embeddings
    out_channels = 16  # Latent dimension size (can be adjusted)

    # Initialize VGAE with the GCN encoder
    encoder = GCNEncoder(in_channels, out_channels)
    model = VGAE(encoder)


    # Train the VGAE in batches
    print("Training model...")
    model, z = train_vgae(data, model, beta, epochs=150, lr=0.001)

    # Evaluate the GRN reconstruction
    print("Reconstructing adjacency matrix...")
    reconstructed_adjacency = GCNEncoder.evaluate_vgae(model, data)
    print("Reconstructed adjacency matrix:")
    print(reconstructed_adjacency)
    visualize_reconstructed_adjacency_matrix(reconstructed_adjacency.detach().cpu().numpy(), 'group2')
    np.save('group2_reconstructed_adjacency_matrix.npy', reconstructed_adjacency.cpu().detach().numpy())

    # Save the embeddings efficiently
    print("Saving embeddings...")
    np.save("vgae_group2_node_embeddings.npy", z.cpu().detach().numpy())

    print("Treatment Group 2 Network")

    X = group3_data.X.toarray()

    similarity_matrix = cosine_similarity(X)

    #threshold = 0.25
    adjacency_matrix = np.where(similarity_matrix > threshold, similarity_matrix, 0)
    sparse_adj = coo_matrix(adjacency_matrix)
    row, col = sparse_adj.row, sparse_adj.col
    weights = sparse_adj.data #similarity values

    #control_embeddings_list = torch.tensor(list(control_embeddings.values()), dtype=torch.float32)
    control_embeddings_list = torch.tensor(X, dtype=torch.float32)

    # Initialize sparse adjacency matrix
    #sparse_adj = build_sparse_adjacency_from_embeddings(control_embeddings_list)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_index = edge_index.to(torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    # Prepare data for PyTorch Geometric
    print("Preparing data for PyTorch Geometric...")
    data = Data(x=control_embeddings_list, edge_index=edge_index)

    #initialize model
    # Determine input and output dimensions
    in_channels = control_embeddings_list.size(1)  # Number of features in the embeddings
    out_channels = 16  # Latent dimension size (can be adjusted)

    # Initialize VGAE with the GCN encoder
    encoder = GCNEncoder(in_channels, out_channels)
    model = VGAE(encoder)

    # Train the VGAE in batches
    print("Training model...")
    model, z = train_vgae(data, model, beta, epochs=150, lr=0.001)

    # Evaluate the GRN reconstruction
    print("Reconstructing adjacency matrix...")
    reconstructed_adjacency = GCNEncoder.evaluate_vgae(model, data)
    print("Reconstructed adjacency matrix:")
    print(reconstructed_adjacency)
    visualize_reconstructed_adjacency_matrix(reconstructed_adjacency.detach().cpu().numpy(), 'group3')
    np.save('group3_reconstructed_adjacency_matrix.npy', reconstructed_adjacency.cpu().detach().numpy())

    # Save the embeddings efficiently
    print("Saving embeddings...")
    np.save("vgae_group3_node_embeddings.npy", z.cpu().detach().numpy())


    print("Done")
