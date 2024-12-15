import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import scanpy as sc


"""
Creates a GCN encoder for the VGAE model 
"""


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    # Forward pass
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    # Build adjacency matrix
    def build_adjacency_matrix(adata, threshold=0.3):
        expression_matrix = adata.X.toarray()
        correlation_matrix = torch.tensor(
            np.corrcoef(expression_matrix), dtype=torch.float32
        )
        correlation_matrix[torch.abs(correlation_matrix) < threshold] = 0
        adjacency_matrix = coo_matrix(correlation_matrix)
        return from_scipy_sparse_matrix(adjacency_matrix)

    # Train VGAE
    def train_vgae(data, in_channels, out_channels, epochs=100, lr=0.01):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGAE(GCNEncoder(in_channels, out_channels)).to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            loss = (
                model.recon_loss(data.edge_index, z)
                + (1 / data.num_nodes) * model.kl_loss()
            )
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

        return model, z

    # Evaluate VGAE
    def evaluate_vgae(model, data):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
            reconstructed_adj = model.decode(z, edge_index=data.edge_index)
        return reconstructed_adj


# Main pipeline
if __name__ == "__main__":
    # Load the .mtx data
    adata = load_data("matrix.mtx", "genes.txt", "barcodes.txt")

    # Preprocess features and adjacency matrix
    scaler = StandardScaler()
    adata.X = scaler.fit_transform(adata.X)  # Normalize gene expression
    x = torch.tensor(adata.X, dtype=torch.float32)
    edge_index, edge_weight = build_adjacency_matrix(adata)

    # Prepare data for PyTorch Geometric
    data = Data(x=x, edge_index=edge_index)

    # Train the VGAE
    model, embeddings = train_vgae(data, in_channels=x.size(1), out_channels=16)

    # Evaluate the GRN reconstruction
    reconstructed_adjacency = evaluate_vgae(model, data)
    print("Reconstructed adjacency matrix:")
    print(reconstructed_adjacency)

    # Save the embeddings
    np.save("node_embeddings.npy", embeddings.cpu().numpy())
