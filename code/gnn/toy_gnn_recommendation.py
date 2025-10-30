"""
GNN-Based Item Recommendation - Toy Example Implementation
===========================================================

This implementation demonstrates link prediction using Graph Neural Networks
for a bipartite user-item graph with the following data:

User-Item Interactions:
- shreya -> movie
- rk -> sports
- rk -> movie
- rp -> book
- shreya -> book
- rp -> movie

Goal: Predict which items a user might be interested in (link prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import networkx as nx

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleGCNLayer(nn.Module):
    """
    A simple Graph Convolutional Network layer.
    
    Implements: H' = σ(D^(-1/2) A D^(-1/2) H W)
    Where:
    - H: Input node features
    - A: Adjacency matrix (with self-loops)
    - D: Degree matrix
    - W: Learnable weight matrix
    - σ: Activation function
    """
    
    def __init__(self, in_features: int, out_features: int):
        super(SimpleGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj_norm: Normalized adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Message passing: aggregate neighbor features
        aggregated = torch.matmul(adj_norm, x)
        # Apply learnable transformation
        output = torch.matmul(aggregated, self.weight)
        return output


class GNNRecommender(nn.Module):
    """
    Two-layer GNN for link prediction in bipartite user-item graphs.
    
    Architecture:
    1. GCN Layer 1: input_dim -> hidden_dim
    2. ReLU activation
    3. GCN Layer 2: hidden_dim -> embedding_dim
    4. Link prediction via dot product of node embeddings
    """
    
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, embedding_dim: int):
        super(GNNRecommender, self).__init__()
        
        # Two GCN layers for learning node embeddings
        self.gcn1 = SimpleGCNLayer(input_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, embedding_dim)
        
        # Initial node features (learnable)
        self.node_features = nn.Parameter(torch.randn(num_nodes, input_dim))
        
    def forward(self, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute node embeddings.
        
        Args:
            adj_norm: Normalized adjacency matrix
            
        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        # Layer 1: Feature transformation + aggregation
        x = self.gcn1(self.node_features, adj_norm)
        x = F.relu(x)
        
        # Layer 2: Generate final embeddings
        x = self.gcn2(x, adj_norm)
        
        return x
    
    def predict_link(self, node_embeddings: torch.Tensor, user_idx: int, item_idx: int) -> torch.Tensor:
        """
        Predict link probability between a user and an item using dot product.
        
        Args:
            node_embeddings: Learned node embeddings
            user_idx: Index of the user node
            item_idx: Index of the item node
            
        Returns:
            Link prediction score
        """
        user_emb = node_embeddings[user_idx]
        item_emb = node_embeddings[item_idx]
        # Dot product similarity
        score = torch.dot(user_emb, item_emb)
        return torch.sigmoid(score)
    
    def predict_all_links(self, node_embeddings: torch.Tensor, user_indices: List[int], item_indices: List[int]) -> torch.Tensor:
        """
        Predict scores for all user-item pairs.
        
        Args:
            node_embeddings: Learned node embeddings
            user_indices: List of user node indices
            item_indices: List of item node indices
            
        Returns:
            Score matrix [num_users, num_items]
        """
        user_embs = node_embeddings[user_indices]  # [num_users, embedding_dim]
        item_embs = node_embeddings[item_indices]  # [num_items, embedding_dim]
        
        # Compute all pairwise dot products
        scores = torch.matmul(user_embs, item_embs.t())  # [num_users, num_items]
        return torch.sigmoid(scores)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    
    Args:
        adj: Adjacency matrix with self-loops
        
    Returns:
        Normalized adjacency matrix
    """
    # Calculate degree matrix
    degree = torch.sum(adj, dim=1)
    
    # D^(-1/2)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    
    # Create diagonal matrix
    degree_mat_inv_sqrt = torch.diag(degree_inv_sqrt)
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    adj_norm = torch.matmul(torch.matmul(degree_mat_inv_sqrt, adj), degree_mat_inv_sqrt)
    
    return adj_norm


def create_graph_data():
    """
    Create the bipartite user-item graph from the toy example.
    
    Returns:
        Tuple of (adjacency matrix, node_id_to_name mapping, user indices, item indices, edge list)
    """
    # Define the interactions
    interactions = [
        ('shreya', 'movie'),
        ('rk', 'sports'),
        ('rk', 'movie'),
        ('rp', 'book'),
        ('shreya', 'book'),
        ('rp', 'movie'),
    ]
    
    # Extract unique users and items
    users = sorted(list(set([u for u, _ in interactions])))
    items = sorted(list(set([i for _, i in interactions])))
    
    print("=" * 60)
    print("GRAPH CONSTRUCTION")
    print("=" * 60)
    print(f"Users: {users}")
    print(f"Items: {items}")
    print(f"Total nodes: {len(users) + len(items)}")
    print(f"Total edges: {len(interactions)}")
    
    # Create node mappings
    node_to_id = {}
    id_to_node = {}
    
    # Assign IDs: users first, then items
    for i, user in enumerate(users):
        node_to_id[user] = i
        id_to_node[i] = user
    
    for i, item in enumerate(items):
        node_to_id[item] = len(users) + i
        id_to_node[len(users) + i] = item
    
    num_nodes = len(users) + len(items)
    user_indices = list(range(len(users)))
    item_indices = list(range(len(users), num_nodes))
    
    # Create adjacency matrix (undirected bipartite graph)
    adj = torch.zeros(num_nodes, num_nodes)
    
    edge_list = []
    for user, item in interactions:
        user_id = node_to_id[user]
        item_id = node_to_id[item]
        adj[user_id, item_id] = 1
        adj[item_id, user_id] = 1  # Undirected
        edge_list.append((user_id, item_id))
    
    # Add self-loops
    adj = adj + torch.eye(num_nodes)
    
    return adj, id_to_node, user_indices, item_indices, edge_list, node_to_id


def visualize_graph(adj: torch.Tensor, id_to_node: dict, edge_list: List[Tuple[int, int]]):
    """Visualize the bipartite user-item graph."""
    G = nx.Graph()
    
    # Add nodes
    users = []
    items = []
    for node_id, name in id_to_node.items():
        G.add_node(node_id, label=name)
        if node_id < 3:  # First 3 are users
            users.append(node_id)
        else:
            items.append(node_id)
    
    # Add edges (excluding self-loops)
    for u, v in edge_list:
        G.add_edge(u, v)
    
    # Create bipartite layout
    pos = {}
    user_y = np.linspace(0, 1, len(users))
    item_y = np.linspace(0, 1, len(items))
    
    for i, user in enumerate(users):
        pos[user] = (0, user_y[i])
    
    for i, item in enumerate(items):
        pos[item] = (1, item_y[i])
    
    plt.figure(figsize=(10, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=users, node_color='lightblue', 
                          node_size=1500, label='Users')
    nx.draw_networkx_nodes(G, pos, nodelist=items, node_color='lightcoral', 
                          node_size=1500, label='Items')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6)
    
    # Draw labels
    labels = {node_id: id_to_node[node_id] for node_id in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.title("User-Item Bipartite Graph", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('user_item_graph.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graph visualization saved as 'user_item_graph.png'")
    plt.close()


def train_model(model: GNNRecommender, adj_norm: torch.Tensor, edge_list: List[Tuple[int, int]], 
                num_nodes: int, epochs: int = 200, lr: float = 0.01):
    """
    Train the GNN model using positive and negative edge samples.
    
    Args:
        model: GNN recommender model
        adj_norm: Normalized adjacency matrix
        edge_list: List of positive edges (user-item interactions)
        num_nodes: Total number of nodes
        epochs: Number of training epochs
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Create training data: positive and negative samples
    pos_edges = torch.tensor(edge_list, dtype=torch.long)
    
    # Generate negative edges (non-existent user-item pairs)
    neg_edges = []
    user_indices = set(range(3))  # First 3 nodes are users
    item_indices = set(range(3, num_nodes))  # Remaining nodes are items
    
    existing_edges = set([(u, v) for u, v in edge_list])
    
    for user in user_indices:
        for item in item_indices:
            if (user, item) not in existing_edges:
                neg_edges.append((user, item))
    
    neg_edges = torch.tensor(neg_edges[:len(edge_list)], dtype=torch.long)  # Balance pos/neg
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Positive samples: {len(pos_edges)}")
    print(f"Negative samples: {len(neg_edges)}")
    print(f"Epochs: {epochs}, Learning rate: {lr}")
    print("-" * 60)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass: get node embeddings
        embeddings = model(adj_norm)
        
        # Compute predictions for positive edges
        pos_scores = []
        for u, v in pos_edges:
            score = model.predict_link(embeddings, u.item(), v.item())
            pos_scores.append(score)
        pos_scores = torch.stack(pos_scores)
        
        # Compute predictions for negative edges
        neg_scores = []
        for u, v in neg_edges:
            score = model.predict_link(embeddings, u.item(), v.item())
            neg_scores.append(score)
        neg_scores = torch.stack(neg_scores)
        
        # Combine scores and labels
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))])
        
        # Compute loss
        loss = criterion(scores, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f}")
    
    print("-" * 60)
    print(f"✓ Training complete! Final loss: {losses[-1]:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print("✓ Training loss plot saved as 'training_loss.png'")
    plt.close()
    
    return losses


def evaluate_recommendations(model: GNNRecommender, adj_norm: torch.Tensor, 
                            id_to_node: dict, user_indices: List[int], 
                            item_indices: List[int], node_to_id: dict):
    """
    Generate and display recommendations for all users.
    
    Args:
        model: Trained GNN model
        adj_norm: Normalized adjacency matrix
        id_to_node: Mapping from node ID to name
        user_indices: List of user node indices
        item_indices: List of item node indices
        node_to_id: Mapping from node name to ID
    """
    model.eval()
    
    with torch.no_grad():
        # Get final embeddings
        embeddings = model(adj_norm)
        
        # Predict scores for all user-item pairs
        scores = model.predict_all_links(embeddings, user_indices, item_indices)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Display recommendation scores
    print("\nPrediction Scores (User x Item):")
    print("-" * 60)
    
    # Header
    item_names = [id_to_node[i] for i in item_indices]
    header = "User".ljust(10) + " | " + " | ".join([name.center(8) for name in item_names])
    print(header)
    print("-" * 60)
    
    # Scores for each user
    for i, user_idx in enumerate(user_indices):
        user_name = id_to_node[user_idx]
        score_str = user_name.ljust(10) + " | "
        score_str += " | ".join([f"{scores[i, j].item():.4f}".center(8) for j in range(len(item_indices))])
        print(score_str)
    
    print("\n" + "=" * 60)
    print("TOP RECOMMENDATIONS (sorted by score)")
    print("=" * 60)
    
    # For each user, show top recommendations
    for i, user_idx in enumerate(user_indices):
        user_name = id_to_node[user_idx]
        user_scores = scores[i].numpy()
        
        # Sort items by score
        sorted_indices = np.argsort(user_scores)[::-1]
        
        print(f"\n{user_name.upper()}:")
        print("-" * 40)
        for rank, idx in enumerate(sorted_indices, 1):
            item_name = id_to_node[item_indices[idx]]
            score = user_scores[idx]
            
            # Check if this is an existing interaction
            actual_edge = (user_idx, item_indices[idx])
            is_known = "✓ (known)" if actual_edge in [(node_to_id[u], node_to_id[i]) 
                                                        for u, i in [('shreya', 'movie'), ('rk', 'sports'), 
                                                                    ('rk', 'movie'), ('rp', 'book'), 
                                                                    ('shreya', 'book'), ('rp', 'movie')]] else "★ (new)"
            
            print(f"  {rank}. {item_name.ljust(10)} - Score: {score:.4f} {is_known}")
    
    # Visualize recommendation heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(scores.numpy(), cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Recommendation Score')
    plt.xticks(range(len(item_indices)), item_names, rotation=45, ha='right')
    plt.yticks(range(len(user_indices)), [id_to_node[i] for i in user_indices])
    plt.xlabel('Items', fontsize=12)
    plt.ylabel('Users', fontsize=12)
    plt.title('Recommendation Score Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(user_indices)):
        for j in range(len(item_indices)):
            plt.text(j, i, f'{scores[i, j].item():.2f}', 
                    ha='center', va='center', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('recommendation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Recommendation heatmap saved as 'recommendation_heatmap.png'")
    plt.close()


def visualize_embeddings(model: GNNRecommender, adj_norm: torch.Tensor, 
                        id_to_node: dict, user_indices: List[int], item_indices: List[int]):
    """Visualize learned node embeddings using PCA."""
    model.eval()
    
    with torch.no_grad():
        embeddings = model(adj_norm).numpy()
    
    # Apply PCA for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 7))
    
    # Plot user embeddings
    user_embs = embeddings_2d[user_indices]
    plt.scatter(user_embs[:, 0], user_embs[:, 1], c='lightblue', s=300, 
               label='Users', edgecolors='black', linewidth=2)
    
    # Plot item embeddings
    item_embs = embeddings_2d[item_indices]
    plt.scatter(item_embs[:, 0], item_embs[:, 1], c='lightcoral', s=300, 
               label='Items', edgecolors='black', linewidth=2, marker='s')
    
    # Add labels
    for idx in user_indices + item_indices:
        plt.annotate(id_to_node[idx], (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    fontsize=11, fontweight='bold', ha='center', va='center')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title('Learned Node Embeddings (2D projection via PCA)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('node_embeddings.png', dpi=300, bbox_inches='tight')
    print("✓ Node embeddings visualization saved as 'node_embeddings.png'")
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("GNN-BASED ITEM RECOMMENDATION - TOY EXAMPLE")
    print("=" * 60)
    print("\nThis implementation demonstrates link prediction using GNNs")
    print("to recommend items to users based on their interaction history.\n")
    
    # Step 1: Create graph data
    adj, id_to_node, user_indices, item_indices, edge_list, node_to_id = create_graph_data()
    
    # Step 2: Normalize adjacency matrix
    adj_norm = normalize_adjacency(adj)
    print("\n✓ Adjacency matrix normalized for GNN processing")
    
    # Step 3: Visualize the graph
    visualize_graph(adj, id_to_node, edge_list)
    
    # Step 4: Initialize the model
    num_nodes = len(id_to_node)
    input_dim = 8      # Initial feature dimension
    hidden_dim = 16    # Hidden layer dimension
    embedding_dim = 8  # Final embedding dimension
    
    model = GNNRecommender(num_nodes, input_dim, hidden_dim, embedding_dim)
    
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(f"Input dimension:     {input_dim}")
    print(f"Hidden dimension:    {hidden_dim}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Total parameters:    {sum(p.numel() for p in model.parameters())}")
    
    # Step 5: Train the model
    losses = train_model(model, adj_norm, edge_list, num_nodes, epochs=200, lr=0.01)
    
    # Step 6: Evaluate and generate recommendations
    evaluate_recommendations(model, adj_norm, id_to_node, user_indices, item_indices, node_to_id)
    
    # Step 7: Visualize learned embeddings
    try:
        visualize_embeddings(model, adj_norm, id_to_node, user_indices, item_indices)
    except ImportError:
        print("\nNote: sklearn not available for PCA visualization")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Graph constructed from user-item interactions")
    print("✓ GNN model trained using link prediction objective")
    print("✓ Node embeddings learned through message passing")
    print("✓ Recommendations generated based on embedding similarity")
    print("✓ Visualizations saved to current directory")
    print("\nThe model successfully learned to:")
    print("  1. Encode user preferences and item characteristics")
    print("  2. Predict likelihood of user-item interactions")
    print("  3. Recommend new items to users")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
