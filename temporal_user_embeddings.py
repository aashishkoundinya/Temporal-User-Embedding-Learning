import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx
import os
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# Define the Temporal Graph Neural Network model
class TemporalGNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128):
        super(TemporalGNN, self).__init__()
        
        # Initial embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # GNN layers
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Prediction layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        
    def forward(self, x, edge_index, batch_user_indices, batch_item_indices):
        # Apply GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Get user and item embeddings from the GNN output
        user_embeddings = x[batch_user_indices]
        num_users = x.size(0) - len(torch.unique(batch_item_indices))
        item_embeddings = x[batch_item_indices + num_users]
        
        # Concatenate user and item embeddings
        combined = torch.cat([user_embeddings, item_embeddings], dim=1)
        
        # Prediction layers
        x = F.relu(self.fc1(combined))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        
        return torch.sigmoid(x).squeeze()
    
    def get_user_embedding(self, x, edge_index, user_idx):
        """Get the embedding for a specific user"""
        # Apply GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # Return the user's embedding
        return x[user_idx].detach().numpy()


# Function to create temporal graph from user-item interactions
def create_temporal_graph(ratings_df, time_windows=3):
    """
    Create temporal graphs from user-item interactions.
    
    Args:
        ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
        time_windows: Number of time windows to divide the data into
    
    Returns:
        List of graph data objects for each time window
    """
    # Sort ratings by timestamp
    ratings_df = ratings_df.sort_values('timestamp')
    
    # Get min and max timestamps
    min_timestamp = ratings_df['timestamp'].min()
    max_timestamp = ratings_df['timestamp'].max()
    
    # Calculate time window boundaries
    window_size = (max_timestamp - min_timestamp) / time_windows
    window_boundaries = [min_timestamp + i * window_size for i in range(time_windows + 1)]
    
    # Create graphs for each time window
    graph_data_list = []
    
    for i in range(time_windows):
        start_time = window_boundaries[i]
        end_time = window_boundaries[i + 1]
        
        # Get interactions for this time window
        window_df = ratings_df[(ratings_df['timestamp'] >= start_time) & 
                               (ratings_df['timestamp'] < end_time)]
        
        if len(window_df) == 0:
            continue
        
        # Create graph
        graph_data = create_graph_from_interactions(window_df)
        graph_data_list.append(graph_data)
        
        print(f"Time window {i+1}: {len(window_df)} interactions, "
              f"{graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    return graph_data_list


# Function to create a graph from a set of interactions
def create_graph_from_interactions(interactions_df, embedding_dim=64):
    """
    Create a bipartite graph from user-item interactions.
    
    Args:
        interactions_df: DataFrame with columns [userId, movieId, rating, timestamp]
        embedding_dim: Dimension of node features to match GNN input expectations
    
    Returns:
        PyTorch Geometric Data object
    """
    # Get unique users and items
    unique_users = interactions_df['userId'].unique()
    unique_items = interactions_df['movieId'].unique()
    
    # Create mapping from original IDs to consecutive indices
    user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_mapping = {item_id: idx + len(user_mapping) for idx, item_id in enumerate(unique_items)}
    
    # Create edges (user -> item and item -> user)
    edge_index = []
    edge_attr = []
    
    for _, row in interactions_df.iterrows():
        user_idx = user_mapping[row['userId']]
        item_idx = item_mapping[row['movieId']]
        rating = row['rating']
        
        # User -> Item edge
        edge_index.append([user_idx, item_idx])
        edge_attr.append(rating)
        
        # Item -> User edge (bidirectional)
        edge_index.append([item_idx, user_idx])
        edge_attr.append(rating)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    
    # Create node features
    num_nodes = len(user_mapping) + len(item_mapping)
    
    # Initialize node features with embedding_dim dimensions to match model expectation
    # We'll use random initialization with a small scale, and encode user/item type 
    # in the first two dimensions
    x = torch.zeros((num_nodes, embedding_dim), dtype=torch.float)
    
    # Initialize with small random values
    x = torch.randn((num_nodes, embedding_dim), dtype=torch.float) * 0.01
    
    # Keep the user/item type encoding in first two dimensions
    x[:len(user_mapping), 0] = 1.0  # User nodes
    x[len(user_mapping):, 1] = 1.0  # Item nodes
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Add user and item mappings as additional attributes
    data.user_mapping = user_mapping
    data.item_mapping = item_mapping
    data.num_users = len(user_mapping)
    data.num_items = len(item_mapping)
    
    return data


# Function to train the model
def train_temporal_gnn(model, graph_data_list, num_epochs=10, learning_rate=0.001):
    """
    Train the Temporal GNN model on a sequence of graphs.
    
    Args:
        model: TemporalGNN model
        graph_data_list: List of PyTorch Geometric Data objects
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
    
    Returns:
        Trained model and loss history
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    # Train for each time window sequentially
    for window_idx, graph_data in enumerate(graph_data_list):
        print(f"Training on time window {window_idx+1}/{len(graph_data_list)}")
        
        # Extract data for this window
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        
        # Create training examples from this graph
        # Each edge (user->item) becomes a training example
        user_indices = []
        item_indices = []
        ratings = []
        
        for i in range(0, edge_index.shape[1], 2):  # Skip item->user edges
            user_idx = edge_index[0, i].item()
            # Important change: item_idx is now relative to the current graph, not the global item index
            item_idx = edge_index[1, i].item() - graph_data.num_users  
            
            # Skip if the item_idx is negative (shouldn't happen, but just in case)
            if item_idx < 0:
                continue
                
            rating = edge_attr[i].item()
            
            user_indices.append(user_idx)
            item_indices.append(item_idx)
            ratings.append(rating)
        
        user_indices = torch.tensor(user_indices, dtype=torch.long)
        item_indices = torch.tensor(item_indices, dtype=torch.long)
        ratings = torch.tensor(ratings, dtype=torch.float) / 5.0  # Normalize ratings to [0,1]
        
        # Training loop for this time window
        window_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            # Modified to make sure we're not going out of bounds
            outputs = model(x, edge_index, user_indices, item_indices)
            
            # Calculate loss
            loss = criterion(outputs, ratings)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            window_losses.append(loss.item())
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        loss_history.append(window_losses)
    
    return model, loss_history


# Function to visualize user embedding evolution
def visualize_user_embedding_evolution(model, graph_data_list, user_ids=None, method='pca'):
    """
    Visualize how user embeddings evolve over time.
    
    Args:
        model: Trained TemporalGNN model
        graph_data_list: List of graphs for different time windows
        user_ids: List of user IDs to visualize (if None, select a few random users)
        method: Dimensionality reduction method ('pca' or 'tsne')
    """
    model.eval()
    
    # If no specific users, select a few random ones to track
    if user_ids is None:
        # Get users that appear in all time windows
        common_users = set(graph_data_list[0].user_mapping.keys())
        for graph_data in graph_data_list[1:]:
            common_users &= set(graph_data.user_mapping.keys())
        
        # Select up to 5 random users
        if len(common_users) > 5:
            user_ids = np.random.choice(list(common_users), 5, replace=False)
        else:
            user_ids = list(common_users)
    
    # Collect embeddings for each user across time windows
    user_embeddings = defaultdict(list)
    
    for window_idx, graph_data in enumerate(graph_data_list):
        x = graph_data.x
        edge_index = graph_data.edge_index
        
        # Get user mappings for this window
        user_mapping = graph_data.user_mapping
        
        # Extract embeddings for selected users
        for user_id in user_ids:
            if user_id in user_mapping:
                user_idx = user_mapping[user_id]
                embedding = model.get_user_embedding(x, edge_index, user_idx)
                user_embeddings[user_id].append(embedding)
    
    # Perform dimensionality reduction for visualization
    # Collect all embeddings
    all_embeddings = []
    for user_id, embeddings in user_embeddings.items():
        all_embeddings.extend(embeddings)
    
    all_embeddings = np.array(all_embeddings)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # Split back by user
    idx = 0
    reduced_user_embeddings = {}
    for user_id, embeddings in user_embeddings.items():
        reduced_user_embeddings[user_id] = reduced_embeddings[idx:idx+len(embeddings)]
        idx += len(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    for user_id, embeddings in reduced_user_embeddings.items():
        x_coords = embeddings[:, 0]
        y_coords = embeddings[:, 1]
        
        plt.plot(x_coords, y_coords, 'o-', label=f'User {user_id}')
        
        # Highlight start and end points
        plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o')
        plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='x')
    
    plt.title(f'User Embedding Evolution Over Time ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'user_embedding_evolution_{method}.png')
    plt.show()
    
    return reduced_user_embeddings


# Function to identify users with the most significant preference changes
def identify_preference_shifts(model, graph_data_list):
    """
    Identify users whose preferences have shifted the most across time windows.
    
    Args:
        model: Trained TemporalGNN model
        graph_data_list: List of graphs for different time windows
        
    Returns:
        Dictionary mapping user IDs to shift magnitudes
    """
    model.eval()
    
    # Get users that appear in both first and last time windows
    first_window_users = set(graph_data_list[0].user_mapping.keys())
    last_window_users = set(graph_data_list[-1].user_mapping.keys())
    common_users = first_window_users & last_window_users
    
    # Calculate embedding shift for each user
    user_shifts = {}
    
    for user_id in common_users:
        # Get user's embedding in first time window
        first_graph = graph_data_list[0]
        user_idx_first = first_graph.user_mapping[user_id]
        first_embedding = model.get_user_embedding(
            first_graph.x, first_graph.edge_index, user_idx_first)
        
        # Get user's embedding in last time window
        last_graph = graph_data_list[-1]
        user_idx_last = last_graph.user_mapping[user_id]
        last_embedding = model.get_user_embedding(
            last_graph.x, last_graph.edge_index, user_idx_last)
        
        # Calculate Euclidean distance between embeddings
        shift = np.sqrt(np.sum((last_embedding - first_embedding) ** 2))
        user_shifts[user_id] = shift
    
    # Sort users by shift magnitude
    sorted_users = sorted(user_shifts.items(), key=lambda x: x[1], reverse=True)
    
    return dict(sorted_users)


# Function to prepare the MovieLens dataset for temporal GNN
def prepare_movielens_for_gnn(file_path="ml-1m/ratings.dat", time_windows=3):
    """
    Prepare MovieLens dataset for temporal GNN.
    
    Args:
        file_path: Path to the MovieLens ratings file
        time_windows: Number of time windows to divide the data into
        
    Returns:
        List of graph data objects for each time window
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please download the MovieLens 1M dataset from: https://grouplens.org/datasets/movielens/1m/")
        print("Using sample data instead...")
        
        # Create sample data
        sample_ratings = [
            {'userId': 1, 'movieId': 101, 'rating': 5.0, 'timestamp': 1000000000},
            {'userId': 1, 'movieId': 102, 'rating': 3.0, 'timestamp': 1000100000},
            {'userId': 1, 'movieId': 103, 'rating': 4.0, 'timestamp': 1000500000},
            {'userId': 2, 'movieId': 101, 'rating': 2.0, 'timestamp': 1000200000},
            {'userId': 2, 'movieId': 103, 'rating': 4.5, 'timestamp': 1000300000},
            {'userId': 2, 'movieId': 104, 'rating': 5.0, 'timestamp': 1000400000},
            {'userId': 3, 'movieId': 102, 'rating': 3.5, 'timestamp': 1000250000},
            {'userId': 3, 'movieId': 103, 'rating': 4.0, 'timestamp': 1000350000},
            {'userId': 3, 'movieId': 104, 'rating': 4.5, 'timestamp': 1000550000},
            {'userId': 1, 'movieId': 104, 'rating': 4.0, 'timestamp': 1001000000},
            {'userId': 1, 'movieId': 105, 'rating': 3.5, 'timestamp': 1002000000},
            {'userId': 2, 'movieId': 105, 'rating': 4.0, 'timestamp': 1002500000},
            {'userId': 3, 'movieId': 105, 'rating': 5.0, 'timestamp': 1003000000},
            # Add more entries for multiple time windows
            {'userId': 1, 'movieId': 106, 'rating': 4.5, 'timestamp': 1004000000},
            {'userId': 1, 'movieId': 107, 'rating': 3.0, 'timestamp': 1005000000},
            {'userId': 2, 'movieId': 106, 'rating': 3.5, 'timestamp': 1004500000},
            {'userId': 2, 'movieId': 107, 'rating': 4.0, 'timestamp': 1005500000},
            {'userId': 3, 'movieId': 106, 'rating': 4.0, 'timestamp': 1004200000},
            {'userId': 3, 'movieId': 107, 'rating': 4.5, 'timestamp': 1005200000}
        ]
        
        ratings_df = pd.DataFrame(sample_ratings)
    else:
        # Load the MovieLens 1M dataset (ratings.dat)
        ratings_df = pd.read_csv(file_path, 
                                sep='::', 
                                header=None, 
                                names=['userId', 'movieId', 'rating', 'timestamp'],
                                engine='python')
    
    # Add human-readable dates
    ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    
    # Convert ratings to float
    ratings_df['rating'] = ratings_df['rating'].astype(float)
    
    # Create temporal graphs
    graph_data_list = create_temporal_graph(ratings_df, time_windows=time_windows)
    
    return graph_data_list

def cluster_users(model, graph_data_list, n_clusters=5):
    """Cluster users based on their final embeddings"""
    from sklearn.cluster import KMeans
    
    # Get final time window
    final_graph = graph_data_list[-1]
    x = final_graph.x
    edge_index = final_graph.edge_index
    
    # Extract user embeddings
    user_embeddings = []
    user_ids = []
    
    for user_id, user_idx in final_graph.user_mapping.items():
        embedding = model.get_user_embedding(x, edge_index, user_idx)
        user_embeddings.append(embedding)
        user_ids.append(user_id)
    
    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(user_embeddings)
    
    # Create a scatter plot with clusters
    reduced = PCA(n_components=2).fit_transform(user_embeddings)
    
    plt.figure(figsize=(12, 8))
    for cluster_id in range(n_clusters):
        cluster_points = reduced[clusters == cluster_id]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            label=f'Cluster {cluster_id}',
            alpha=0.7
        )
    
    plt.title('User Embedding Clusters')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('user_clusters.png')
    
    return user_ids, clusters


def create_preference_heatmap(model, graph_data_list, user_clusters, ratings_df):
    """Generate a heatmap showing preference patterns by cluster"""
    import seaborn as sns
    
    # Get most popular items from the original ratings DataFrame
    item_counts = ratings_df['movieId'].value_counts().to_dict()
    
    # Get top 10 most popular items
    top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_item_ids = [item[0] for item in top_items]
    
    # Create preference matrix for each cluster
    user_ids, clusters = user_clusters  # Unpack the tuple
    n_clusters = len(np.unique(clusters))
    cluster_preferences = np.zeros((n_clusters, len(top_item_ids)))
    
    # Fill preference matrix using the original ratings DataFrame
    for i, user_id in enumerate(user_ids):
        cluster_id = clusters[i]
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        for j, item_id in enumerate(top_item_ids):
            # Look up user's rating for this item if available
            item_ratings = user_ratings[user_ratings['movieId'] == item_id]
            
            if len(item_ratings) > 0:
                # Average if there are multiple ratings
                cluster_preferences[cluster_id, j] += item_ratings['rating'].mean()
    
    # Normalize by cluster size
    for cluster_id in range(cluster_preferences.shape[0]):
        cluster_size = np.sum(clusters == cluster_id)
        if cluster_size > 0:
            cluster_preferences[cluster_id, :] /= cluster_size
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cluster_preferences, 
        annot=True, 
        cmap='YlGnBu',
        xticklabels=[f'Item {item_id}' for item_id in top_item_ids],
        yticklabels=[f'Cluster {i}' for i in range(n_clusters)]
    )
    plt.title('Preference Patterns by User Cluster')
    plt.tight_layout()
    plt.savefig('cluster_preferences.png')


def evaluate_model(model, test_graph_data, ratings_df):
    """Evaluate model on test data and generate performance metrics"""
    from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
    
    # Extract data
    x = test_graph_data.x
    edge_index = test_graph_data.edge_index
    edge_attr = test_graph_data.edge_attr
    
    # Create test examples
    user_indices = []
    item_indices = []
    ratings = []
    
    # Get original user and item IDs for reporting
    user_id_mapping = test_graph_data.user_mapping
    item_id_mapping = test_graph_data.item_mapping
    
    for i in range(0, edge_index.shape[1], 2):  # Skip item->user edges
        user_idx = edge_index[0, i].item()
        item_idx = edge_index[1, i].item() - test_graph_data.num_users
        
        # Skip if the item_idx is negative
        if item_idx < 0:
            continue
            
        rating = edge_attr[i].item()
        
        user_indices.append(user_idx)
        item_indices.append(item_idx)
        ratings.append(rating)
    
    user_indices = torch.tensor(user_indices, dtype=torch.long)
    item_indices = torch.tensor(item_indices, dtype=torch.long)
    
    # Normalize ratings to [0,1] for the model
    normalized_ratings = torch.tensor(ratings, dtype=torch.float) / 5.0
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(x, edge_index, user_indices, item_indices)
    
    # Convert back to original scale
    predictions_numpy = predictions.numpy() * 5.0
    true_ratings = normalized_ratings.numpy() * 5.0
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(true_ratings, predictions_numpy))
    
    # For confusion matrix, round predictions to nearest integer
    rounded_preds = np.round(predictions_numpy).astype(int)
    rounded_true = np.round(true_ratings).astype(int)
    
    # Ensure values are within rating range (1-5)
    rounded_preds = np.clip(rounded_preds, 1, 5)
    rounded_true = np.clip(rounded_true, 1, 5)
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(rounded_true, rounded_preds)
    conf_matrix = confusion_matrix(rounded_true, rounded_preds, labels=range(1, 6))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(1, 6), 
                yticklabels=range(1, 6))
    plt.title('Rating Prediction Confusion Matrix')
    plt.xlabel('Predicted Rating')
    plt.ylabel('True Rating')
    plt.savefig('confusion_matrix.png')
    
    # Print metrics
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return rmse, accuracy, conf_matrix


# Main function to run the entire pipeline
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    embedding_dim = 64
    hidden_dim = 128
    num_epochs = 10
    learning_rate = 0.001
    time_windows = 3
    
    # Load the original ratings data
    file_path = "ml-1m/ratings.dat"
    if os.path.exists(file_path):
        # Load the MovieLens 1M dataset
        ratings_df = pd.read_csv(file_path, 
                            sep='::', 
                            header=None, 
                            names=['userId', 'movieId', 'rating', 'timestamp'],
                            engine='python')
        
        # Add human-readable dates
        ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
        
        # Convert ratings to float
        ratings_df['rating'] = ratings_df['rating'].astype(float)
    else:
        print(f"File not found: {file_path}")
        print("Using sample data instead...")
        
        # Create sample data
        sample_ratings = [
            {'userId': 1, 'movieId': 101, 'rating': 5.0, 'timestamp': 1000000000},
            {'userId': 1, 'movieId': 102, 'rating': 3.0, 'timestamp': 1000100000},
            # ... add all your sample data points here ...
            {'userId': 3, 'movieId': 107, 'rating': 4.5, 'timestamp': 1005200000}
        ]
        
        ratings_df = pd.DataFrame(sample_ratings)
        ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    
    # Prepare data - create graph data list from ratings_df
    graph_data_list = create_temporal_graph(ratings_df, time_windows=time_windows)
    
    if not graph_data_list:
        print("No data available. Exiting...")
        return
    
    # Get number of users and items from the first graph
    first_graph = graph_data_list[0]
    num_nodes = first_graph.num_nodes
    num_users = first_graph.num_users
    num_items = first_graph.num_items
    
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    
    # Initialize model
    model = TemporalGNN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )
    
    # Train model
    trained_model, loss_history = train_temporal_gnn(
        model=model,
        graph_data_list=graph_data_list,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    # Save the model
    torch.save(trained_model.state_dict(), 'temporal_gnn_model.pth')
    
    # Visualize embedding evolution
    visualize_user_embedding_evolution(
        model=trained_model,
        graph_data_list=graph_data_list,
        method='pca'
    )
    
    # Identify preference shifts
    user_shifts = identify_preference_shifts(
        model=trained_model,
        graph_data_list=graph_data_list
    )

    # Cluster users
    user_ids, clusters = cluster_users(trained_model, graph_data_list)

    # Create preference heatmap - passing ratings_df
    create_preference_heatmap(trained_model, graph_data_list, (user_ids, clusters), ratings_df)

    # Evaluate model performance - passing ratings_df
    rmse, accuracy, conf_matrix = evaluate_model(trained_model, graph_data_list[-1], ratings_df)
    
    # Print users with most significant shifts
    print("\nUsers with the most significant preference shifts:")
    for user_id, shift in list(user_shifts.items())[:5]:
        print(f"User {user_id}: Shift magnitude = {shift:.4f}")
    
    print("\nUsers with the least significant preference shifts:")
    for user_id, shift in list(user_shifts.items())[-5:]:
        print(f"User {user_id}: Shift magnitude = {shift:.4f}")
    
    # Analyze user neighborhoods
    print("\nAnalyzing user neighborhoods...")
    
    # Create a visualization of the final time window graph
    last_graph = graph_data_list[-1]
    
    # Create NetworkX graph for visualization
    G = nx.Graph()
    
    # Add nodes
    for user_id, user_idx in last_graph.user_mapping.items():
        G.add_node(f"U{user_id}", type="user")
    
    for item_id, item_idx in last_graph.item_mapping.items():
        G.add_node(f"I{item_id}", type="item")
    
    # Add edges
    for i in range(0, last_graph.edge_index.shape[1], 2):  # Skip item->user edges
        user_idx = last_graph.edge_index[0, i].item()
        item_idx = last_graph.edge_index[1, i].item()
        rating = last_graph.edge_attr[i].item()
        
        # Get original IDs
        user_id = [k for k, v in last_graph.user_mapping.items() if v == user_idx][0]
        item_id = [k for k, v in last_graph.item_mapping.items() if v == item_idx][0]
        
        G.add_edge(f"U{user_id}", f"I{item_id}", rating=rating)
    
    # Calculate layout
    pos = nx.spring_layout(G, seed=42)
    
    # Plot graph
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    user_nodes = [n for n in G.nodes if n.startswith('U')]
    item_nodes = [n for n in G.nodes if n.startswith('I')]
    
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='blue', node_size=200, alpha=0.8, label='Users')
    nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, node_color='red', node_size=100, alpha=0.8, label='Items')
    
    # Draw edges
    edge_colors = [G[u][v]['rating'] / 5.0 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5, edge_color=edge_colors, edge_cmap=plt.cm.YlGnBu)
    
    # Draw labels
    user_labels = {n: n for n in user_nodes[:10]}  # Only show first 10 user labels to avoid clutter
    nx.draw_networkx_labels(G, pos, labels=user_labels, font_size=10)
    
    plt.title('User-Item Interaction Graph (Final Time Window)')
    plt.legend()
    plt.axis('off')
    plt.savefig('user_item_graph.png')
    
    print("\nDone!")


if __name__ == "__main__":
    main()