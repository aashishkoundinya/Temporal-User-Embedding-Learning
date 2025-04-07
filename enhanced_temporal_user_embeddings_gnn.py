import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx
import os
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report


# Define a more powerful hybrid GNN model
class EnhancedTemporalGNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, hidden_dim=256, num_heads=4, dropout=0.3):
        super(EnhancedTemporalGNN, self).__init__()
        
        # Initial embeddings with larger dimensions
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize with Xavier/Glorot
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Mix of GAT and GCN layers for better representational power
        self.gat1 = GATConv(embedding_dim, hidden_dim // num_heads, heads=num_heads)
        self.gcn1 = GCNConv(embedding_dim, hidden_dim)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Deeper prediction network
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln4 = nn.LayerNorm(hidden_dim // 2)
        
        # Final layer with larger hidden dim before output
        self.fc3 = nn.Linear(hidden_dim // 2, 128)
        self.ln5 = nn.LayerNorm(128)
        
        # Output layer for 5-class classification
        self.output = nn.Linear(128, 5)
        
        # Dropout
        self.dropout = dropout
        
        # Initialize all linear layers
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch_user_indices, batch_item_indices):
        # Parallel GAT and GCN paths
        x_gat1 = F.elu(self.gat1(x, edge_index))
        x_gcn1 = F.elu(self.gcn1(x, edge_index))
        
        # Combine and normalize
        x = x_gat1 + x_gcn1
        x = self.ln1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer with residual connection
        x_gat2 = F.elu(self.gat2(x, edge_index))
        x_gcn2 = F.elu(self.gcn2(x, edge_index))
        x = x_gat2 + x_gcn2 + x  # Residual connection
        x = self.ln2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Get user and item embeddings
        user_embeddings = x[batch_user_indices]
        num_users = len(torch.unique(batch_user_indices))
        item_embeddings = x[batch_item_indices + num_users]
        
        # Concatenate user and item embeddings
        combined = torch.cat([user_embeddings, item_embeddings], dim=1)
        
        # Fully connected layers with normalization
        x = F.elu(self.fc1(combined))
        x = self.ln3(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.elu(self.fc2(x))
        x = self.ln4(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.elu(self.fc3(x))
        x = self.ln5(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        logits = self.output(x)
        
        return logits
    
    def predict_rating(self, x, edge_index, batch_user_indices, batch_item_indices):
        # Forward pass
        logits = self.forward(x, edge_index, batch_user_indices, batch_item_indices)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get predicted class (add 1 for 1-5 rating scale)
        predictions = torch.argmax(logits, dim=1) + 1
        
        return predictions, probs
    
    def get_user_embedding(self, x, edge_index, user_idx):
        # Forward pass through GNN layers
        x_gat1 = F.elu(self.gat1(x, edge_index))
        x_gcn1 = F.elu(self.gcn1(x, edge_index))
        x = x_gat1 + x_gcn1
        x = self.ln1(x)
        
        x_gat2 = F.elu(self.gat2(x, edge_index))
        x_gcn2 = F.elu(self.gcn2(x, edge_index))
        x = x_gat2 + x_gcn2 + x
        x = self.ln2(x)
        
        # Return user embedding
        return x[user_idx].detach().cpu().numpy()


# Enhanced function to create temporal graph with improved features
def create_balanced_temporal_graph(ratings_df, time_windows=3, sample_size=None, balance_ratings=True):
    """
    Create temporal graphs with class balancing to handle rating imbalance.
    
    Args:
        ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
        time_windows: Number of time windows to divide the data into
        sample_size: Optional limit on the number of users to include
        balance_ratings: Whether to balance rating classes
    
    Returns:
        List of graph data objects for each time window
    """
    # Sort ratings by timestamp
    ratings_df = ratings_df.sort_values('timestamp')
    
    # Sample users if requested
    if sample_size is not None:
        sampled_users = np.random.choice(ratings_df['userId'].unique(), 
                                         size=min(sample_size, len(ratings_df['userId'].unique())), 
                                         replace=False)
        ratings_df = ratings_df[ratings_df['userId'].isin(sampled_users)]
    
    # Balance rating classes if requested
    if balance_ratings:
        # Count original distribution
        rating_counts = ratings_df['rating'].value_counts()
        print("Original rating distribution:")
        for rating, count in sorted(rating_counts.items()):
            print(f"Rating {rating}: {count} ({count/len(ratings_df)*100:.1f}%)")
        
        # Calculate target count - use sqrt of counts to balance but not equalize completely
        # This preserves some of the natural distribution while reducing extreme imbalance
        min_count = int(np.sqrt(rating_counts.min()) * np.sqrt(rating_counts.max()))
        
        # Sample from each rating class
        balanced_dfs = []
        for rating in sorted(ratings_df['rating'].unique()):
            rating_df = ratings_df[ratings_df['rating'] == rating]
            
            # If we have more than target, downsample
            if len(rating_df) > min_count:
                # For 1, 2, 5 stars, use min_count * 2 to give them more weight
                target = min_count * 2 if rating in [1.0, 2.0, 5.0] else min_count
                
                # Ensure we don't ask for more than we have
                target = min(len(rating_df), target)
                
                # Sample without replacement
                balanced_dfs.append(rating_df.sample(target, random_state=42))
            else:
                # If less than target, use all and oversample
                # Only oversample for rare ratings (1, 2, 5)
                if rating in [1.0, 2.0, 5.0]:
                    # Oversample with replacement to reach target count
                    oversampled = rating_df.sample(min_count, replace=True, random_state=42)
                    balanced_dfs.append(oversampled)
                else:
                    balanced_dfs.append(rating_df)
        
        # Combine balanced samples
        balanced_df = pd.concat(balanced_dfs)
        
        # Count new distribution
        new_rating_counts = balanced_df['rating'].value_counts()
        print("\nBalanced rating distribution:")
        for rating, count in sorted(new_rating_counts.items()):
            print(f"Rating {rating}: {count} ({count/len(balanced_df)*100:.1f}%)")
        
        ratings_df = balanced_df
    
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
        
        # Create graph with improved feature encoding
        graph_data = create_enhanced_graph(window_df)
        graph_data_list.append(graph_data)
        
        print(f"Time window {i+1}: {len(window_df)} interactions, "
              f"{graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    return graph_data_list


# Enhanced graph creation with richer features
def create_enhanced_graph(interactions_df, embedding_dim=128):
    """
    Create a graph from user-item interactions with enhanced features.
    """
    # Get unique users and items
    unique_users = interactions_df['userId'].unique()
    unique_items = interactions_df['movieId'].unique()
    
    # Create mapping from original IDs to consecutive indices
    user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_mapping = {item_id: idx + len(user_mapping) for idx, item_id in enumerate(unique_items)}
    
    # Create edges with rating-based weights
    edge_index = []
    edge_attr = []
    edge_weights = []  # Normalized edge weights based on ratings
    
    for _, row in interactions_df.iterrows():
        user_idx = user_mapping[row['userId']]
        item_idx = item_mapping[row['movieId']]
        rating = row['rating']
        
        # Normalize rating to [0,1] for edge weight
        weight = (rating - 1) / 4.0
        
        # User -> Item edge
        edge_index.append([user_idx, item_idx])
        edge_attr.append(rating)
        edge_weights.append(weight)
        
        # Item -> User edge (bidirectional)
        edge_index.append([item_idx, user_idx])
        edge_attr.append(rating)
        edge_weights.append(weight)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    # Create node features with richer encoding
    num_nodes = len(user_mapping) + len(item_mapping)
    
    # Initialize features
    x = torch.zeros((num_nodes, embedding_dim), dtype=torch.float)
    
    # One-hot encoding for node type (user vs item)
    x[:len(user_mapping), 0] = 1.0  # User nodes
    x[len(user_mapping):, 1] = 1.0  # Item nodes
    
    # Compute user features
    user_features = {}
    for user_id in unique_users:
        user_data = interactions_df[interactions_df['userId'] == user_id]
        
        # Basic stats
        avg_rating = user_data['rating'].mean()
        std_rating = user_data['rating'].std() if len(user_data) > 1 else 0
        rating_count = len(user_data)
        
        # Rating distribution
        rating_dist = [0] * 5  # Counts for ratings 1-5
        for r in user_data['rating']:
            rating_dist[int(r)-1] += 1
        
        # Normalize distribution
        if rating_count > 0:
            rating_dist = [count / rating_count for count in rating_dist]
        
        # Store features
        user_features[user_id] = {
            'avg_rating': (avg_rating - 3) / 2,  # Center around 0
            'std_rating': std_rating / 2,  # Scale to reasonable range
            'log_count': np.log1p(rating_count) / 5,  # Log-scale count
            'rating_dist': rating_dist  # Rating distribution
        }
    
    # Compute item features
    item_features = {}
    for item_id in unique_items:
        item_data = interactions_df[interactions_df['movieId'] == item_id]
        
        # Basic stats
        avg_rating = item_data['rating'].mean()
        std_rating = item_data['rating'].std() if len(item_data) > 1 else 0
        rating_count = len(item_data)
        
        # Rating distribution
        rating_dist = [0] * 5  # Counts for ratings 1-5
        for r in item_data['rating']:
            rating_dist[int(r)-1] += 1
        
        # Normalize distribution
        if rating_count > 0:
            rating_dist = [count / rating_count for count in rating_dist]
        
        # Store features
        item_features[item_id] = {
            'avg_rating': (avg_rating - 3) / 2,
            'std_rating': std_rating / 2,
            'popularity': np.log1p(rating_count) / 5,
            'rating_dist': rating_dist
        }
    
    # Add features to node embeddings
    for user_id, idx in user_mapping.items():
        features = user_features[user_id]
        
        # Add basic features
        x[idx, 2] = features['avg_rating']
        x[idx, 3] = features['std_rating']
        x[idx, 4] = features['log_count']
        
        # Add rating distribution
        x[idx, 5:10] = torch.tensor(features['rating_dist'], dtype=torch.float)
    
    for item_id, idx in item_mapping.items():
        features = item_features[item_id]
        
        # Add basic features
        x[idx, 2] = features['avg_rating']
        x[idx, 3] = features['std_rating']
        x[idx, 4] = features['popularity']
        
        # Add rating distribution
        x[idx, 5:10] = torch.tensor(features['rating_dist'], dtype=torch.float)
    
    # Add random noise to remaining features for better initialization
    if embedding_dim > 10:
        x[:, 10:] = torch.randn((num_nodes, embedding_dim - 10), dtype=torch.float) * 0.01
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weights)
    
    # Add metadata
    data.user_mapping = user_mapping
    data.item_mapping = item_mapping
    data.num_users = len(user_mapping)
    data.num_items = len(item_mapping)
    
    return data


# Improved training function with focal loss and class balancing
def train_enhanced_model(model, graph_data_list, num_epochs=30, learning_rate=0.001, 
                       weight_decay=1e-5, focal_gamma=2.0, validation_split=0.1,
                       patience=7, batch_size=2048):
    """
    Train the model with focal loss and advanced techniques.
    """
    # Define optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler with warm-up
    def lr_lambda(epoch):
        # Warm-up for 3 epochs, then cosine decay
        if epoch < 3:
            return epoch / 3
        else:
            return 0.5 * (1 + np.cos((epoch - 3) / (num_epochs - 3) * np.pi))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # History for tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    # Define focal loss with class balancing
    def focal_loss(logits, targets, gamma=focal_gamma, alpha=None):
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=5).float()
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Calculate focal loss
        pt = torch.sum(targets_one_hot * probs, dim=1)
        focal_weight = (1 - pt) ** gamma
        
        # Apply alpha weighting if provided
        if alpha is not None:
            alpha_weight = torch.sum(targets_one_hot * alpha, dim=1)
            focal_weight = alpha_weight * focal_weight
        
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply focal weighting
        loss = focal_weight * ce_loss
        
        return loss.mean()
    
    # Train on each time window
    for window_idx, graph_data in enumerate(graph_data_list):
        print(f"\nTraining on time window {window_idx+1}/{len(graph_data_list)}")
        
        # Extract data
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        edge_weight = graph_data.edge_weight if hasattr(graph_data, 'edge_weight') else None
        
        # Create training examples
        user_indices = []
        item_indices = []
        ratings = []
        
        for i in range(0, edge_index.shape[1], 2):  # Skip item->user edges
            user_idx = edge_index[0, i].item()
            item_idx = edge_index[1, i].item() - graph_data.num_users
            
            if item_idx < 0:
                continue
                
            rating = int(edge_attr[i].item())
            
            user_indices.append(user_idx)
            item_indices.append(item_idx)
            ratings.append(rating - 1)  # Convert to 0-indexed
        
        # Convert to tensors
        user_indices = torch.tensor(user_indices, dtype=torch.long)
        item_indices = torch.tensor(item_indices, dtype=torch.long)
        ratings = torch.tensor(ratings, dtype=torch.long)
        
        # Calculate class weights for focal loss
        class_counts = torch.bincount(ratings, minlength=5)
        class_weights = 1.0 / (class_counts.float() + 1e-8)
        class_weights = class_weights / class_weights.sum() * 5  # Normalize
        
        # Increase weights for underrepresented classes
        class_weights[0] *= 1.5  # Rating 1
        class_weights[1] *= 1.5  # Rating 2
        class_weights[4] *= 1.5  # Rating 5
        
        print(f"Class weights: {class_weights.numpy()}")
        
        # Split into train and validation
        indices = torch.randperm(len(ratings))
        val_size = int(validation_split * len(ratings))
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Training loop with batching and early stopping
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_loss = 0
            correct = 0
            
            # Process in batches
            permutation = torch.randperm(len(train_indices))
            num_batches = (len(train_indices) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
                batch_perm = permutation[start_idx:end_idx]
                batch_indices = train_indices[batch_perm]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(x, edge_index,
                              user_indices[batch_indices],
                              item_indices[batch_indices])
                
                # Calculate loss with focal loss
                loss = focal_loss(outputs, ratings[batch_indices], gamma=focal_gamma, alpha=class_weights)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item() * len(batch_indices)
                
                # Calculate accuracy
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == ratings[batch_indices]).sum().item()
            
            # Calculate average training metrics
            avg_train_loss = total_loss / len(train_indices)
            train_accuracy = correct / len(train_indices)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            
            with torch.no_grad():
                # Process in batches
                num_val_batches = (len(val_indices) + batch_size - 1) // batch_size
                
                for batch_idx in range(num_val_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(val_indices))
                    batch_indices = val_indices[start_idx:end_idx]
                    
                    # Forward pass
                    outputs = model(x, edge_index,
                                  user_indices[batch_indices],
                                  item_indices[batch_indices])
                    
                    # Calculate loss
                    batch_loss = focal_loss(outputs, ratings[batch_indices], gamma=focal_gamma, alpha=class_weights)
                    val_loss += batch_loss.item() * len(batch_indices)
                    
                    # Calculate accuracy
                    pred = torch.argmax(outputs, dim=1)
                    val_correct += (pred == ratings[batch_indices]).sum().item()
            
            # Calculate average validation metrics
            avg_val_loss = val_loss / len(val_indices)
            val_accuracy = val_correct / len(val_indices)
            
            # Update learning rate
            scheduler.step()
            
            # Save metrics
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Train Acc: {train_accuracy:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}")
            
            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"  Early stopping after {epoch+1} epochs")
                break
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'][-len(history['train_loss']):], label='Train Loss')
        plt.plot(history['val_loss'][-len(history['val_loss']):], label='Val Loss')
        plt.title(f'Loss Curves - Window {window_idx+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'][-len(history['train_accuracy']):], label='Train Accuracy')
        plt.plot(history['val_accuracy'][-len(history['val_accuracy']):], label='Val Accuracy')
        plt.title(f'Accuracy Curves - Window {window_idx+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_curves_window_{window_idx+1}_enhanced.png')
        plt.close()
    
    return model, history


# Enhanced evaluation with better metrics and visualizations
def evaluate_enhanced_model(model, test_graph_data, ratings_df=None):
    """
    Comprehensive evaluation of the model with detailed metrics.
    """
    # Extract data
    x = test_graph_data.x
    edge_index = test_graph_data.edge_index
    edge_attr = test_graph_data.edge_attr
    
    # Create test examples
    user_indices = []
    item_indices = []
    ratings = []
    original_user_ids = []
    original_item_ids = []
    
    # Get mappings
    user_id_mapping = test_graph_data.user_mapping
    item_id_mapping = test_graph_data.item_mapping
    
    # Reverse mappings for reporting
    reverse_user_mapping = {idx: user_id for user_id, idx in user_id_mapping.items()}
    reverse_item_mapping = {idx: item_id for item_id, idx in item_id_mapping.items()}
    
    for i in range(0, edge_index.shape[1], 2):  # Skip item->user edges
        user_idx = edge_index[0, i].item()
        item_idx = edge_index[1, i].item() - test_graph_data.num_users
        
        # Skip if item_idx is negative
        if item_idx < 0:
            continue
            
        rating = edge_attr[i].item()
        
        user_indices.append(user_idx)
        item_indices.append(item_idx)
        ratings.append(rating)
        
        # Store original IDs for detailed analysis
        if user_idx in reverse_user_mapping:
            original_user_ids.append(reverse_user_mapping[user_idx])
        if item_idx + test_graph_data.num_users in reverse_item_mapping:
            original_item_ids.append(reverse_item_mapping[item_idx + test_graph_data.num_users])
    
    # Convert to tensors
    user_indices = torch.tensor(user_indices, dtype=torch.long)
    item_indices = torch.tensor(item_indices, dtype=torch.long)
    true_ratings = torch.tensor(ratings, dtype=torch.long)
    true_ratings_class = true_ratings - 1  # Convert to 0-indexed classes
    
    # Batch size for evaluation
    batch_size = 4096
    num_batches = (len(user_indices) + batch_size - 1) // batch_size
    
    # Make predictions
    model.eval()
    all_logits = []
    all_preds = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(user_indices))
            
            # Get batch indices
            batch_user_indices = user_indices[start_idx:end_idx]
            batch_item_indices = item_indices[start_idx:end_idx]
            
            # Forward pass
            logits = model(x, edge_index, batch_user_indices, batch_item_indices)
            
            # Convert to probabilities and predictions
            batch_preds = torch.argmax(logits, dim=1) + 1  # Convert to 1-5 scale
            
            # Store results
            all_logits.append(logits)
            all_preds.append(batch_preds)
    
    # Combine all predictions
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_logits = torch.cat(all_logits)
    
    # Convert to numpy for metrics calculation
    true_ratings_np = true_ratings.cpu().numpy()
    
    # Calculate accuracy metrics
    rmse = np.sqrt(mean_squared_error(true_ratings_np, all_preds))
    accuracy = accuracy_score(true_ratings_np, all_preds)
    conf_matrix = confusion_matrix(true_ratings_np, all_preds, labels=range(1, 6))
    
    # Detailed classification report
    class_report = classification_report(
        true_ratings_np, 
        all_preds,
        labels=range(1, 6),
        target_names=[f'Rating {i}' for i in range(1, 6)],
        zero_division=0
    )
    
    # Calculate per-class metrics manually to avoid warnings
    true_positives = np.diag(conf_matrix)
    false_positives = np.sum(conf_matrix, axis=0) - true_positives
    false_negatives = np.sum(conf_matrix, axis=1) - true_positives
    
    # Handle division by zero
    precision = np.divide(
        true_positives, 
        true_positives + false_positives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(true_positives + false_positives) != 0
    )
    
    recall = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(true_positives + false_negatives) != 0
    )
    
    # Create enhanced confusion matrix visualization
    plt.figure(figsize=(12, 10))
    
    # Plot raw counts
    plt.subplot(2, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(1, 6), 
                yticklabels=range(1, 6))
    plt.title('Confusion Matrix (Raw Counts)', fontsize=14)
    plt.xlabel('Predicted Rating', fontsize=12)
    plt.ylabel('True Rating', fontsize=12)
    
    # Plot normalized by true labels (recall)
    row_sums = conf_matrix.sum(axis=1)
    norm_conf_matrix = np.divide(
        conf_matrix, 
        row_sums[:, np.newaxis],
        out=np.zeros_like(conf_matrix, dtype=float),
        where=row_sums[:, np.newaxis] != 0
    )
    
    plt.subplot(2, 2, 2)
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(1, 6), 
                yticklabels=range(1, 6))
    plt.title('Confusion Matrix (Normalized by True Ratings - Recall)', fontsize=14)
    plt.xlabel('Predicted Rating', fontsize=12)
    plt.ylabel('True Rating', fontsize=12)
    
    # Plot normalized by predicted labels (precision)
    col_sums = conf_matrix.sum(axis=0)
    norm_conf_matrix_pred = np.divide(
        conf_matrix, 
        col_sums[np.newaxis, :],
        out=np.zeros_like(conf_matrix, dtype=float),
        where=col_sums[np.newaxis, :] != 0
    )
    
    plt.subplot(2, 2, 3)
    sns.heatmap(norm_conf_matrix_pred, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(1, 6), 
                yticklabels=range(1, 6))
    plt.title('Confusion Matrix (Normalized by Predicted Ratings - Precision)', fontsize=14)
    plt.xlabel('Predicted Rating', fontsize=12)
    plt.ylabel('True Rating', fontsize=12)
    
    # Plot distribution comparison
    plt.subplot(2, 2, 4)
    
    true_dist = np.bincount(true_ratings_np, minlength=6)[1:]
    pred_dist = np.bincount(all_preds, minlength=6)[1:]
    
    x = np.arange(5)
    width = 0.35
    
    plt.bar(x - width/2, true_dist, width, label='True Ratings')
    plt.bar(x + width/2, pred_dist, width, label='Predicted Ratings')
    
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Rating Distribution Comparison', fontsize=14)
    plt.xticks(x, range(1, 6))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_enhanced.png', dpi=300)
    plt.close()
    
    # Print evaluation metrics
    print("\nEnhanced Model Evaluation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    # Print per-class metrics
    print("\nPrecision by class:")
    for i, p in enumerate(precision):
        print(f"Rating {i+1}: {p:.4f}")
    
    print("\nRecall by class:")
    for i, r in enumerate(recall):
        print(f"Rating {i+1}: {r:.4f}")
    
    # Analyze prediction errors
    error_analysis = pd.DataFrame({
        'user_id': original_user_ids,
        'item_id': original_item_ids,
        'true_rating': true_ratings_np,
        'predicted_rating': all_preds,
        'error': np.abs(true_ratings_np - all_preds)
    })
    
    # Identify extreme errors (difference of 3 or more)
    extreme_errors = error_analysis[error_analysis['error'] >= 3]
    if len(extreme_errors) > 0:
        print(f"\nFound {len(extreme_errors)} predictions with extreme errors (≥3):")
        print(extreme_errors.head(10))
    
    return rmse, accuracy, conf_matrix, class_report, error_analysis


# Function to visualize user embedding evolution
def visualize_user_embedding_evolution(model, graph_data_list, user_ids=None, method='pca'):
    """
    Visualize how user embeddings evolve over time with improved visualization.
    
    Args:
        model: Trained GNN model
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
        
        # Select up to 8 random users
        if len(common_users) > 8:
            user_ids = np.random.choice(list(common_users), 8, replace=False)
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
                user_embeddings[user_id].append((window_idx, embedding))
    
    # Filter out users that don't have embeddings for all windows
    user_embeddings = {k: v for k, v in user_embeddings.items() 
                     if len(v) == len(graph_data_list)}
    
    if not user_embeddings:
        print("No users appear in all time windows. Cannot visualize embedding evolution.")
        return
    
    # Collect all embeddings for dimensionality reduction
    all_embeddings = []
    for user_id, embeddings in user_embeddings.items():
        all_embeddings.extend([e[1] for e in embeddings])
    
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # Split back by user
    idx = 0
    reduced_user_embeddings = {}
    for user_id, embeddings in user_embeddings.items():
        num_windows = len(embeddings)
        reduced_user_embeddings[user_id] = reduced_embeddings[idx:idx+num_windows]
        idx += num_windows
    
    # Plot with improved visualization
    plt.figure(figsize=(15, 10))
    
    # Color map for different users
    cmap = plt.cm.get_cmap('tab10', len(reduced_user_embeddings))
    
    # Plot trajectories
    for i, (user_id, embeddings) in enumerate(reduced_user_embeddings.items()):
        x_coords = embeddings[:, 0]
        y_coords = embeddings[:, 1]
        
        # Plot trajectory with arrows to show direction
        plt.plot(x_coords, y_coords, '-', color=cmap(i), alpha=0.7, linewidth=2, label=f'User {user_id}')
        
        for t in range(len(x_coords) - 1):
            plt.annotate('', 
                        xy=(x_coords[t+1], y_coords[t+1]), 
                        xytext=(x_coords[t], y_coords[t]),
                        arrowprops=dict(arrowstyle='->', color=cmap(i), lw=1.5, alpha=0.7))
        
        # Highlight start and end points
        plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', edgecolor='black')
        plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='x', linewidth=2)
        
        # Add user ID labels
        plt.text(x_coords[0], y_coords[0], f' Start User {user_id}', fontsize=9, 
                verticalalignment='bottom', horizontalalignment='left')
        plt.text(x_coords[-1], y_coords[-1], f' End User {user_id}', fontsize=9, 
                verticalalignment='top', horizontalalignment='right')
    
    # Add time window indicators in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    time_labels = [f'Window {i+1}' for i in range(len(graph_data_list))]
    markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
               plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='End')]
    
    # Customize plot
    plt.title(f'User Preference Evolution Over Time ({method.upper()})', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(handles=handles + markers, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'user_embedding_evolution_{method}.png', dpi=300)
    plt.close()
    
    return reduced_user_embeddings


# Function to identify users with the most significant preference changes
def identify_preference_shifts(model, graph_data_list):
    """
    Identify users whose preferences have shifted the most across time windows.
    
    Args:
        model: Trained model
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
    
    # Visualize top shifts
    top_n = min(20, len(sorted_users))
    top_users = dict(sorted_users[:top_n])
    
    plt.figure(figsize=(12, 8))
    plt.barh(
        [f'User {user_id}' for user_id in top_users.keys()],
        list(top_users.values()),
        color='skyblue'
    )
    plt.xlabel('Embedding Shift Magnitude')
    plt.ylabel('User')
    plt.title('Users with Most Significant Preference Shifts')
    plt.tight_layout()
    plt.savefig('user_preference_shifts.png')
    plt.close()
    
    return dict(sorted_users)


# Enhanced visualization of user preferences
def visualize_user_preferences(model, graph_data_list, ratings_df):
    """Create visualizations of user preference patterns"""
    from sklearn.cluster import KMeans
    
    # Get user embeddings from the last time window
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
    
    user_embeddings = np.array(user_embeddings)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    user_embeddings_2d = pca.fit_transform(user_embeddings)
    
    # K-means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(user_embeddings)
    
    # Get user ratings by cluster
    cluster_ratings = defaultdict(list)
    
    for i, user_id in enumerate(user_ids):
        cluster = clusters[i]
        user_data = ratings_df[ratings_df['userId'] == user_id]
        avg_rating = user_data['rating'].mean() if len(user_data) > 0 else 0
        cluster_ratings[cluster].append(avg_rating)
    
    # Calculate average rating by cluster
    cluster_avg_ratings = {k: np.mean(v) for k, v in cluster_ratings.items()}
    
    # Plot user embeddings colored by cluster
    plt.figure(figsize=(12, 10))
    
    # Set a colormap
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    
    # Plot each cluster
    for cluster in range(n_clusters):
        cluster_points = user_embeddings_2d[clusters == cluster]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            c=[cmap(cluster)],
            label=f'Cluster {cluster+1} (Avg Rating: {cluster_avg_ratings[cluster]:.2f})',
            alpha=0.7,
            s=50
        )
    
    # Add title and labels
    plt.title('User Preference Clusters', fontsize=16)
    plt.xlabel('PCA Dimension 1', fontsize=14)
    plt.ylabel('PCA Dimension 2', fontsize=14)
    
    # Add a legend
    plt.legend(fontsize=12)
    
    # Show grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some styling
    plt.tight_layout()
    plt.savefig('user_preference_clusters.png', dpi=300)
    
    # Bar chart of average ratings by cluster
    plt.figure(figsize=(10, 6))
    clusters = sorted(cluster_avg_ratings.keys())
    avg_ratings = [cluster_avg_ratings[c] for c in clusters]
    
    bars = plt.bar(
        [f'Cluster {c+1}' for c in clusters],
        avg_ratings,
        color=[cmap(c) for c in clusters]
    )
    
    # Add average rating values on top of bars
    for bar, rating in zip(bars, avg_ratings):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f'{rating:.2f}',
            ha='center',
            fontsize=11
        )
    
    plt.title('Average Rating by User Cluster', fontsize=16)
    plt.ylabel('Average Rating', fontsize=14)
    plt.ylim(0, 5.5)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cluster_average_ratings.png', dpi=300)
    
    return user_ids, clusters


# Main function with enhanced training pipeline
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    embedding_dim = 128
    hidden_dim = 256
    num_heads = 4
    num_epochs = 30
    learning_rate = 0.001
    weight_decay = 1e-5
    time_windows = 3
    sample_size = 5000  # Limit users for faster training
    
    # Load the ratings data
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
        
        print(f"Original dataset: {len(ratings_df)} ratings from {ratings_df['userId'].nunique()} users on {ratings_df['movieId'].nunique()} movies")
    else:
        print("File not found. Using sample data instead.")
        # Create sample data (abbreviated for brevity)
        ratings_df = pd.DataFrame({
            'userId': [1, 1, 2, 2, 3],
            'movieId': [101, 102, 101, 103, 102],
            'rating': [5.0, 3.0, 4.0, 2.0, 5.0],
            'timestamp': [1000000000, 1000100000, 1000200000, 1000300000, 1000400000]
        })
    
    # Analyze initial rating distribution
    plt.figure(figsize=(10, 6))
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.title('Rating Distribution in Original Dataset')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('original_rating_distribution.png')
    plt.close()
    
    # Split data chronologically
    ratings_df = ratings_df.sort_values('timestamp')
    split_idx = int(0.9 * len(ratings_df))
    train_df = ratings_df.iloc[:split_idx]
    test_df = ratings_df.iloc[split_idx:]
    
    print(f"Training set: {len(train_df)} ratings")
    print(f"Test set: {len(test_df)} ratings")
    
    # Create balanced temporal graphs for training
    train_graph_data_list = create_balanced_temporal_graph(
        train_df, 
        time_windows=time_windows, 
        sample_size=sample_size,
        balance_ratings=True  # Enable rating balancing
    )
    
    if not train_graph_data_list:
        print("No training data available. Exiting...")
        return
    
    # Create test graph
    test_graph_data = create_enhanced_graph(test_df)
    print(f"Test graph: {test_graph_data.num_nodes} nodes, {test_graph_data.num_edges} edges")
    
    # Get dimensions from first graph
    first_graph = train_graph_data_list[0]
    num_users = first_graph.num_users
    num_items = first_graph.num_items
    
    print(f"Number of users in training: {num_users}")
    print(f"Number of items in training: {num_items}")
    
    # Initialize enhanced model
    model = EnhancedTemporalGNN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.3
    )
    
    # Print model architecture
    print("\nEnhanced Model Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train with advanced techniques
    trained_model, history = train_enhanced_model(
        model=model,
        graph_data_list=train_graph_data_list,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        focal_gamma=2.0,  # Focal loss parameter
        patience=7  # Early stopping patience
    )
    
    # Save the model
    torch.save(trained_model.state_dict(), 'enhanced_temporal_gnn_model.pth')
    
    # Evaluate model with comprehensive metrics
    print("\nEvaluating enhanced model on test data...")
    rmse, accuracy, conf_matrix, class_report, error_analysis = evaluate_enhanced_model(
        model=trained_model,
        test_graph_data=test_graph_data,
        ratings_df=test_df
    )
    
    # Save error analysis
    error_analysis.to_csv('prediction_error_analysis.csv', index=False)
    
    # Visualize user embeddings evolution
    visualize_user_embedding_evolution(
        model=trained_model,
        graph_data_list=train_graph_data_list,
        method='pca'
    )
    
    # Identify and visualize preference shifts
    user_shifts = identify_preference_shifts(
        model=trained_model,
        graph_data_list=train_graph_data_list
    )
    
    # Print users with most significant shifts
    print("\nUsers with the most significant preference shifts:")
    for user_id, shift in list(user_shifts.items())[:5]:
        print(f"User {user_id}: Shift magnitude = {shift:.4f}")
    
    # Create user preference visualizations
    user_clusters = visualize_user_preferences(
        model=trained_model,
        graph_data_list=train_graph_data_list,
        ratings_df=train_df
    )
    
    print("\nEnhanced analysis complete!")


if __name__ == "__main__":
    main()
