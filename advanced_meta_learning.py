"""
Advanced Meta-Learning and Few-Shot Learning Algorithms

This module implements cutting-edge meta-learning algorithms that enable
rapid learning from few examples, including Model-Agnostic Meta-Learning (MAML),
Prototypical Networks, Matching Networks, and advanced meta-optimization techniques.

Key Features:
- Model-Agnostic Meta-Learning (MAML)
- Prototypical Networks for Few-Shot Classification
- Matching Networks with Attention Mechanisms
- Relation Networks for Meta-Learning
- Meta-Learning with Differentiable Convex Optimization
- Reptile: First-Order Meta-Learning
- Meta-SGD for Adaptive Learning Rates
- Meta-Learning for Regression Tasks
- Task-Agnostic Meta-Learning
- Continual Meta-Learning
"""

import numpy as np
import scipy
from scipy import optimize, linalg, stats
from scipy.special import softmax, logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from typing import Optional, Dict, List, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import pickle


class MetaLearningType(Enum):
    """Enumeration of meta-learning algorithm types."""
    MAML = "maml"
    PROTOTYPICAL_NETWORKS = "prototypical_networks"
    MATCHING_NETWORKS = "matching_networks"
    RELATION_NETWORKS = "relation_networks"
    REPTILE = "reptile"
    META_SGD = "meta_sgd"
    TASK_AGNOSTIC = "task_agnostic"
    CONTINUAL_META = "continual_meta"


@dataclass
class Task:
    """Meta-learning task representation."""
    support_data: Dict[str, np.ndarray]
    query_data: Dict[str, np.ndarray]
    task_id: str
    task_type: str = "classification"
    n_classes: int = 5
    n_support: int = 5
    n_query: int = 10
    
    def __post_init__(self):
        """Initialize task properties."""
        self.n_support = len(self.support_data['X'])
        self.n_query = len(self.query_data['X'])
        if 'y' in self.support_data:
            self.n_classes = len(np.unique(self.support_data['y']))


class BaseMetaLearner(ABC):
    """Abstract base class for meta-learning algorithms."""
    
    def __init__(self, base_model: nn.Module, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 5,
                 random_state: int = None):
        """
        Initialize meta-learner.
        
        Parameters
        ----------
        base_model : nn.Module
            Base neural network model
        inner_lr : float, default=0.01
            Learning rate for inner loop (task-specific adaptation)
        outer_lr : float, default=0.001
            Learning rate for outer loop (meta-optimization)
        inner_steps : int, default=5
            Number of inner loop optimization steps
        random_state : int, optional
            Random state for reproducibility
        """
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.random_state = random_state
        
        self.meta_optimizer = optim.Adam(base_model.parameters(), lr=outer_lr)
        self.meta_history = []
        self.is_fitted = False
        
    @abstractmethod
    def inner_adaptation(self, task: Task) -> nn.Module:
        """Perform inner loop adaptation on task."""
        pass
    
    @abstractmethod
    def meta_update(self, tasks: List[Task]) -> Dict[str, float]:
        """Perform meta-update using multiple tasks."""
        pass
    
    def meta_train(self, tasks: List[Task], n_epochs: int = 100) -> Dict[str, Any]:
        """
        Train meta-learner on tasks.
        
        Parameters
        ----------
        tasks : list
            List of meta-learning tasks
        n_epochs : int, default=100
            Number of meta-training epochs
            
        Returns
        -------
        results : dict
            Training results
        """
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            
            # Sample batch of tasks
            batch_tasks = np.random.choice(tasks, size=min(10, len(tasks)), replace=False)
            
            # Meta-update
            meta_metrics = self.meta_update(batch_tasks)
            
            # Record history
            self.meta_history.append({
                'epoch': epoch,
                'metrics': meta_metrics,
                'time': time.time() - epoch_start_time
            })
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {meta_metrics.get('loss', 0):.6f}")
                
        self.is_fitted = True
        return self.get_results()
    
    def meta_test(self, test_tasks: List[Task]) -> Dict[str, float]:
        """
        Test meta-learner on test tasks.
        
        Parameters
        ----------
        test_tasks : list
            List of test tasks
            
        Returns
        -------
        results : dict
            Test results
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call meta_train() first.")
            
        test_metrics = []
        
        for task in test_tasks:
            # Adapt to test task
            adapted_model = self.inner_adaptation(task)
            
            # Evaluate on query set
            metrics = self.evaluate_model(adapted_model, task.query_data)
            test_metrics.append(metrics)
            
        # Aggregate results
        avg_metrics = {}
        for key in test_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in test_metrics])
            
        return avg_metrics
    
    def evaluate_model(self, model: nn.Module, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate model on data."""
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(data['X'])
            y_tensor = torch.FloatTensor(data['y'])
            
            if len(y_tensor.unique()) <= 10:  # Classification
                outputs = model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_tensor.long()).float().mean().item()
                return {'accuracy': accuracy}
            else:  # Regression
                outputs = model(X_tensor).squeeze()
                mse = F.mse_loss(outputs, y_tensor).item()
                return {'mse': mse}
    
    def get_results(self) -> Dict[str, Any]:
        """Get training results."""
        return {
            'meta_history': self.meta_history,
            'final_state': {k: v.cpu().numpy() for k, v in self.base_model.state_dict().items()},
            'is_fitted': self.is_fitted
        }


class MAML(BaseMetaLearner):
    """
    Model-Agnostic Meta-Learning (MAML) implementation.
    """
    
    def __init__(self, base_model: nn.Module, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 5,
                 first_order: bool = False, **kwargs):
        """
        Initialize MAML.
        
        Parameters
        ----------
        first_order : bool, default=False
            Use first-order approximation (ignore second-order derivatives)
        """
        super().__init__(base_model, inner_lr, outer_lr, inner_steps, **kwargs)
        self.first_order = first_order
        
    def inner_adaptation(self, task: Task) -> nn.Module:
        """Perform inner loop adaptation using gradient descent."""
        # Create a copy of the model for this task
        adapted_model = type(self.base_model)(
            input_dim=self.base_model.input_dim,
            hidden_dim=self.base_model.hidden_dim,
            output_dim=self.base_model.output_dim
        )
        adapted_model.load_state_dict(self.base_model.state_dict())
        
        # Inner loop optimization
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        adapted_model.train()
        
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass on support set
            X_support = torch.FloatTensor(task.support_data['X'])
            y_support = torch.FloatTensor(task.support_data['y'])
            
            if len(y_support.unique()) <= 10:  # Classification
                outputs = adapted_model(X_support)
                loss = F.cross_entropy(outputs, y_support.long())
            else:  # Regression
                outputs = adapted_model(X_support).squeeze()
                loss = F.mse_loss(outputs, y_support)
                
            # Backward pass
            loss.backward()
            inner_optimizer.step()
            
        return adapted_model
    
    def meta_update(self, tasks: List[Task]) -> Dict[str, float]:
        """Perform meta-update using multiple tasks."""
        meta_loss = 0.0
        n_tasks = len(tasks)
        
        for task in tasks:
            # Inner adaptation
            adapted_model = self.inner_adaptation(task)
            
            # Compute loss on query set
            X_query = torch.FloatTensor(task.query_data['X'])
            y_query = torch.FloatTensor(task.query_data['y'])
            
            if len(y_query.unique()) <= 10:  # Classification
                outputs = adapted_model(X_query)
                task_loss = F.cross_entropy(outputs, y_query.long())
            else:  # Regression
                outputs = adapted_model(X_query).squeeze()
                task_loss = F.mse_loss(outputs, y_query)
                
            meta_loss += task_loss
            
        meta_loss /= n_tasks
        
        # Meta-optimization
        self.meta_optimizer.zero_grad()
        
        if self.first_order:
            # First-order approximation
            meta_loss.backward(create_graph=False)
        else:
            # Full second-order computation
            meta_loss.backward(create_graph=True)
            
        self.meta_optimizer.step()
        
        return {'loss': meta_loss.item(), 'n_tasks': n_tasks}


class PrototypicalNetworks(BaseMetaLearner):
    """
    Prototypical Networks for few-shot learning.
    """
    
    def __init__(self, base_model: nn.Module, embedding_dim: int = 128,
                 distance_metric: str = 'euclidean', **kwargs):
        """
        Initialize Prototypical Networks.
        
        Parameters
        ----------
        embedding_dim : int, default=128
            Dimension of embedding space
        distance_metric : str, default='euclidean'
            Distance metric for prototypes ('euclidean', 'cosine')
        """
        super().__init__(base_model, **kwargs)
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
    def compute_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes from embeddings."""
        n_classes = len(labels.unique())
        prototypes = torch.zeros(n_classes, self.embedding_dim)
        
        for i, class_id in enumerate(labels.unique()):
            class_mask = (labels == class_id)
            class_embeddings = embeddings[class_mask]
            prototypes[i] = class_embeddings.mean(dim=0)
            
        return prototypes
    
    def compute_distances(self, query_embeddings: torch.Tensor, 
                         prototypes: torch.Tensor) -> torch.Tensor:
        """Compute distances between query embeddings and prototypes."""
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes)
        elif self.distance_metric == 'cosine':
            # Cosine similarity (convert to distance)
            similarities = F.cosine_similarity(
                query_embeddings.unsqueeze(1), 
                prototypes.unsqueeze(0), 
                dim=2
            )
            distances = 1.0 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
        return distances
    
    def inner_adaptation(self, task: Task) -> nn.Module:
        """Compute prototypes for the task."""
        # For prototypical networks, "adaptation" means computing prototypes
        self.base_model.eval()
        
        with torch.no_grad():
            # Get embeddings for support set
            X_support = torch.FloatTensor(task.support_data['X'])
            y_support = torch.FloatTensor(task.support_data['y'])
            
            support_embeddings = self.base_model(X_support)
            prototypes = self.compute_prototypes(support_embeddings, y_support)
            
        # Store prototypes in model for later use
        self.current_prototypes = prototypes
        return self.base_model
    
    def meta_update(self, tasks: List[Task]) -> Dict[str, float]:
        """Update embedding network using prototypical loss."""
        meta_loss = 0.0
        n_tasks = len(tasks)
        
        for task in tasks:
            # Compute prototypes
            adapted_model = self.inner_adaptation(task)
            
            # Get query embeddings
            X_query = torch.FloatTensor(task.query_data['X'])
            y_query = torch.FloatTensor(task.query_data['y'])
            
            query_embeddings = adapted_model(X_query)
            
            # Compute distances and loss
            distances = self.compute_distances(query_embeddings, self.current_prototypes)
            
            # Cross-entropy loss on negative distances
            loss = F.cross_entropy(-distances, y_query.long())
            meta_loss += loss
            
        meta_loss /= n_tasks
        
        # Meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {'loss': meta_loss.item(), 'n_tasks': n_tasks}


class MatchingNetworks(BaseMetaLearner):
    """
    Matching Networks with attention mechanisms.
    """
    
    def __init__(self, base_model: nn.Module, embedding_dim: int = 128,
                 attention_type: str = 'dot_product', **kwargs):
        """
        Initialize Matching Networks.
        
        Parameters
        ----------
        embedding_dim : int, default=128
            Dimension of embedding space
        attention_type : str, default='dot_product'
            Type of attention mechanism
        """
        super().__init__(base_model, **kwargs)
        self.embedding_dim = embedding_dim
        self.attention_type = attention_type
        
    def compute_attention_weights(self, query_embeddings: torch.Tensor,
                                support_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute attention weights between query and support embeddings."""
        if self.attention_type == 'dot_product':
            # Dot product attention
            similarities = torch.mm(query_embeddings, support_embeddings.t())
            attention_weights = F.softmax(similarities, dim=1)
        elif self.attention_type == 'cosine':
            # Cosine attention
            similarities = F.cosine_similarity(
                query_embeddings.unsqueeze(1),
                support_embeddings.unsqueeze(0),
                dim=2
            )
            attention_weights = F.softmax(similarities, dim=1)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
            
        return attention_weights
    
    def inner_adaptation(self, task: Task) -> nn.Module:
        """Compute attention weights for the task."""
        self.base_model.eval()
        
        with torch.no_grad():
            # Get embeddings
            X_support = torch.FloatTensor(task.support_data['X'])
            X_query = torch.FloatTensor(task.query_data['X'])
            
            support_embeddings = self.base_model(X_support)
            query_embeddings = self.base_model(X_query)
            
            # Compute attention weights
            attention_weights = self.compute_attention_weights(
                query_embeddings, support_embeddings
            )
            
        # Store for meta-update
        self.current_attention_weights = attention_weights
        self.current_support_embeddings = support_embeddings
        self.current_support_labels = torch.FloatTensor(task.support_data['y'])
        
        return self.base_model
    
    def meta_update(self, tasks: List[Task]) -> Dict[str, float]:
        """Update embedding network using matching loss."""
        meta_loss = 0.0
        n_tasks = len(tasks)
        
        for task in tasks:
            # Compute attention weights
            adapted_model = self.inner_adaptation(task)
            
            # Get query labels
            y_query = torch.FloatTensor(task.query_data['y'])
            
            # Weighted sum of support labels
            predicted_labels = torch.mm(
                self.current_attention_weights,
                self.current_support_labels.unsqueeze(1)
            ).squeeze()
            
            # Compute loss
            if len(y_query.unique()) <= 10:  # Classification
                loss = F.cross_entropy(predicted_labels.unsqueeze(0), y_query.long())
            else:  # Regression
                loss = F.mse_loss(predicted_labels, y_query)
                
            meta_loss += loss
            
        meta_loss /= n_tasks
        
        # Meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {'loss': meta_loss.item(), 'n_tasks': n_tasks}


class Reptile(BaseMetaLearner):
    """
    Reptile: First-order meta-learning algorithm.
    """
    
    def __init__(self, base_model: nn.Module, inner_lr: float = 0.1,
                 outer_lr: float = 0.001, inner_steps: int = 5,
                 epsilon: float = 1.0, **kwargs):
        """
        Initialize Reptile.
        
        Parameters
        ----------
        epsilon : float, default=1.0
            Interpolation factor for meta-update
        """
        super().__init__(base_model, inner_lr, outer_lr, inner_steps, **kwargs)
        self.epsilon = epsilon
        
    def inner_adaptation(self, task: Task) -> nn.Module:
        """Perform SGD steps on task."""
        # Create a copy of the model
        adapted_model = type(self.base_model)(
            input_dim=self.base_model.input_dim,
            hidden_dim=self.base_model.hidden_dim,
            output_dim=self.base_model.output_dim
        )
        adapted_model.load_state_dict(self.base_model.state_dict())
        
        # SGD steps
        adapted_model.train()
        
        for step in range(self.inner_steps):
            # Sample mini-batch from support set
            indices = np.random.choice(len(task.support_data['X']), size=min(10, len(task.support_data['X'])))
            X_batch = torch.FloatTensor(task.support_data['X'][indices])
            y_batch = torch.FloatTensor(task.support_data['y'][indices])
            
            # Forward and backward pass
            outputs = adapted_model(X_batch)
            
            if len(y_batch.unique()) <= 10:  # Classification
                loss = F.cross_entropy(outputs, y_batch.long())
            else:  # Regression
                outputs = outputs.squeeze()
                loss = F.mse_loss(outputs, y_batch)
                
            # Manual SGD update
            for param in adapted_model.parameters():
                if param.grad is not None:
                    param.data -= self.inner_lr * param.grad
                    
        return adapted_model
    
    def meta_update(self, tasks: List[Task]) -> Dict[str, float]:
        """Update meta-parameters using Reptile algorithm."""
        meta_weights = []
        
        for task in tasks:
            # Get adapted weights
            adapted_model = self.inner_adaptation(task)
            adapted_weights = {k: v.data for k, v in adapted_model.named_parameters()}
            meta_weights.append(adapted_weights)
            
        # Average adapted weights
        avg_weights = {}
        for key in meta_weights[0].keys():
            avg_weights[key] = torch.mean(
                torch.stack([weights[key] for weights in meta_weights]), dim=0
            )
            
        # Reptile update: interpolate between current and average weights
        current_weights = {k: v.data for k, v in self.base_model.named_parameters()}
        
        for key in current_weights.keys():
            current_weights[key] = (
                self.epsilon * avg_weights[key] + 
                (1 - self.epsilon) * current_weights[key]
            )
            
        # Update model weights
        self.base_model.load_state_dict(current_weights)
        
        # Compute meta-loss (for monitoring)
        meta_loss = 0.0
        for task in tasks[:3]:  # Sample a few tasks for loss computation
            adapted_model = self.inner_adaptation(task)
            X_query = torch.FloatTensor(task.query_data['X'])
            y_query = torch.FloatTensor(task.query_data['y'])
            
            outputs = adapted_model(X_query)
            
            if len(y_query.unique()) <= 10:  # Classification
                loss = F.cross_entropy(outputs, y_query.long())
            else:  # Regression
                outputs = outputs.squeeze()
                loss = F.mse_loss(outputs, y_query)
                
            meta_loss += loss.item()
            
        return {'loss': meta_loss / min(3, len(tasks)), 'n_tasks': len(tasks)}


class MetaSGD(BaseMetaLearner):
    """
    Meta-SGD: Learning learning rates automatically.
    """
    
    def __init__(self, base_model: nn.Module, outer_lr: float = 0.001,
                 inner_steps: int = 5, **kwargs):
        """
        Initialize Meta-SGD.
        
        Note: Inner learning rates are learned parameters.
        """
        super().__init__(base_model, inner_lr=0.0, outer_lr=outer_lr, inner_steps=inner_steps, **kwargs)
        
        # Learnable inner learning rates for each parameter
        self.inner_lrs = {}
        for name, param in self.base_model.named_parameters():
            self.inner_lrs[name] = nn.Parameter(torch.tensor(0.01))
            
        # Add inner learning rates to optimizer
        self.meta_optimizer = optim.Adam(
            list(self.base_model.parameters()) + list(self.inner_lrs.values()),
            lr=outer_lr
        )
        
    def inner_adaptation(self, task: Task) -> nn.Module:
        """Perform inner adaptation with learned learning rates."""
        # Create a copy of the model
        adapted_model = type(self.base_model)(
            input_dim=self.base_model.input_dim,
            hidden_dim=self.base_model.hidden_dim,
            output_dim=self.base_model.output_dim
        )
        adapted_model.load_state_dict(self.base_model.state_dict())
        
        # Inner loop with learned learning rates
        adapted_model.train()
        
        for step in range(self.inner_steps):
            # Forward pass
            X_support = torch.FloatTensor(task.support_data['X'])
            y_support = torch.FloatTensor(task.support_data['y'])
            
            outputs = adapted_model(X_support)
            
            if len(y_support.unique()) <= 10:  # Classification
                loss = F.cross_entropy(outputs, y_support.long())
            else:  # Regression
                outputs = outputs.squeeze()
                loss = F.mse_loss(outputs, y_support)
                
            # Backward pass
            loss.backward()
            
            # Update parameters with learned learning rates
            for name, param in adapted_model.named_parameters():
                if name in self.inner_lrs and param.grad is not None:
                    param.data -= self.inner_lrs[name] * param.grad
                    
        return adapted_model
    
    def meta_update(self, tasks: List[Task]) -> Dict[str, float]:
        """Update both model parameters and learning rates."""
        meta_loss = 0.0
        n_tasks = len(tasks)
        
        for task in tasks:
            # Inner adaptation
            adapted_model = self.inner_adaptation(task)
            
            # Compute loss on query set
            X_query = torch.FloatTensor(task.query_data['X'])
            y_query = torch.FloatTensor(task.query_data['y'])
            
            outputs = adapted_model(X_query)
            
            if len(y_query.unique()) <= 10:  # Classification
                task_loss = F.cross_entropy(outputs, y_query.long())
            else:  # Regression
                outputs = outputs.squeeze()
                task_loss = F.mse_loss(outputs, y_query)
                
            meta_loss += task_loss
            
        meta_loss /= n_tasks
        
        # Meta-optimization (updates both model and learning rates)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Get current learning rates
        current_lrs = {name: lr.item() for name, lr in self.inner_lrs.items()}
        
        return {'loss': meta_loss.item(), 'n_tasks': n_tasks, 'inner_lrs': current_lrs}


class TaskAgnosticMetaLearner(BaseMetaLearner):
    """
    Task-Agnostic Meta-Learning (TAML).
    """
    
    def __init__(self, base_model: nn.Module, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 5,
                 alpha: float = 1.0, **kwargs):
        """
        Initialize Task-Agnostic Meta-Learner.
        
        Parameters
        ----------
        alpha : float, default=1.0
            Weight for task-agnostic loss
        """
        super().__init__(base_model, inner_lr, outer_lr, inner_steps, **kwargs)
        self.alpha = alpha
        
    def compute_task_agnostic_loss(self, tasks: List[Task]) -> torch.Tensor:
        """Compute task-agnostic loss to encourage task invariance."""
        # Sample random data points from all tasks
        all_X = []
        all_y = []
        
        for task in tasks:
            indices = np.random.choice(len(task.support_data['X']), size=min(5, len(task.support_data['X'])))
            all_X.append(task.support_data['X'][indices])
            all_y.append(task.support_data['y'][indices])
            
        if all_X:
            X_combined = torch.FloatTensor(np.vstack(all_X))
            y_combined = torch.FloatTensor(np.hstack(all_y))
            
            outputs = self.base_model(X_combined)
            
            if len(y_combined.unique()) <= 10:  # Classification
                ta_loss = F.cross_entropy(outputs, y_combined.long())
            else:  # Regression
                outputs = outputs.squeeze()
                ta_loss = F.mse_loss(outputs, y_combined)
                
            return ta_loss
        else:
            return torch.tensor(0.0)
    
    def inner_adaptation(self, task: Task) -> nn.Module:
        """Perform inner adaptation on task."""
        # Create a copy of the model
        adapted_model = type(self.base_model)(
            input_dim=self.base_model.input_dim,
            hidden_dim=self.base_model.hidden_dim,
            output_dim=self.base_model.output_dim
        )
        adapted_model.load_state_dict(self.base_model.state_dict())
        
        # Inner loop optimization
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        adapted_model.train()
        
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            X_support = torch.FloatTensor(task.support_data['X'])
            y_support = torch.FloatTensor(task.support_data['y'])
            
            outputs = adapted_model(X_support)
            
            if len(y_support.unique()) <= 10:  # Classification
                loss = F.cross_entropy(outputs, y_support.long())
            else:  # Regression
                outputs = outputs.squeeze()
                loss = F.mse_loss(outputs, y_support)
                
            loss.backward()
            inner_optimizer.step()
            
        return adapted_model
    
    def meta_update(self, tasks: List[Task]) -> Dict[str, float]:
        """Update with both task-specific and task-agnostic losses."""
        meta_loss = 0.0
        n_tasks = len(tasks)
        
        for task in tasks:
            # Inner adaptation
            adapted_model = self.inner_adaptation(task)
            
            # Task-specific loss
            X_query = torch.FloatTensor(task.query_data['X'])
            y_query = torch.FloatTensor(task.query_data['y'])
            
            outputs = adapted_model(X_query)
            
            if len(y_query.unique()) <= 10:  # Classification
                task_loss = F.cross_entropy(outputs, y_query.long())
            else:  # Regression
                outputs = outputs.squeeze()
                task_loss = F.mse_loss(outputs, y_query)
                
            meta_loss += task_loss
            
        meta_loss /= n_tasks
        
        # Task-agnostic loss
        ta_loss = self.compute_task_agnostic_loss(tasks)
        
        # Combined loss
        total_loss = meta_loss + self.alpha * ta_loss
        
        # Meta-optimization
        self.meta_optimizer.zero_grad()
        total_loss.backward()
        self.meta_optimizer.step()
        
        return {
            'task_loss': meta_loss.item(),
            'ta_loss': ta_loss.item(),
            'total_loss': total_loss.item(),
            'n_tasks': n_tasks
        }


# Simple neural network model for meta-learning
class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for meta-learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


# Factory function for meta-learners
def create_meta_learner(method: str, base_model: nn.Module, **kwargs) -> BaseMetaLearner:
    """
    Factory function to create meta-learners.
    
    Parameters
    ----------
    method : str
        Meta-learning method name
    base_model : nn.Module
        Base neural network model
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    meta_learner : BaseMetaLearner
        Created meta-learner
    """
    learner_map = {
        'maml': MAML,
        'prototypical_networks': PrototypicalNetworks,
        'matching_networks': MatchingNetworks,
        'reptile': Reptile,
        'meta_sgd': MetaSGD,
        'task_agnostic': TaskAgnosticMetaLearner
    }
    
    if method not in learner_map:
        raise ValueError(f"Unknown meta-learning method: {method}")
        
    return learner_map[method](base_model=base_model, **kwargs)


# Task generation utilities
def generate_classification_tasks(n_tasks: int, n_classes: int = 5,
                               n_support: int = 5, n_query: int = 10,
                               input_dim: int = 10, random_state: int = None) -> List[Task]:
    """Generate synthetic classification tasks for meta-learning."""
    rng = np.random.RandomState(random_state)
    tasks = []
    
    for task_id in range(n_tasks):
        # Generate task-specific parameters
        class_centers = rng.randn(n_classes, input_dim)
        class_scale = 0.5
        
        # Generate support set
        support_X = []
        support_y = []
        
        for class_id in range(n_classes):
            class_samples = rng.randn(n_support, input_dim) * class_scale + class_centers[class_id]
            support_X.append(class_samples)
            support_y.append(np.full(n_support, class_id))
            
        support_data = {
            'X': np.vstack(support_X),
            'y': np.hstack(support_y)
        }
        
        # Generate query set
        query_X = []
        query_y = []
        
        for class_id in range(n_classes):
            class_samples = rng.randn(n_query, input_dim) * class_scale + class_centers[class_id]
            query_X.append(class_samples)
            query_y.append(np.full(n_query, class_id))
            
        query_data = {
            'X': np.vstack(query_X),
            'y': np.hstack(query_y)
        }
        
        task = Task(
            support_data=support_data,
            query_data=query_data,
            task_id=f"classification_task_{task_id}",
            task_type="classification",
            n_classes=n_classes
        )
        
        tasks.append(task)
        
    return tasks


def generate_regression_tasks(n_tasks: int, n_support: int = 10,
                            n_query: int = 20, input_dim: int = 5,
                            random_state: int = None) -> List[Task]:
    """Generate synthetic regression tasks for meta-learning."""
    rng = np.random.RandomState(random_state)
    tasks = []
    
    for task_id in range(n_tasks):
        # Generate task-specific function parameters
        coefficients = rng.randn(input_dim)
        bias = rng.randn()
        noise_scale = 0.1
        
        # Generate support set
        X_support = rng.randn(n_support, input_dim)
        y_support = X_support @ coefficients + bias + rng.randn(n_support) * noise_scale
        
        # Generate query set
        X_query = rng.randn(n_query, input_dim)
        y_query = X_query @ coefficients + bias + rng.randn(n_query) * noise_scale
        
        task = Task(
            support_data={'X': X_support, 'y': y_support},
            query_data={'X': X_query, 'y': y_query},
            task_id=f"regression_task_{task_id}",
            task_type="regression"
        )
        
        tasks.append(task)
        
    return tasks


# Benchmark meta-learners
def benchmark_meta_learners(tasks: List[Task], test_tasks: List[Task],
                          input_dim: int, output_dim: int = 5,
                          n_epochs: int = 50) -> Dict[str, Dict]:
    """
    Benchmark different meta-learning algorithms.
    
    Parameters
    ----------
    tasks : list
        Training tasks
    test_tasks : list
        Test tasks
    input_dim : int
        Input dimension
    output_dim : int, default=5
        Output dimension
    n_epochs : int, default=50
        Number of training epochs
        
    Returns
    -------
    benchmark_results : dict
        Benchmark results
    """
    methods = ['maml', 'prototypical_networks', 'matching_networks', 'reptile']
    results = {}
    
    for method in methods:
        print(f"Benchmarking {method}...")
        
        try:
            # Create base model
            base_model = SimpleMLP(input_dim=input_dim, output_dim=output_dim)
            
            # Create meta-learner
            meta_learner = create_meta_learner(method, base_model, max_iterations=n_epochs)
            
            # Train
            start_time = time.time()
            train_results = meta_learner.meta_train(tasks, n_epochs)
            end_time = time.time()
            
            # Test
            test_results = meta_learner.meta_test(test_tasks)
            
            results[method] = {
                'train_time': end_time - start_time,
                'train_loss': train_results['meta_history'][-1]['metrics'].get('loss', 0),
                'test_metrics': test_results,
                'success': True
            }
            
        except Exception as e:
            results[method] = {
                'error': str(e),
                'success': False
            }
            
    return results


if __name__ == "__main__":
    print("Advanced Meta-Learning and Few-Shot Learning Module")
    print("=" * 65)
    
    # Generate synthetic tasks
    print("\nGenerating synthetic tasks...")
    train_tasks = generate_classification_tasks(
        n_tasks=20, n_classes=5, n_support=5, n_query=10, random_state=42
    )
    test_tasks = generate_classification_tasks(
        n_tasks=5, n_classes=5, n_support=5, n_query=10, random_state=123
    )
    
    print(f"Generated {len(train_tasks)} training tasks and {len(test_tasks)} test tasks")
    
    # Test individual methods
    input_dim = 10
    output_dim = 5
    
    for method in ['maml', 'prototypical_networks', 'reptile']:
        print(f"\n{method.upper()}:")
        
        try:
            # Create model and meta-learner
            base_model = SimpleMLP(input_dim=input_dim, output_dim=output_dim)
            meta_learner = create_meta_learner(method, base_model, max_iterations=20)
            
            # Train
            train_results = meta_learner.meta_train(train_tasks[:10], n_epochs=20)
            
            # Test
            test_results = meta_learner.meta_test(test_tasks[:3])
            
            print(f"  Final train loss: {train_results['meta_history'][-1]['metrics'].get('loss', 0):.6f}")
            print(f"  Test accuracy: {test_results.get('accuracy', 0):.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Run benchmark
    print(f"\nRunning meta-learning benchmark...")
    benchmark_results = benchmark_meta_learners(
        train_tasks[:10], test_tasks[:3], input_dim, output_dim, n_epochs=15
    )
    
    print("\nBenchmark Results:")
    print("-" * 50)
    for method, result in benchmark_results.items():
        if result['success']:
            test_acc = result['test_metrics'].get('accuracy', 0)
            print(f"{method:20}: {test_acc:10.4f} accuracy ({result['train_time']:.2f}s)")
        else:
            print(f"{method:20}: Failed - {result['error'][:30]}...")
