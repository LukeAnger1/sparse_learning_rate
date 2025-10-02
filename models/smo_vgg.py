'''VGG11/13/16/19 in Pytorch with Sparse Training and L1 Regularization.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class SparseVGG(nn.Module):
    def __init__(self, vgg_name, l1_lambda=1e-5, sparsity_target=0.5):
        super(SparseVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.l1_lambda = l1_lambda  # L1 regularization strength
        self.sparsity_target = sparsity_target  # Target sparsity level (0.5 = 50% zeros)
        
        # Initialize binary masks for each layer (all ones initially)
        self.masks = {}
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize binary masks for sparse connections"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                self.masks[name] = torch.ones_like(param.data)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def apply_masks(self):
        """Apply binary masks to enforce sparsity"""
        for name, param in self.named_parameters():
            if 'weight' in name and name in self.masks:
                param.data *= self.masks[name]
    
    def update_masks(self, prune_percentile=None):
        """
        Update masks based on weight magnitudes.
        Removes smallest weights to achieve target sparsity.
        """
        if prune_percentile is None:
            prune_percentile = self.sparsity_target
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Get absolute values of weights
                weight_abs = torch.abs(param.data)
                
                # Calculate threshold for pruning
                threshold = torch.quantile(weight_abs.flatten(), prune_percentile)
                
                # Update mask: keep weights above threshold
                self.masks[name] = (weight_abs > threshold).float()
                
                # Apply mask immediately
                param.data *= self.masks[name]
    
    def get_sparsity_stats(self):
        """Calculate current sparsity statistics"""
        total_params = 0
        zero_params = 0
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'sparsity': sparsity,
            'density': 1 - sparsity
        }
    
    def l1_regularization_loss(self):
        """
        Calculate L1 regularization loss (sum of absolute values of weights).
        This encourages weights to become zero.
        """
        l1_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss
    
    def magnitude_based_loss(self, alpha=1.0):
        """
        Additional magnitude-based penalty that more aggressively pushes small weights to zero.
        Uses a smooth approximation of L0 norm.
        """
        mag_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Smooth approximation: 1 - exp(-alpha * w^2)
                # This is near 0 for small weights and near 1 for large weights
                mag_loss += torch.sum(1 - torch.exp(-alpha * param ** 2))
        return mag_loss
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def training_step_with_sparsity(model, optimizer, inputs, targets, 
                                  criterion, use_magnitude_loss=True):
    """
    Example training step that incorporates L1 regularization and optional magnitude loss.
    
    Args:
        model: SparseVGG model
        optimizer: torch optimizer
        inputs: input batch
        targets: target labels
        criterion: loss function (e.g., CrossEntropyLoss)
        use_magnitude_loss: whether to use additional magnitude-based penalty
    
    Returns:
        total_loss, classification_loss, regularization_loss
    """
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    
    # Classification loss
    cls_loss = criterion(outputs, targets)
    
    # L1 regularization loss
    l1_loss = model.l1_regularization_loss()
    
    # Total loss
    total_loss = cls_loss + l1_loss
    
    # Optional: Add magnitude-based loss
    if use_magnitude_loss:
        mag_loss = model.magnitude_based_loss(alpha=0.01)
        total_loss += 0.0001 * mag_loss  # Small weight for magnitude loss
    
    # Backward pass
    total_loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Apply masks to enforce sparsity
    model.apply_masks()
    
    return total_loss.item(), cls_loss.item(), l1_loss.item()


def iterative_magnitude_pruning(model, dataloader, criterion, optimizer, 
                                  epochs_per_prune=5, num_prune_iterations=10,
                                  final_sparsity=0.9):
    """
    Implement Iterative Magnitude Pruning (similar to Lottery Ticket Hypothesis).
    
    Args:
        model: SparseVGG model
        dataloader: training dataloader
        criterion: loss function
        optimizer: torch optimizer
        epochs_per_prune: number of epochs to train between pruning iterations
        num_prune_iterations: number of pruning iterations
        final_sparsity: target final sparsity level
    """
    import numpy as np
    
    # Calculate sparsity schedule
    sparsity_schedule = np.linspace(0, final_sparsity, num_prune_iterations + 1)[1:]
    
    print(f"Starting Iterative Magnitude Pruning")
    print(f"Target sparsity schedule: {sparsity_schedule}")
    
    for prune_iter in range(num_prune_iterations):
        print(f"\n=== Pruning Iteration {prune_iter + 1}/{num_prune_iterations} ===")
        
        # Train for several epochs
        for epoch in range(epochs_per_prune):
            model.train()
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                loss, cls_loss, reg_loss = training_step_with_sparsity(
                    model, optimizer, inputs, targets, criterion
                )
                
                if batch_idx % 100 == 0:
                    stats = model.get_sparsity_stats()
                    print(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Loss={loss:.4f}, Sparsity={stats['sparsity']:.2%}")
        
        # Prune weights after training
        target_sparsity = sparsity_schedule[prune_iter]
        model.update_masks(prune_percentile=target_sparsity)
        
        stats = model.get_sparsity_stats()
        print(f"After pruning: Sparsity = {stats['sparsity']:.2%}, "
              f"Active params = {stats['total_params'] - stats['zero_params']}")


def test():
    # Test basic functionality
    net = SparseVGG('VGG11', l1_lambda=1e-5, sparsity_target=0.5)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(f"Output size: {y.size()}")
    
    # Test L1 regularization
    l1_loss = net.l1_regularization_loss()
    print(f"L1 Loss: {l1_loss.item():.6f}")
    
    # Test magnitude loss
    mag_loss = net.magnitude_based_loss()
    print(f"Magnitude Loss: {mag_loss.item():.6f}")
    
    # Test sparsity statistics
    stats = net.get_sparsity_stats()
    print(f"Initial sparsity: {stats['sparsity']:.2%}")
    
    # Simulate pruning
    net.update_masks(prune_percentile=0.5)
    stats = net.get_sparsity_stats()
    print(f"After 50% pruning: {stats['sparsity']:.2%}")

if __name__ == '__main__':
    test()