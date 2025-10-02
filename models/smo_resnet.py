'''Sparse ResNet in PyTorch with L1 Regularization and Pruning.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(SparseBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, 
                 l1_lambda=1e-5, sparsity_target=0.5):
        super(SparseResNet, self).__init__()
        self.in_planes = 64
        self.l1_lambda = l1_lambda  # L1 regularization strength
        self.sparsity_target = sparsity_target  # Target sparsity level

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        # Initialize binary masks for sparsity
        self.masks = {}
        self._initialize_masks()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_masks(self):
        """Initialize binary masks for sparse connections"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:  # Conv and Linear layers
                self.masks[name] = torch.ones_like(param.data)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def apply_masks(self):
        """Apply binary masks to enforce sparsity"""
        for name, param in self.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]

    def update_masks(self, prune_percentile=None):
        """
        Update masks based on weight magnitudes.
        Removes smallest weights to achieve target sparsity.
        
        Args:
            prune_percentile: Fraction of weights to prune (0-1)
        """
        if prune_percentile is None:
            prune_percentile = self.sparsity_target

        for name, param in self.named_parameters():
            if name in self.masks:
                # Get absolute values of weights
                weight_abs = torch.abs(param.data)
                
                # Calculate threshold for pruning
                threshold = torch.quantile(weight_abs.flatten(), prune_percentile)
                
                # Update mask: keep weights above threshold
                self.masks[name] = (weight_abs > threshold).float()
                
                # Apply mask immediately
                param.data *= self.masks[name]

    def update_masks_layerwise(self, prune_percentile=None):
        """
        Update masks layer-by-layer based on weight magnitudes.
        This can be better for structured pruning.
        
        Args:
            prune_percentile: Fraction of weights to prune per layer (0-1)
        """
        if prune_percentile is None:
            prune_percentile = self.sparsity_target

        for name, param in self.named_parameters():
            if name in self.masks:
                # Get absolute values of weights
                weight_abs = torch.abs(param.data)
                
                # Calculate threshold for this layer
                threshold = torch.quantile(weight_abs.flatten(), prune_percentile)
                
                # Update mask: keep weights above threshold
                self.masks[name] = (weight_abs > threshold).float()
                
                # Apply mask immediately
                param.data *= self.masks[name]

    def get_sparsity_stats(self):
        """Calculate current sparsity statistics"""
        total_params = 0
        zero_params = 0
        layer_stats = {}
        
        for name, param in self.named_parameters():
            if name in self.masks:
                layer_total = param.numel()
                layer_zeros = (param.data == 0).sum().item()
                
                total_params += layer_total
                zero_params += layer_zeros
                
                layer_sparsity = layer_zeros / layer_total if layer_total > 0 else 0
                layer_stats[name] = {
                    'total': layer_total,
                    'zeros': layer_zeros,
                    'sparsity': layer_sparsity
                }
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'sparsity': overall_sparsity,
            'density': 1 - overall_sparsity,
            'layer_stats': layer_stats
        }

    def l1_regularization_loss(self):
        """
        Calculate L1 regularization loss (sum of absolute values of weights).
        This encourages weights to become zero.
        """
        l1_loss = 0
        for name, param in self.named_parameters():
            if name in self.masks:
                l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

    def magnitude_based_loss(self, alpha=1.0):
        """
        Additional magnitude-based penalty that more aggressively pushes small weights to zero.
        Uses a smooth approximation of L0 norm: 1 - exp(-alpha * w^2)
        
        Args:
            alpha: Scaling factor for the exponential (higher = more aggressive)
        """
        mag_loss = 0
        for name, param in self.named_parameters():
            if name in self.masks:
                # Smooth approximation: 1 - exp(-alpha * w^2)
                mag_loss += torch.sum(1 - torch.exp(-alpha * param ** 2))
        return mag_loss

    def hoyer_sparsity_loss(self, layer_name=None):
        """
        Hoyer sparsity measure: (sqrt(n) - L1/L2) / (sqrt(n) - 1)
        Ranges from 0 (dense) to 1 (sparse)
        Can be used as an additional regularization term.
        """
        hoyer_loss = 0
        for name, param in self.named_parameters():
            if name in self.masks:
                if layer_name is None or layer_name in name:
                    n = param.numel()
                    l1_norm = torch.sum(torch.abs(param))
                    l2_norm = torch.sqrt(torch.sum(param ** 2))
                    
                    if l2_norm > 0:
                        hoyer = (torch.sqrt(torch.tensor(n, dtype=torch.float)) - l1_norm / l2_norm) / \
                                (torch.sqrt(torch.tensor(n, dtype=torch.float)) - 1)
                        hoyer_loss += (1 - hoyer)  # Minimize (1 - hoyer) to maximize sparsity
        
        return hoyer_loss


def training_step_with_sparsity(model, optimizer, inputs, targets, 
                                  criterion, use_magnitude_loss=True,
                                  use_hoyer_loss=False, mag_weight=0.0001):
    """
    Training step that incorporates L1 regularization and optional additional losses.
    
    Args:
        model: SparseResNet model
        optimizer: torch optimizer
        inputs: input batch
        targets: target labels
        criterion: loss function (e.g., CrossEntropyLoss)
        use_magnitude_loss: whether to use magnitude-based penalty
        use_hoyer_loss: whether to use Hoyer sparsity measure
        mag_weight: weight for magnitude loss
    
    Returns:
        Dictionary with loss components
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
    mag_loss = 0
    if use_magnitude_loss:
        mag_loss = model.magnitude_based_loss(alpha=0.01)
        total_loss += mag_weight * mag_loss
    
    # Optional: Add Hoyer sparsity loss
    hoyer_loss = 0
    if use_hoyer_loss:
        hoyer_loss = model.hoyer_sparsity_loss()
        total_loss += 0.001 * hoyer_loss
    
    # Backward pass
    total_loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Apply masks to enforce sparsity
    model.apply_masks()
    
    return {
        'total_loss': total_loss.item(),
        'cls_loss': cls_loss.item(),
        'l1_loss': l1_loss.item(),
        'mag_loss': mag_loss if isinstance(mag_loss, float) else mag_loss.item(),
        'hoyer_loss': hoyer_loss if isinstance(hoyer_loss, float) else hoyer_loss.item()
    }


def iterative_magnitude_pruning(model, train_loader, test_loader, criterion, 
                                  optimizer, device='cuda',
                                  epochs_per_prune=5, num_prune_iterations=10,
                                  final_sparsity=0.9, use_layerwise=False):
    """
    Implement Iterative Magnitude Pruning (IMP) for ResNet.
    Similar to the Lottery Ticket Hypothesis approach.
    
    Args:
        model: SparseResNet model
        train_loader: training dataloader
        test_loader: test dataloader
        criterion: loss function
        optimizer: torch optimizer
        device: device to train on
        epochs_per_prune: number of epochs to train between pruning iterations
        num_prune_iterations: number of pruning iterations
        final_sparsity: target final sparsity level
        use_layerwise: whether to prune layer-by-layer or globally
    """
    import numpy as np
    
    # Calculate sparsity schedule (gradually increase sparsity)
    sparsity_schedule = np.linspace(0, final_sparsity, num_prune_iterations + 1)[1:]
    
    print(f"Starting Iterative Magnitude Pruning for ResNet")
    print(f"Target sparsity schedule: {sparsity_schedule}")
    
    model = model.to(device)
    
    for prune_iter in range(num_prune_iterations):
        print(f"\n{'='*60}")
        print(f"Pruning Iteration {prune_iter + 1}/{num_prune_iterations}")
        print(f"{'='*60}")
        
        # Train for several epochs
        for epoch in range(epochs_per_prune):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                losses = training_step_with_sparsity(
                    model, optimizer, inputs, targets, criterion,
                    use_magnitude_loss=True
                )
                
                train_loss += losses['total_loss']
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()
                
                if batch_idx % 100 == 0:
                    stats = model.get_sparsity_stats()
                    print(f"Epoch {epoch+1}/{epochs_per_prune}, "
                          f"Batch {batch_idx}/{len(train_loader)}: "
                          f"Loss={losses['total_loss']:.4f}, "
                          f"Acc={100.*correct/total:.2f}%, "
                          f"Sparsity={stats['sparsity']:.2%}")
            
            # Test accuracy after each epoch
            test_acc = evaluate_model(model, test_loader, device)
            print(f"Epoch {epoch+1} - Train Acc: {100.*correct/total:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")
        
        # Prune weights after training
        target_sparsity = sparsity_schedule[prune_iter]
        if use_layerwise:
            model.update_masks_layerwise(prune_percentile=target_sparsity)
        else:
            model.update_masks(prune_percentile=target_sparsity)
        
        stats = model.get_sparsity_stats()
        print(f"\nAfter pruning to {target_sparsity:.1%}:")
        print(f"  Overall Sparsity: {stats['sparsity']:.2%}")
        print(f"  Active params: {stats['total_params'] - stats['zero_params']:,}")
        print(f"  Zero params: {stats['zero_params']:,}")
        
        # Print per-layer statistics
        print("\nPer-layer sparsity:")
        for name, layer_stat in stats['layer_stats'].items():
            if 'conv' in name or 'linear' in name:
                print(f"  {name}: {layer_stat['sparsity']:.2%}")


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


# Factory functions for different ResNet architectures
def SparseResNet18(l1_lambda=1e-5, sparsity_target=0.5, num_classes=10):
    return SparseResNet(SparseBasicBlock, [2, 2, 2, 2], 
                        num_classes=num_classes,
                        l1_lambda=l1_lambda, 
                        sparsity_target=sparsity_target)


def SparseResNet34(l1_lambda=1e-5, sparsity_target=0.5, num_classes=10):
    return SparseResNet(SparseBasicBlock, [3, 4, 6, 3],
                        num_classes=num_classes,
                        l1_lambda=l1_lambda,
                        sparsity_target=sparsity_target)


def SparseResNet50(l1_lambda=1e-5, sparsity_target=0.5, num_classes=10):
    return SparseResNet(SparseBottleneck, [3, 4, 6, 3],
                        num_classes=num_classes,
                        l1_lambda=l1_lambda,
                        sparsity_target=sparsity_target)


def SparseResNet101(l1_lambda=1e-5, sparsity_target=0.5, num_classes=10):
    return SparseResNet(SparseBottleneck, [3, 4, 23, 3],
                        num_classes=num_classes,
                        l1_lambda=l1_lambda,
                        sparsity_target=sparsity_target)


def SparseResNet152(l1_lambda=1e-5, sparsity_target=0.5, num_classes=10):
    return SparseResNet(SparseBottleneck, [3, 8, 36, 3],
                        num_classes=num_classes,
                        l1_lambda=l1_lambda,
                        sparsity_target=sparsity_target)


def test():
    # Test basic functionality
    net = SparseResNet18(l1_lambda=1e-4, sparsity_target=0.5)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(f"Output size: {y.size()}")
    
    # Test L1 regularization
    l1_loss = net.l1_regularization_loss()
    print(f"\nL1 Loss: {l1_loss.item():.6f}")
    
    # Test magnitude loss
    mag_loss = net.magnitude_based_loss()
    print(f"Magnitude Loss: {mag_loss.item():.6f}")
    
    # Test Hoyer sparsity
    hoyer_loss = net.hoyer_sparsity_loss()
    print(f"Hoyer Loss: {hoyer_loss.item():.6f}")
    
    # Test sparsity statistics
    stats = net.get_sparsity_stats()
    print(f"\nInitial sparsity: {stats['sparsity']:.2%}")
    print(f"Total params: {stats['total_params']:,}")
    
    # Simulate pruning
    print("\nPruning to 70% sparsity...")
    net.update_masks(prune_percentile=0.7)
    stats = net.get_sparsity_stats()
    print(f"After pruning: {stats['sparsity']:.2%}")
    print(f"Active params: {stats['total_params'] - stats['zero_params']:,}")
    
    # Show per-layer sparsity
    print("\nPer-layer sparsity (sample):")
    count = 0
    for name, layer_stat in stats['layer_stats'].items():
        if count < 5:  # Show first 5 layers
            print(f"  {name}: {layer_stat['sparsity']:.2%}")
            count += 1


if __name__ == '__main__':
    test()