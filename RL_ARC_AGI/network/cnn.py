import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for IMPALA ResNet architecture."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class IMPALAResNet(nn.Module):
    """IMPALA ResNet backbone for processing grid observations."""
    
    def __init__(self, input_channels=1, num_classes=512):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual blocks
        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 32, 2, stride=2)
        
        # Global average pooling and final linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)
        
        # Initialize weights
        # self.apply(self._init_weights)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = x.float()
        x = x / 10.0
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Final linear layer
        features = self.fc(x)
        
        return features


class CNNPolicy(nn.Module):
    """CNN-based policy network using IMPALA ResNet architecture."""
    
    def __init__(self, input_channels=1, hidden_dim=512, action_size=9000, dropout=0.1):
        super().__init__()
        
        # IMPALA ResNet backbone
        self.backbone = IMPALAResNet(input_channels, hidden_dim)
        
        # Policy head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_size)
        )
        
        # Initialize weights
        # self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass returning action logits."""
        # Extract features using ResNet backbone
        features = self.backbone(x)
        
        # Policy head
        action_logits = self.actor(features)
        
        return action_logits


class CNNValue(nn.Module):
    """CNN-based value network using IMPALA ResNet architecture."""
    
    def __init__(self, input_channels=1, hidden_dim=512, dropout=0.1):
        super().__init__()
        
        # IMPALA ResNet backbone
        self.backbone = IMPALAResNet(input_channels, hidden_dim)
        
        # Value head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        # self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass returning state value."""
        # Extract features using ResNet backbone
        features = self.backbone(x)
        
        # Value head
        state_value = self.critic(features)
        
        return state_value


class CNNQValue(nn.Module):
    """CNN-based Q-value network using IMPALA ResNet architecture."""
    
    def __init__(self, input_channels=1, hidden_dim=512, action_size=9000, dropout=0.1):
        super().__init__()
        
        # IMPALA ResNet backbone
        self.backbone = IMPALAResNet(input_channels, hidden_dim)
        
        # Q-value head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_size)
        )
        
        # Initialize weights
        # self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass returning Q-values for all actions."""
        # Extract features using ResNet backbone
        features = self.backbone(x)
        
        # Q-value head
        q_values = self.critic(features)
        
        return q_values