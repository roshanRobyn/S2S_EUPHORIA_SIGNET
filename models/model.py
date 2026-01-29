import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph import Graph

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_class=100, in_channels=3, edge_strategy='spatial'):
        super().__init__()
        self.graph = Graph(edge_strategy)
        # Register Adjacency Matrix as a buffer (not a trainable parameter)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # ST-GCN Architecture (9 Layers)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, spatial_kernel_size, 1),
            STGCNBlock(64, 64, spatial_kernel_size, 1),
            STGCNBlock(64, 64, spatial_kernel_size, 1),
            STGCNBlock(64, 128, spatial_kernel_size, 2),
            STGCNBlock(128, 128, spatial_kernel_size, 1),
            STGCNBlock(128, 128, spatial_kernel_size, 1),
            STGCNBlock(128, 256, spatial_kernel_size, 2),
            STGCNBlock(256, 256, spatial_kernel_size, 1),
            STGCNBlock(256, 256, spatial_kernel_size, 1),
        ))

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # Input shape: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # Forward pass through GCN blocks
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)

        # Global Pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        
        # Prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x

    def load_pretrained_weights(self, weights_path):
        """
        Smart Loader: Loads Kinetics weights but ignores the shape-mismatch layers 
        (like the Graph Adjacency matrix) so it works on our 109-node setup.
        """
        print(f"Loading weights from {weights_path}...")
        pretrained_dict = torch.load(weights_path)
        model_dict = self.state_dict()

        # 1. Filter out unnecessary keys
        # We ignore 'A' (adjacency) because our graph is bigger.
        # We ignore 'fcn' (final layer) because we have 100 classes, not 400.
        filtered_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # 2. Update the current model
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)
        
        print(f"Weights loaded! {len(filtered_dict)} layers matched.")
        print(f"Skipped layers (retraining from scratch): {len(pretrained_dict) - len(filtered_dict)}")