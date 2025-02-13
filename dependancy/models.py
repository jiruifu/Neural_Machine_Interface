import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool1d
class NeuralInterface_1D(nn.Module):
    def __init__(self, numChannels=64, classes=4, winlen=120, numNodes=[128, 128, 128, 64, 256]):
        """
        :param numChannels:
        :param classes:
        :param numNodes: number of nodes in hidden layer
        Structure of CNN: CONV1 => RELU => CONV2 => RELU => POOLING => DROPOUT
        """
        # Call the parent constructor
        super(NeuralInterface_1D, self).__init__()
        self.classes = classes
        self.channels = numChannels

        conv1 = torch.nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3)
        relu1 = torch.nn.ReLU()
        conv2 = torch.nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3)
        relu2 = torch.nn.ReLU()
        maxpool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        dropout2 = nn.Dropout(p=0.5)
        cnnBlock1 = nn.Sequential(conv1, relu1, conv2, relu2, maxpool2, dropout2)

        # initialize second set of CNN
        conv3 = torch.nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3)
        relu3 = torch.nn.ReLU()
        conv4 = torch.nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3)
        relu4 = torch.nn.ReLU()
        maxpool4 = MaxPool1d(kernel_size=2, stride=2)
        dropout4 = nn.Dropout(p=0.5)
        flatten = nn.Flatten()
        cnnBlock2 = nn.Sequential(conv3, relu3, conv4, relu4, maxpool4, dropout4, flatten)
        self.feature_extractor = nn.Sequential(cnnBlock1, cnnBlock2)

        dummy_input = torch.randn(1, numChannels, winlen)
        dummy_output = self.feature_extractor(dummy_input)
        n_features = dummy_output.numel() // dummy_output.size(0)

        self.outputs = nn.ModuleList()
        # For each motor unit, recognize if the spike is 1 or 0
        for _ in range(classes):
            self.outputs.append(nn.Sequential(
                nn.Linear(n_features, numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        deep_feature = self.feature_extractor(x)  # Flatten all dimensions except batch
        outputlist = []

        # Each class corresponds to a motor unit, for n classes, this FNN makes n predictions.
        for i in range(self.classes):
            outputlist.append(self.outputs[i](deep_feature))

        output = torch.cat(outputlist, dim=1)
        return output
        return torch.cat([output(features) for output in self.outputs], dim=1)


class NeuralInterface_3D(nn.Module):
    def __init__(self, numChannels, classes, size: tuple, numNodes=[32, 32, 32, 32, 256]):
        super().__init__()  # Simplified parent constructor call
        self.classes = classes
        self.channels = numChannels
        
        # First CNN block
        self.cnnblock1 = nn.Sequential(
            nn.Conv3d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(numNodes[0]),
            nn.Conv3d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(numNodes[1]),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )

        # Second CNN block
        self.cnnblock2 = nn.Sequential(
            nn.Conv3d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(numNodes[2]),
            nn.Conv3d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(numNodes[3]),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )
        
        # Calculate output size dynamically
        with torch.no_grad():
            x_t = torch.zeros(1, numChannels, *size)
            x1_t = self.cnnblock1(x_t)
            x2_t = self.cnnblock2(x1_t)
            flat_size = torch.flatten(x2_t, 1).shape[1]
        
        # Output layers
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_size, numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ) for _ in range(classes)
        ])
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.cnnblock1(x)
        x = self.cnnblock2(x)
        x = torch.flatten(x, 1)
        
        # More efficient list comprehension
        outputs = [layer(x) for layer in self.outputs]
        return torch.cat(outputs, dim=1)