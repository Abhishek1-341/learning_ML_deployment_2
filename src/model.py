import torch 
import torch.nn as nn

class MyModel(nn.Module) :
    def __init__(self,input_dim) :
        super().__init__()
        self.simple_NN = nn.Sequential(
            nn.Linear(input_dim,8), nn.ReLU(),
            nn.Linear(8,8)        , nn.ReLU(),
            nn.Linear(8,1)        ,nn.Sigmoid()
        )

    def forward(self, X) :
        return self.simple_NN(X)