import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, feat_dim, out_dim, hidden_dims=[]):
        super(MLP, self).__init__()
        layers = []
        in_dim = feat_dim
        for i, out_dim in enumerate(hidden_dims + [out_dim]):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(hidden_dims):
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.model = nn.Sequential(*layers)
        assert len(self.enc_params) + len(self.clf_params) \
            == len([_ for _ in self.parameters()])
    
    @property
    def enc_params(self):
        return [p for p in self.model.parameters()][:-2]
    
    @property
    def clf_params(self):
        return [p for p in self.model.parameters()][-2:]
        
    def forward(self, X):
        return self.model(X)