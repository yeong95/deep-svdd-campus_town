from .efficientnet import EfficientNetAutoEncoder
import torch 
import torch.nn as nn 

from base.base_net import BaseNet


class Efficientnet_encoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.model = EfficientNetAutoEncoder.from_pretrained('efficientnet-b0')
        self.rep_dim = 100
        

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = x.flatten(start_dim=1)
        x = nn.Linear(x.shape[1], self.rep_dim, bias=False)(x)

        return x

class Efficientnet_autoencoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.model = EfficientNetAutoEncoder.from_pretrained('efficientnet-b0')
       
    def forward(self, inputs):
        ae_output, _= self.model(inputs)        
        return ae_output

if __name__ =='__main__':
    inputs = torch.rand(1,3,640,640)
    model = EfficientNetAutoEncoder.from_pretrained('efficientnet-b0')
    model.eval()
    ae_output, latent_fc_output = model(inputs)
    import pdb;pdb.set_trace()