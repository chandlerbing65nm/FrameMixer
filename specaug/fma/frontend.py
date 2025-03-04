import os
import torch
import torch.nn as nn
# from pydiffres import DiffRes as pydiffres
from specaug.fma.modules.fma import FrameMixer


class FMA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim, temp=0.2, Tr=0.0):
        super(FMA, self).__init__()
        self.model = FrameMixer(
            in_t_dim=in_t_dim,
            in_f_dim=in_f_dim,
            temp=temp, 
            Tr=Tr,
        )

    def forward(self, data):
        return self.model(data)

# if __name__ == "__main__":
#     # Test code
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Create a random spectrogram with dimensions [Batchsize, T-steps, F-bins]
#     data = torch.randn(32, 251, 128).to(device)

#     # Initialize the DiffRes
#     extractor = DiffRes(
#         in_t_dim=251,
#         in_f_dim=128,
#         dimension_reduction_rate=0.75,
#         learn_pos_emb=False
#     ).to(device)

#     # Process the data
#     data = extractor(data)

#     # Access the outputs
#     guide_loss = data["guide_loss"]
#     feature = data["feature"]
#     avgpool = data["avgpool"]
#     maxpool = data["maxpool"]
#     resolution_enc = data["resolution_enc"]

#     # Print the shapes of the outputs
#     print("Guide Loss:", guide_loss.item())
#     print("Feature Shape:", feature.shape)
#     print("AvgPool Shape:", avgpool.shape)
#     print("MaxPool Shape:", maxpool.shape)
#     print("Resolution Encoding Shape:", resolution_enc.shape)

#     # Example usage in a loss function
#     # loss = your_loss_function(output, target) + guide_loss

#     # Example usage in a classification task
#     # logits = your_classifier(feature)

