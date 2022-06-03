import torch.nn as nn
import torch
class DINOHead(nn.Module):
    """Network hooked up to the CLS token embedding.
    Just a MLP with the last layer being normalized in a particular way.
    Parameters
    ----------
    in_dim : int
        The dimensionality of the token embedding.
    out_dim : int
        The dimensionality of the final layer (we compute the softmax over).
    hidden_dim : int
        Dimensionality of the hidden layers.
    bottleneck_dim : int
        Dimensionality of the second last layer.
    n_layers : int
        The number of layers.
    norm_last_layer : bool
        If True, then we freeze the norm of the weight of the last linear layer
        to 1.
    Attributes
    ----------
    mlp : nn.Sequential
        Vanilla multi-layer perceptron.
    last_layer : nn.Linear
        Reparametrized linear layer with weight normalization. That means
        that that it will have `weight_g` and `weight_v` as learnable
        parameters instead of a single `weight`.
    """
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, in_dim)`.
        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, out_dim)`.
        """
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.

    Parameters:
    backbone : vision transformer
        Instantiated Vision Transformer. Note that we will take the `head` attribute and replace it with `nn.Identity`.
    head : DINOHead
        New head that is going to be put on top of the `backbone`.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        '''
        The different crops are concatenated along the batch dimension and then a single forward pass is fun. The resulting tensor
        is then chunked back to per crop tensors.
        '''
        # convert to list
        if not isinstance(x, list):
            print('multicrop',x.shape)
            x = [x]
        n_crops=len(x)
        concatenated=torch.cat(x,dim=0)
        cls_embedding=self.backbone(concatenated)
        logits=self.head(cls_embedding)
        chunks=logits.chunk(n_crops)
        return chunks