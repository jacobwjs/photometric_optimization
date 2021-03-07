import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np




class StyleEncoder(nn.Module):
    def __init__(self, dims_in=(18,512), dims_out=(18,32), one_hot=True, num_classes=6):
        super(StyleEncoder, self).__init__()
        
        # Example of classes.
        #
        # shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
        
        self.one_hot = one_hot
#         if one_hot:
#             dims_in = (dims_in[0], dims_in[-1] + num_classes)
            
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.style_dim = dims_in[0] # 18
        
        # Build up the independent linear layers.
        #
        self.layers = [nn.Linear(dims_in[-1], dims_out[-1]) \
                       for _ in range(0, dims_in[0])]
        
        # Using ModuleList so that this layer list can be moved to CUDA                      
        self.layers = torch.nn.ModuleList(self.layers)
        
    def forward(self, x, labels=None):
        ''' 
        Pass each style dimension from w+ through the
        independent linear mapping network
        
        args:
            x: w+ used with StyleGAN(2); shape=(N, 18, 512)
            labels: one hot vector for classes; shape=(N, 1, num_classes)
        returns:
            outputs: independent linear mapping to lower dimensional
                     space; dims=(18,32)
        '''
        outputs = []
        
        if self.one_hot:
            labels = labels.to(dtype=x.dtype)
            labels = labels.repeat(1, self.style_dim, 1)
            x = torch.cat((x, labels), dim=-1)
        
        for idx, layer in enumerate(self.layers):
            out = layer(x[:,idx,:])
            out = out.unsqueeze(1) # [N,32] -> [N,1,32]
            outputs.append(out)
        
        outputs = torch.cat(outputs, dim=1)
#         print("\noutputs: ", outputs.shape)
        
        return outputs
        
        
        
class StyleDecoder(nn.Module):
    def __init__(self, dims_in, dims_out=(18, 512)):
        super(StyleDecoder, self).__init__()
        
        self.dims_in = dims_in
        self.dims_out = dims_out
        
        ### TODO:
        ### - Update this to produce the correct results based on dimensions
        ###   of input and output.
        ###
        self.layers = [nn.Linear(dims_in[-1], dims_out[-1]) \
                       for _ in range(0, dims_in[0])]
        
        # Using ModuleList so that this layer list can be moved to CUDA
        #
        self.layers = torch.nn.ModuleList(self.layers)
        

    def forward(self, w_enc, params):
        ''' 
        Concatenate x and p and then decode.
        
        Args:
            w_enc: Encoded latent
            params: 3DMM params
        
        Returns:
            Decoded output
        ''' 
        # Flatten lighting parameter to match dimensions of other parameters in list
        # so that we can flatten all parameters to use as input to linear layers.
        #
        batch_size = w_enc.shape[0]
        params = list(params)
        params[-1] = params[-1].view(batch_size, -1) # [B,9,3] -> [B,27]
        
        
        # Flatten all parameters and repeat for each style dimension.
        #
        params_flat = torch.cat(params, dim=-1)
        style_dims = self.dims_in[0] # 18
        params_flat = params_flat.unsqueeze(1).expand(-1, style_dims, -1) 
        
        
        # Concatenate output from encoder with parameters we are "mixing" with.
        #
        x = torch.cat((w_enc, params_flat), dim=-1)

        
        # Treat each "style_dim" independently, therefore pass each
        # through an independent linear layer separately and concatenate to
        # form final latent for use with generator.
        #
        outputs = []
        for idx, layer in enumerate(self.layers):
            out = layer(x[:,idx,:])
            out = out.unsqueeze(1) # [B,512] -> [B,1,512]
            outputs.append(out)
    
        outputs = torch.cat(outputs, dim=1) # [B,18,512]
        return outputs
    
    

class RigNet(nn.Module):
    def __init__(self,
        dim_enc_in=(18,512),
        dim_enc_out=(18,32),
        dim_dec_in=(18, 268),
        dim_dec_out=(18, 512),
        one_hot=False,
        num_classes=6
    ):
        
        super(RigNet, self).__init__()

        # If conditionally creating parameters, we add in the labels to the size
        # of the input to the decoder. That is, we are concatenating one-hot labels
        # to the input of each linear model in the decoder.
        #
        self.one_hot = one_hot
        if one_hot:
            dim_enc_in = (dim_enc_in[0], dim_enc_in[-1] + num_classes)
        
        # Instantiate the encoder, passing optional one-hot labels if we are
        # conditionally transferring parameters.
        #
        self.encoder = StyleEncoder(dim_enc_in, dim_enc_out, one_hot, num_classes)
        
        self.decoder = StyleDecoder(dim_dec_in, dim_dec_out)
        
        
    def forward(self, latent, params, labels=None):
        '''
        Encode, "mix" w/ 3DMM params, and decode.
        
        Args:
            latent: Latent from mapping network of StyleGAN(2) generator.
            params: 3DMM params from DFR.
            labels (optional): 1-hot labels of size 'num_classes'
            
        Returns:
            output: Latent from "mixing" 3DMM with latent
        '''
        x = self.encoder(latent, labels)
        x = self.decoder(x, params)
        output = torch.add(x, latent)
        
        return output    


