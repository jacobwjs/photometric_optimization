import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np




class StyleEncoder(nn.Module):
    def __init__(self, dims_in=(18,512), dims_out=(18,32)):
        super(StyleEncoder, self).__init__()
        
        # Example of classes.
        #
        # shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.style_dim = dims_in[0] # 18
        
        # Build up the independent linear layers.
        #
        self.layers = [nn.Linear(dims_in[-1], dims_out[-1]) \
                       for _ in range(0, dims_in[0])]
        
        # Using ModuleList so that this layer list can be moved to CUDA                      
        self.layers = torch.nn.ModuleList(self.layers)
        
    def forward(self, x):
        ''' 
        Pass each style dimension from w+ through the
        independent linear mapping network
        
        args:
            x: w+ used with StyleGAN(2); shape=(N, 18, 512)
        returns:
            outputs: independent linear mapping to lower dimensional
                     space; dims=(18,32)
        '''
        w_enc = []
        
        for idx, layer in enumerate(self.layers):
            out = layer(x[:,idx,:]).unsqueeze(1) # [N,32] -> [N,1,32]
            w_enc.append(out)
        
        w_enc = torch.cat(w_enc, dim=1)
#         print("\nw_enc: ", outputs.shape)
        
        return w_enc
        
        
        
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
        

    def forward(self, w_enc, pose):
        ''' 
        Concatenate x and p and then decode.
        
        Args:
            w_enc: Encoded latent
            pose: 3DMM pose parameter
        
        Returns:
            Decoded output
        ''' 
        
        # Repeat for each style dimension.
        #
        style_dims = self.dims_in[0] # 18
        pose = pose.unsqueeze(1).repeat(1, style_dims, 1)
        
        
        # Concatenate output from encoder with parameters we are "mixing" with.
        #
        x = torch.cat((w_enc, pose), dim=-1)

        
        # Treat each "style_dim" independently, therefore pass each
        # through an independent linear layer separately and concatenate to
        # form final latent for use with generator.
        #
        w_dec = []
        for idx, layer in enumerate(self.layers):
            out = layer(x[:,idx,:]).unsqueeze(1) # [B,512] -> [B,1,512]
            w_dec.append(out)
    
        w_dec = torch.cat(w_dec, dim=1) # [B,18,512]
        return w_dec
    
    

class RigNet(nn.Module):
    def __init__(self,
        dim_enc_in=(18,512),
        dim_enc_out=(18,32),
        dim_dec_in=(18, 38),
        dim_dec_out=(18, 512),
    ):
        
        super(RigNet, self).__init__()

        self.encoder = StyleEncoder(dim_enc_in, dim_enc_out)
        self.decoder = StyleDecoder(dim_dec_in, dim_dec_out)
        
        
    def forward(self, latent, params):
        '''
        Encode, "mix" w/ 3DMM params, and decode.
        
        Args:
            latent: Latent from mapping network of StyleGAN(2) generator.
            params: 3DMM params from DFR.
            
        Returns:
            output: Latent from "mixing" 3DMM with latent
        '''
        x = self.encoder(latent)
        x = self.decoder(x, params)
        output = torch.add(x, latent)
        
        return output    


