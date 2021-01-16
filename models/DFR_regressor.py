import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



class DFRParamRegressor(nn.Module):
    def __init__(self, config):
        super(DFRParamRegressor, self).__init__()
        
        input_size = 18*512 # W+ dims from StyleGAN(2)
        
        output_size = config.shape_params + \
                      config.expression_params + \
                      config.pose_params + \
                      config.tex_params + \
                      config.camera_params + \
                      np.prod(config.light_params)
        
    
#         output_size = config.shape_params + \
#                       config.expression_params + \
#                       config.pose_params
        
        # Used to index output.
        #
        self.config = config
        self.shape_idx = config.shape_params
        self.expression_idx = config.expression_params
        self.pose_idx = config.pose_params
        self.tex_idx = config.tex_params
        self.cam_idx = config.camera_params
        self.light_idx = np.prod(config.light_params) # Light params are typically [9,3]
        
        
        print("Output params dims: ", output_size)
        
       # dim1 = 1024
       # dim2 = 1024
        dim1 = 512
        dim2 = 512
#         dim1 = 256
#         dim2 = 256
#        dim1 = 128
#        dim2 = 128
 
#         dim1 = 2048
#         dim2 = 1024
       # dim1 = 1024
       # dim2 = 512
       # dim1 = 512
       # dim2 = 256
#        dim1 = 256
#        dim2 = 128

        self.fc1 = nn.Linear(input_size, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc3 = nn.Linear(dim2, output_size)
        
        self.dtype = torch.float32
#         self.register_buffer('fc1', fc1)
#         self.register_buffer('fc2', fc2)
#         self.register_buffer('fc3', fc3)
    
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ELU()(x)
        x = self.fc2(x)
        x = nn.ELU()(x)
        x = self.fc3(x)

        
        # Return values split into shape, expression, pose
        #
#         return torch.split(x, [self.shape_idx, self.expression_idx, self.pose_idx], dim=-1)

#         return torch.split(x, [self.shape_idx,
#                                self.expression_idx,
#                                self.pose_idx,
#                                self.tex_idx,
#                                self.cam_idx], dim=-1)

        idx_split = [
            self.shape_idx,
            self.expression_idx,
            self.pose_idx,
            self.tex_idx,
            self.cam_idx,
            self.light_idx
        ]
        shape, expr, pose, tex, cam, light =  torch.split(x, idx_split, dim=-1)
        # Reshape light params to expected form.
        # [B,9,3]
        #
        light = light.reshape(shape=[-1] + self.config.light_params) 
        
        return shape, expr, pose, tex, cam, light
        
