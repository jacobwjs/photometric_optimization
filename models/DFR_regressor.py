import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def _get_config():
    
    def dict2obj(d):
        if isinstance(d, list):
            d = [dict2obj(x) for x in d]
        if not isinstance(d, dict):
            return d

        class C(object):
            pass

        o = C()
        for k in d:
            o.__dict__[k] = dict2obj(d[k])
        
        return o


    config = {
#         # FLAME
#         'flame_model_path': './data/generic_model.pkl',  # acquire it from FLAME project page
#         'flame_lmk_embedding_path': './data/landmark_embedding.npy',
#         'tex_space_path': './data/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'light_params': [9,3],
        'use_face_contour': True,

#         'cropped_size': 256,
#         'batch_size': 1,
#         'image_size': 224,
#         'e_lr': 1e-4,
#         'e_wd': 0.0001,
#         'savefolder': './test_results',
#         # weights of losses and reg terms
#         'w_pho': 8,
#         'w_lmks': 1,
#         'w_shape_reg': 1e-4,
#         'w_expr_reg': 1e-4,
#         'w_pose_reg': 0,
    }
    
    return dict2obj(config)


class DFRParamRegressor(nn.Module):
    def __init__(self, config=None):
        super(DFRParamRegressor, self).__init__()
        
        if config is None:
            config = _get_config()
            
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
        

        dim1 = 512
        dim2 = 512

        self.fc1 = nn.Linear(input_size, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc3 = nn.Linear(dim2, output_size)
        
        self.dtype = torch.float32
#         self.register_buffer('fc1', fc1)
#         self.register_buffer('fc2', fc2)
#         self.register_buffer('fc3', fc3)
    
    ### TODO:
    ### - Update to use nn.Flatten()(x) on input, instead of having to 
    ###   flatten before moving through model.
    ### - Return dictionary with keys for each parameter.
    ###
    def forward(self, x):
        ### x = nn.Flatten()(x)
        x = self.fc1(x)
        x = nn.ELU()(x)
        x = self.fc2(x)
        x = nn.ELU()(x)
        x = self.fc3(x)

        
        # Return values split into shape, expression, pose
        #
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
        
