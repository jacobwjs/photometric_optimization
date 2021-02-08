import numpy as np
import torch
import torch.nn.functional as F
import os
import sys


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


def facesegment(base_path):
    '''
    Load model that is responsible for segmenting face
    from image.
    Used to form image masks.
    '''    
    sys.path.append(f'{base_path}')
    from Face_Seg.models import LinkNet34
    
#     print("Loading face segmentation to device: ", device)

    path_faceseg_repo = f'{base_path}/Face_Seg'
    face_seg = LinkNet34();
    face_seg.load_state_dict(torch.load(f'{path_faceseg_repo}/linknet.pth'));
#     face_seg = face_seg.to(device);
    face_seg.eval();

    return face_seg


def dfr(base_path, load_weights=False, train=False):
    sys.path.append(f'{base_path}/photometric_optimization/models')
    from models.DFR_regressor import DFRParamRegressor
    
#     print("Loading DFR regression model to device: ", device)

    model_name = 'dfr_ckpt_epoch000499.pt'
    local_path_to_model = f'{base_path}/pretrained_models' 
    path_to_weights_dfr = f'{local_path_to_model}/{model_name}'
    
    # Initialize generator 
    #
    dfr = DFRParamRegressor()
    
    if load_weights:
        print("Loading weights... ", checkpoint_dfr)
        checkpoint_dfr = path_to_weights_dfr
        dfr.load_state_dict(torch.load(checkpoint_dfr)['dfr'], strict=False)
    
#     dfr = dfr.to(device)
    if train:
        dfr.train();
    else:
        dfr.eval();
    
    return dfr



def rignet(base_path, load_weights=False, train=False, one_hot=False):
    sys.path.append(f'{base_path}/photometric_optimization/models')
    from models.RigNet import RigNet    
#     print("Loading RigNet model to device: ", device)
     
    
    # Initialize model 
    #
    notification = "Creating RigNet model..."
    if one_hot:
        notification += "ONE_HOT"
    
    print(notification)
    rignet = RigNet(one_hot=one_hot)
    if load_weights:
        print("Loading weights... ", path_to_weights_rignet)
        model_name = 'dfr_ckpt_epoch000499.pt'
        local_path_to_model = f'{base_path}/pretrained_models' 
        path_to_weights_rignet = f'{local_path_to_model}/{model_name}'
        rignet.load_state_dict(torch.load(path_to_weights_rignet)['rignet'], strict=False)

#     rignet = rignet.to(device)
    if train:
        rignet.train();
    else:
        rignet.eval();
    
    return rignet



def generator(base_path):
#     sys.path.append(f'{base_path}/stylegan2_pytorch')
    path_stylegan = f'{base_path}/stylegan2_pytorch'
    if path_stylegan not in sys.path:
        sys.path.insert(0, path_stylegan)
        sys.path.insert(0, f'{path_stylegan}/op')

    
    from model import Generator
#     from stylegan2_pytorch.model import Generator
    
#     print("Loading generator to device: ", device)

    # Structure of bucket in S3.
    #
    model_to_mount = '/stylegan2/pytorch'
    model_name = 'stylegan2-ffhq-config-f.pt'

    # Check if we have copied the weights locally already after mounting.
    # Usually the case when multiple starts and stops of a VM have occurred.
    #
    if os.path.exists(f'{base_path}/pretrained_models_local'):
        local_path_to_model = f'{base_path}/pretrained_models_local'
    else:
        local_path_to_model = f'{base_path}/pretrained_models' 

#     if not os.path.exists(local_path_to_model):
#         !mkdir -p $local_path_to_model
        
    path_to_weights_ffhq_sg2 = f'{local_path_to_model}/{model_name}'
    print("path to StyleGAN2 weights: ", path_to_weights_ffhq_sg2)
    
    # Initialize generator 
    #
    checkpoint_sg2 = path_to_weights_ffhq_sg2
    print("Loading weights... ", checkpoint_sg2)
    g_ema = Generator(1024, 512, 8) # (resolution, latent_dim, mapping_layers)
    g_ema.load_state_dict(torch.load(checkpoint_sg2)['g_ema'], strict=False)
#     g_ema = g_ema.to(device)
    g_ema.eval();
    
    return g_ema


def _3ddfa(args, base_path):
    '''
    Responsible for 3d landmarks, among other possibilities
    '''
    path_3ddfa = f'{base_path}/3DDFA_V2'
    if path_3ddfa not in sys.path:
        sys.path.append(path_3ddfa)
        
    from Landmark3d import Landmark3dModel

    cfg = yaml.load(open(f'{path_3ddfa}/configs/mb1_120x120.yml'),
                    Loader=yaml.SafeLoader)
#      'checkpoint_fp': 'weights/mb1_120x120.pth',
#      'bfm_fp': 'configs/bfm_noneck_v3.pkl',
    cfg['checkpoint_fp'] = f'{path_3ddfa}/weights/mb1_120x120.pth'
    cfg['bfm_fp'] = f'{path_3ddfa}/configs/bfm_noneck_v3.pkl'
    lmk_3d_model = Landmark3dModel(**cfg)
    
    return lmk_3d_model



def flame(base_path, config):
    _path = f'{base_path}/photometric_optimization/models'
    if _path not in sys.path:
        sys.path.append(_path)
        
    from FLAME import FLAME
    
    config = dict2obj(config)
    return FLAME(config)
    
    

def flametex(base_path, config):
    _path = f'{base_path}/photometric_optimization/models'
    if _path not in sys.path:
        sys.path.append(_path)
        
    from FLAME import FLAMETex
    
    config = dict2obj(config)
    return FLAMETex(config)


def renderer(base_path, config):
    _path = f'{base_path}/photometric_optimization/models'
    if _path not in sys.path:
        sys.path.append(_path)
    from renderer import Renderer
    
    config = dict2obj(config)
    mesh_file = config.head_template_mesh
    render = Renderer(config.image_size, obj_filename=mesh_file)
    
    return render
    