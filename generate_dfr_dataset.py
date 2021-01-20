
import os, sys
import argparse
import numpy as np 
from numpy import inf
import yaml

from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import face_alignment

from PIL import Image
from skimage import io
import cv2


def load_facesegment(base_path, device):
    '''
    Load model that is responsible for segmenting face
    from image.
    Used to form image masks.
    '''    
    sys.path.append(f'{base_path}')
    from Face_Seg.models import LinkNet34

    path_faceseg_repo = f'{base_path}/Face_Seg'
    face_seg = LinkNet34();
    face_seg.load_state_dict(torch.load(f'{path_faceseg_repo}/linknet.pth'));
    face_seg.eval();
    face_seg.to(device);
    return face_seg



def load_generator(base_path, device):
    sys.path.append(f'{base_path}/stylegan2_pytorch')
#     from model import Generator
    from stylegan2_pytorch.model import Generator
    
    # Create a directory and mount bucket.
    #
#     print("Bucket name: ", bucket_name)

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
    device = 'cuda'
    checkpoint_sg2 = path_to_weights_ffhq_sg2
    print("Loading weights... ", checkpoint_sg2)
    g_ema = Generator(1024, 512, 8) # (resolution, latent_dim, mapping_layers)
    g_ema.load_state_dict(torch.load(checkpoint_sg2)['g_ema'], strict=False)
    g_ema = g_ema.to(device)
    g_ema.eval();
    
    return g_ema




def load_3ddfa(args, base_path):
    '''
    Responsible for 3d landmarks, among other possibilities
    '''
    path_3ddfa = f'{base_path}/3DDFA_V2'
    sys.path.append(path_3ddfa)
    from Landmark3d import Landmark3dModel

    
#     sys.path.append(path_3ddfa)
#     from FaceBoxes import FaceBoxes
#     from TDDFA import TDDFA
# #     # from utils.render import render
# #     from utils.depth import depth
# #     from utils.pncc import pncc
# #     from utils.uv import uv_tex
# #     from utils.pose import viz_pose
# #     from utils.serialization import ser_to_ply, ser_to_obj
# #     from utils.functions import draw_landmarks, cv_draw_landmark, get_suffix
#     print("Loading 3DDFA_v2 model to ...", device)
#     print("\tconfig=", cfg)
#     face_boxes = FaceBoxes()
#     tddfa = TDDFA(gpu_mode=False, **cfg)

    cfg = yaml.load(open(f'{path_3ddfa}/configs/mb1_120x120.yml'),
                    Loader=yaml.SafeLoader)
#      'checkpoint_fp': 'weights/mb1_120x120.pth',
#      'bfm_fp': 'configs/bfm_noneck_v3.pkl',
    cfg['checkpoint_fp'] = f'{path_3ddfa}/weights/mb1_120x120.pth'
    cfg['bfm_fp'] = f'{path_3ddfa}/configs/bfm_noneck_v3.pkl'
    lmk_model3d = Landmark3dModel(**cfg)
    
    return lmk_model3d
    
    

    
    
def generate_train_data(args, g_ema,
                        seg_model,
                        lmk_model2d,
                        lmk_model3d,
                        dims=(224, 224),
                        **kwargs):
    
    
    
    # Get SG2 generated image that we will learn the rendering from.
    # -----------------------------------
    #

    # Generate data in batches, otherwise we quickly exhaust RAM.
    #
    dims = (224, 224)
    batches = args.size // args.batch_size
    pbar = tqdm(range(0, batches), dynamic_ncols=True, smoothing=0.01)
    for i in pbar:
        with torch.no_grad():
            images = []
            batch_latents = g_ema.get_latent(torch.randn((args.batch_size,
                                                          18,
                                                          512)).to(device)).to('cpu')
            
            imgs_gen, _ = g_ema([batch_latents.to(device)], **kwargs)
            imgs_gen = imgs_gen.cpu()
            images.append(normalize_float_img(imgs_gen))
            
            # Downsample images to size (i.e. dims) for rendering optimization expects.
            #
            images = torch.cat(images, dim=0)
            images = F.interpolate(images, dims)

            # Form masks that segment out only the face, as well as getting
            # 2d and 3d landmarks to be used as "ground truth".
            #
            image_masks, landmarks_2d_gt, landmarks_3d_gt = get_masks_landmarks(args,
                                                                                images,
                                                                                seg_model,
                                                                                lmk_model2d,
                                                                                lmk_model3d)
    

            for j, (latent, landmark_2d, landmark_3d, image, mask) in enumerate(zip(batch_latents,
                                                                    landmarks_2d_gt,
                                                                    landmarks_3d_gt,
                                                                    images,
                                                                    image_masks)):
                example = None
                example = {
                    'latents': latent,
                    'landmarks_2d_gt': landmark_2d,
                    'landmarks_3d_gt': landmark_3d,
                    'images': image,
                    'image_masks': mask
                }

                idx = (i * args.batch_size) + j
                filename = str(idx).zfill(6) + '.pkl'
#                 print("writing: ", filename)
                torch.save(example, f'{path_train_data}/{filename}')
                
        


def get_masks_landmarks(args, images, seg_model, lmk_model2d, lmk_model3d):
    '''
    Run various models against the generated images to produce representative data.
    
    Args:
        images: StyleGAN2 generated images.
        seg_model: Segments face from rest of image (e.g. removes hair, background, etc.)
        lmk_2d_model: "face_alignment" model. Provides 3d capabilities, but 3DDFA_v2 improves on
                      them, so only using for 2d landmarks (which 3DDFA_v2 does not provide).
        lmk_3d_model: 3DDFA_v2 model (supports 3d landmarks, pose, depth, etc.)
        
    Returns:
        image_masks: binary mask for face only in image
        landmarks_2d_gt: 2d landmarks for all images. Serves as "ground truth" in training regressor.
        landmarks_3d_gt: 3d landmarks for all images. Serves as "ground truth" in training regressor.
    '''

    # Get the face masks and landmarks from generated image.
    # -----------------------------------
    #
    image_masks = []
    landmarks_2d_gt = []
    landmarks_3d_gt = []
    imgs_lndmrks = []

    
    # Batch up the generated images to send to the face segmentation model to produce masks.
    # Choose batch size that won't exhaust memory.
    #
    dataloader_images = DataLoader(images,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers)
    
#     pbar = tqdm(dataloader_images, dynamic_ncols=True, smoothing=0.01)
#     for i, imgs in enumerate(pbar):
    for i, imgs in enumerate(dataloader_images):
        images_to_face_seg = to_faceseg_tensor(imgs, dims)
        masks = face_seg(images_to_face_seg.to(device))
        masks = masks.detach().cpu() # Save GPU RAM
        masks = masks > 0.8 
        
        
        # Form the binary mask, shape=[N,1,H,W].
        # 
        masks_bn = torch.zeros_like(masks).to(dtype=torch.float)
        masks_bn[torch.where(masks != 0)] = 1.
        image_masks.append(masks_bn)
        
        
        # Form "ground truth" landmarks from each generated image.
        #
        ### TODO:
        ### - 3DDFA_v2 doesn't seem to support batching.
        ### - Update to allow batches to 'face_boxes' and the rest.
        ###
        for img in imgs:
            img = img.clone() * 255.
            img = img.to(dtype=torch.uint8)
            img = img.squeeze().permute(1,2,0).numpy()

            
            # 2d landmarks
            # ------------------------------
            #
            gt_landmarks_2d = lmk_model2d.get_landmarks(img)
            assert len(gt_landmarks_2d) is not 0
            landmark_2d = gt_landmarks_2d[-1]
            
            # Normalize landmarks to image size (i.e. [-1, 1])
            #
            landmark_2d[:, 0] = landmark_2d[:, 0] / float(img.shape[1]) * 2 - 1
            landmark_2d[:, 1] = landmark_2d[:, 1] / float(img.shape[0]) * 2 - 1
            landmarks_2d_gt.append(torch.from_numpy(landmark_2d)[None, :, :].float())


            # 3d landmarks
            # ------------------------------
            #
#             boxes = face_boxes(img)
#             assert boxes is not None
            
#             # regress 3DMM params
#             #
#             with torch.no_grad():
#                 param_lst, roi_box_lst = tddfa(img, boxes)
            landmark_3d = lmk_model3d(img)
    
#             # reconstruct vertices and visualizing sparse landmarks
#             #
#             dense_flag = False
#             ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
#             landmark_3d = ver_lst[0].T # shape = (68,3)
            
    #         print("landmark_3d: ", landmark_3d.shape)
    #         img_landmarks_3d = cv_draw_landmark(img[:,:,:], landmark_3d.T, box=None, size=1)
    #         imgs_lndmrks.append(img_landmarks_3d)

            ### TODO:
            ### - We don't know anything about depth dimension, so how do we scale this?
            ###   Sticking to scaling x,y coords for the time being.
            ###
            # Normalize landmarks to image size (i.e. [-1, 1])
            #
            landmark_3d[:, 0] = landmark_3d[:, 0] / float(img.shape[1]) * 2 - 1
            landmark_3d[:, 1] = landmark_3d[:, 1] / float(img.shape[0]) * 2 - 1
            landmarks_3d_gt.append(torch.from_numpy(landmark_3d)[None, :, :].float().to(device))
            


    image_masks = torch.cat(image_masks, dim=0)
    landmarks_2d_gt = torch.cat(landmarks_2d_gt, dim=0)
    landmarks_3d_gt = torch.cat(landmarks_3d_gt, dim=0)
#     print("image_masks: ", image_masks.shape)
#     print("landmarks_2d_gt: ", landmarks_gt.shape)
    
    return image_masks, landmarks_2d_gt, landmarks_3d_gt

        
        
def normalize_float_img(x):
#     if len(x.size()) > 3:
#         x = x.squeeze()
    return x.mul_(127.5/255.).add_(0.5)



def convert_to_uint8(images):
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    scale = 255 / 2
    images = images.mul(scale) \
            .add_(0.5 + scale) \
            .clamp(0, 255) \
            .permute(0, 2, 3, 1) \
            .to('cpu', torch.uint8) \
            .squeeze()
    return images



def to_faceseg_tensor(x, dims=(256,256)):
    x = F.interpolate(x, dims)
    if len(x.size()) > 3:
        # Batched.
        #
        _imgs = []
        for img in x:
            img = transform_to_faceseg(img.squeeze())
            _imgs.append(img.unsqueeze(0)) # Return batch dim [B,C,H,W]
        x = torch.cat(_imgs, dim=0)
    else:
        x = transform_to_faceseg(x)
        
    return x




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
#     parser.add_argument("--lndmrk_2d", type=bool, default=True)
#     parser.add_argument("--lndmrk_3d", type=bool, default=False)
    parser.add_argument("--truncation", type=float,
                        help="truncation value for stylegan generator",
                        default=0.5)
    parser.add_argument("--workers", type=int,
                        help="number of parallel workers for Dataloader",
                        default=4)
    parser.add_argument("--save_dir", type=str,
                        help="path to directory where to save dataset",
                        default=None)
    parser.add_argument("--base_path", type=str,
                        help="top level directory where all repos, etc. live",
                        default=None)

    
    args = parser.parse_args()
    device = 'cuda'
    
    
    transform_to_faceseg = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225], inplace=False)
    ])
    transform_uint8 = transforms.Compose([
        # transforms.Resize(256),
        transforms.Lambda(convert_to_uint8)
    ])
    
    
    # Instantiate models - generation, segmentation, landmarks 2d, landmarks 3d
    #
    tddfa = load_3ddfa(args, args.base_path)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                      flip_input=False)
    face_seg = load_facesegment(args.base_path, device)
    g_ema = load_generator(args.base_path, device)
    
    
    
    if args.save_dir is None:
        path_train_data = f'{base_path}/training_data_dfr'
    else:
        path_train_data = args.save_dir
        
        
    if not os.path.exists(path_train_data):
        os.makedirs(path_train_data, exist_ok=True)
    
    w_mean = g_ema.mean_latent(int(10e3)) # For truncation.
    noise = g_ema.make_noise()
    kwargs = {
        'truncation_latent': w_mean,
        'truncation': args.truncation,
        'noise': noise,
        'randomize_noise': False,
        'input_is_latent': True,
        'return_latents': False
    }
    dims = (224, 224)
    generate_train_data(args, g_ema, face_seg, fa, tddfa, dims, **kwargs)
    
    
#     # No longer need generator. Free memory before loading other models.
#     #
#     del g_ema
    
    
    