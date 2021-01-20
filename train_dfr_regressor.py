import argparse


import os, sys
import numpy as np


import cv2
import torch
import torchvision
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


sys.path.append('./models/')
from DFR_regressor import DFRParamRegressor
from FLAME import FLAME, FLAMETex
from renderer import Renderer
import util


from tqdm import tqdm


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


def get_config():
    config = {
        # FLAME
        'flame_model_path': './data/generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': './data/landmark_embedding.npy',
        'tex_space_path': './data/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'light_params': [9,3],
        'use_face_contour': True,

        'cropped_size': 256,
        'batch_size': 1,
        'image_size': 224,
        'e_lr': 1e-4,
        'e_wd': 0.0001,
        'savefolder': './test_results',
        # weights of losses and reg terms
        'w_pho': 8,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }
    
    return config



def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier
    
    
def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()
    
    

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
    

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            
             
class DatasetDFR(data.Dataset):
    def __init__(self, path_to_dir):
        self.path = path_to_dir
        
    def __len__(self):
        import glob
        return len(glob.glob1(self.path, '*.pkl'))
    
    def __getitem__(self, index):
        filename = str(index).zfill(6) + '.pkl'
        x = torch.load(f'{self.path}/{filename}')
        return x
    
    

def save_checkpoint(path, epoch, model, losses=None):
    epoch = str(epoch).zfill(6)
    
    if losses is not None:
        torch.save(losses, f'{path}/losses_epoch{epoch}.pkl')
    
    model_data = {
        'dfr': model.cpu().state_dict(),
    }
    torch.save(model_data, f'{path}/dfr_ckpt_epoch{epoch}.pt')


    
def save_rendered_imgs(savefolder, epoch, images, predicted_images, shape_images, albedos, albedo_images,
                       landmarks_gt, landmarks2d, landmarks3d):
    grids = {}
    # visind = range(bz)  # [0]
    grids['images'] = torchvision.utils.make_grid(images).detach().cpu()
    grids['landmarks_gt'] = torchvision.utils.make_grid(
        util.tensor_vis_landmarks(images.clone().detach(), landmarks_gt))
    grids['landmarks2d'] = torchvision.utils.make_grid(
        util.tensor_vis_landmarks(images, landmarks2d))
    grids['landmarks3d'] = torchvision.utils.make_grid(
        util.tensor_vis_landmarks(images, landmarks3d))
    grids['albedoimage'] = torchvision.utils.make_grid(albedo_images)
    grids['render'] = torchvision.utils.make_grid(predicted_images.detach().float().cpu())
#     shape_images = render.render_shape(vertices, trans_vertices, images)
    grids['shape'] = torchvision.utils.make_grid(
        F.interpolate(shape_images, [224, 224])).detach().float().cpu()


    grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos, [224, 224])).detach().cpu()
    grid = torch.cat(list(grids.values()), 1)
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

    cv2.imwrite('{}/{}.jpg'.format(savefolder, str(epoch).zfill(6)), grid_image)
    
    
class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
    
    def forward(self, verts1, verts2):
        return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()
    
    

def train(args, config, loader, dfr, flame, flametex, render, tex_mean, device):
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H.%M.%S")     # dd/mm/YY H:M:S
    savefolder = os.path.sep.join(['./test_results', f'{dt_string}'])
    if not os.path.exists(savefolder):
        os.makedirs(savefolder, exist_ok=True)

        
#     lights = nn.Parameter(torch.zeros(args.batch_size, 9, 3).float().to(device))    
#     optim = torch.optim.Adam(
#                 list(dfr.parameters()) + [lights],
#                 lr=config.e_lr,
#                 weight_decay=config.e_wd
#     )

#     cam = torch.zeros(args.batch_size, config.camera_params).to(device)
#     cam[:, 0] = 5.0
#     optim = torch.optim.Adam(
#                 list(dfr.parameters()) + [cam],
#                 lr=config.e_lr,
#                 weight_decay=config.e_wd
#     )

    optim = torch.optim.Adam(
                dfr.parameters(),
                lr=1e-4,
                weight_decay=0.00001 # config.e_wd
    )

    
    loader = sample_data(loader)
#     pbar = range(args.iter)
    
    losses_to_plot = {}
    losses_to_plot['all_loss'] = []
    losses_to_plot['landmark_2d'] = []
    losses_to_plot['landmark_3d'] = []
    losses_to_plot['shape_reg'] = []
    losses_to_plot['shape_reg'] = []
    losses_to_plot['expression_reg'] = []
    losses_to_plot['pose_reg'] = []
    losses_to_plot['photometric_texture'] = []
    losses_to_plot['texture_reg'] = []
    
    
    loss_mse = nn.MSELoss()
    
    
    idx_rigid_stop = args.iter_rigid
    modulo_save_imgs = args.iter_save_img
    modulo_save_model = args.iter_save_chkpt
    
    
    pbar = tqdm(range(0, idx_rigid_stop), dynamic_ncols=True, smoothing=0.01)
    k = 0
    for k in pbar:
        for example in dataloader:
            latents = example['latents'].to(device)
            landmarks_2d_gt = example['landmarks_2d_gt'].to(device)
            images = example['images'].to(device)
            image_masks = example['image_masks'].to(device)


            shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            vertices, landmarks2d, landmarks3d = flame(shape_params=shape,
                                                       expression_params=expression,
                                                       pose_params=pose)


            trans_vertices = util.batch_orth_proj(vertices, cam);
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]


            losses = {}
            losses['landmark_2d'] = util.l2_distance(landmarks2d[:, 17:, :2],
                                                     landmarks_2d_gt[:, 17:, :2]) * config.w_lmks
            
#             losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * 1e-4 #config.w_pose_reg


            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
                losses_to_plot[key].append(losses[key].item()) # Store for plotting later.


            losses['all_loss'] = all_loss
            losses_to_plot['all_loss'].append(losses['all_loss'].item())


            optim.zero_grad()
            all_loss.backward()
            optim.step()


            pbar.set_description(
                (
                    f"total: {losses['all_loss']:.4f}; landmark_2d: {losses['landmark_2d']:.4f}; "
                )
            )

            
            if (k % modulo_save_imgs == 0):
                try:
                    grids = {}
                    grids['images'] = torchvision.utils.make_grid(images.detach().cpu())
                    grids['landmarks_2d_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images, landmarks_2d_gt))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images, landmarks2d))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images, landmarks3d))

                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                    cv2.imwrite('{}/{}.jpg'.format(savefolder, str(k).zfill(6)), grid_image)
                except:
                    print("Error saving images... continuing")
                    continue
                
            if k % modulo_save_model == 0:
                save_checkpoint(path=savefolder,
                                epoch=k+1,
                                losses=losses_to_plot,
                                model=dfr)   
    
    # Save final epoch for rigid fitting.
    #
    if k > 0:
        save_checkpoint(path=savefolder,
                        epoch=k+1,
                        losses=losses_to_plot,
                        model=dfr)
    
    
    # Second stage training. Adding in photometric loss.
    #
    pbar = tqdm(range(idx_rigid_stop, args.iter), dynamic_ncols=True, smoothing=0.01)
    for k in pbar:
        for example in dataloader:
            latents = example['latents'].to(device)
            landmarks_2d_gt = example['landmarks_2d_gt'].to(device)
            landmarks_3d_gt = example['landmarks_3d_gt'].to(device)
            images = example['images'].to(device)
            image_masks = example['image_masks'].to(device)


#             shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            vertices, landmarks2d, landmarks3d = flame(shape_params=shape,
                                                       expression_params=expression,
                                                       pose_params=pose)


            trans_vertices = util.batch_orth_proj(vertices, cam);
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]


            losses = {}
            
#             if k < 250:
#                 losses['landmark_2d'] = util.l2_distance(landmarks2d[:, 17:, :2],
#                                                       landmarks_2d_gt[:, 17:, :2]) * 2.0 #config.w_lmks
#             else:
#                 losses['landmark_2d'] = util.l2_distance(landmarks2d[:, :, :2],
#                                                       landmarks_2d_gt[:, :, :2]) * 2.0
            losses['landmark_2d'] = util.l2_distance(landmarks2d[:, :, :2],
                                                      landmarks_2d_gt[:, :, :2]) * 2.0
    
            losses['landmark_3d'] = util.l2_distance(landmarks3d[:, :, :2],
                                                      landmarks_3d_gt[:, :, :2]) * 1.0
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(expression ** 2) / 2) * config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * config.w_pose_reg


            ## render
            albedos = flametex(tex) / 255.
            losses['texture_reg'] = loss_mse(albedos, tex_mean.repeat(args.batch_size, 1, 1, 1)) #* 1e-3 # Regularize learned texture.
            ops = render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['photometric_texture'] = (image_masks * (predicted_images - images).abs()).mean() \
                                            * config.w_pho


            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
                losses_to_plot[key].append(losses[key].item()) # Store for plotting later.

            losses['all_loss'] = all_loss
            losses_to_plot['all_loss'].append(losses['all_loss'].item())


            optim.zero_grad()
            all_loss.backward()
            optim.step()
#             scheduler.step(all_loss)


            pbar.set_description(
                (
                    f"total: {losses['all_loss']:.4f}; landmark_2d: {losses['landmark_2d']:.4f}; "
                    f"landmark_3d: {losses['landmark_3d']:.4f}; "
                    f"shape: {losses['shape_reg']:.4f}; express: {losses['expression_reg']:.4f}; "
                    f"photo: {losses['photometric_texture']:.4f}; "
                )
            )


            # visualize
            if k % modulo_save_imgs == 0:
                shape_images = render.render_shape(vertices, trans_vertices, images)
                save_rendered_imgs(savefolder, k, images, predicted_images, shape_images,
                                   albedos, ops, landmarks_2d_gt, landmarks2d, landmarks3d)
#                 try:
# #                     grids = {}
# #     #                 visind = range(bz)  # [0]
# #                     grids['images'] = torchvision.utils.make_grid(images).detach().cpu()
# #                     grids['landmarks_gt'] = torchvision.utils.make_grid(
# #                         util.tensor_vis_landmarks(images.clone().detach(), landmarks_gt))
# #                     grids['landmarks2d'] = torchvision.utils.make_grid(
# #                         util.tensor_vis_landmarks(images, landmarks2d))
# #                     grids['landmarks3d'] = torchvision.utils.make_grid(
# #                         util.tensor_vis_landmarks(images, landmarks3d))
# #                     grids['albedoimage'] = torchvision.utils.make_grid(
# #                         (ops['albedo_images']).detach().cpu())
# #                     grids['render'] = torchvision.utils.make_grid(predicted_images.detach().float().cpu())
# #                     shape_images = render.render_shape(vertices, trans_vertices, images)
# #                     grids['shape'] = torchvision.utils.make_grid(
# #                         F.interpolate(shape_images, [224, 224])).detach().float().cpu()


# #                     grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos, [224, 224])).detach().cpu()
# #                     grid = torch.cat(list(grids.values()), 1)
# #                     grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
# #                     grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

# #                     cv2.imwrite('{}/{}.jpg'.format(savefolder, str(k).zfill(6)), grid_image) 

#                     shape_images = render.render_shape(vertices, trans_vertices, images)
#                     save_rendered_imgs(savefolder, k, images, predicted_images, shape_images, albedos, ops,
#                                        landmarks_gt, landmarks2d, landmarks3d)
#                 except:
#                     print("Error saving images and renderings... continuing")
#                     continue
        
            
            if k % modulo_save_model == 0:
                save_checkpoint(path=savefolder,
                                epoch=k+1,
                                losses=losses_to_plot,
                                model=dfr)
    
    
    # Save final epoch renderings and checkpoints.
    #
    shape_images = render.render_shape(vertices, trans_vertices, images)
    save_rendered_imgs(savefolder, k+1, images, predicted_images, shape_images, albedos, ops,
                       landmarks_2d_gt, landmarks2d, landmarks3d)
    
    
    save_checkpoint(path=savefolder,
                    epoch=k+1,
                    losses=losses_to_plot,
                    model=dfr)
        
    print("cam: ", cam)
    print("landmarks3d.mean: ", landmarks3d.mean())
    print("landmarks3d.min: ", landmarks3d.min())
    print("landmarks3d.max: ", landmarks3d.max())
    
        
        
        

### TODO:
### - Does not work
###
def train_distributed(args, config):
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H.%M.%S")     # dd/mm/YY H:M:S
    savefolder = os.path.sep.join(['./test_results', f'{dt_string}'])
    if not os.path.exists(savefolder):
        os.makedirs(savefolder, exist_ok=True)
    
    
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://'
    )  
    
    args.local_rank = torch.distributed.get_rank()
    
    # this is the total # of GPUs across all nodes
    # if using 2 nodes with 4 GPUs each, world size is 8
    args.world_size = torch.distributed.get_world_size()
    print("### global rank of curr node: {} of {}".format(torch.distributed.get_rank(),
                                                         torch.distributed.get_world_size()))
    
    
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    #
    torch.cuda.set_device(args.local_rank)

    
    dfr = DFRParamRegressor(config)
    if args.ckpt is not None:
        dfr.load_state_dict(torch.load(args.ckpt)['dfr'], strict=False)
        dfr.train();
    
    
    flame = FLAME(config)
    flametex = FLAMETex(config)
    mesh_file = f'{args.flame_data}/head_template_mesh.obj'
    render = Renderer(config.image_size, obj_filename=mesh_file)
    
    
    dfr.cuda(args.local_rank)
    flame.cuda(args.local_rank)
    flametex.cuda(args.local_rank)
    render.cuda(args.local_rank)

    
    texture_mean = flametex.get_texture_mean(args.batch_size) / 255.
    texture_mean = texture_mean.cuda(args.local_rank)
    
    
    batch_size = args.batch_size
    workers = args.workers
    
    
    dfr = torch.nn.parallel.DistributedDataParallel(
        dfr,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )
#     flame = torch.nn.parallel.DistributedDataParallel(
#         flame,
#         device_ids=[args.local_rank],
#         output_device=args.local_rank
#     )
#     flametex = torch.nn.parallel.DistributedDataParallel(
#         flametex,
#         device_ids=[args.local_rank],
#         output_device=args.local_rank
#     )
#     render = torch.nn.parallel.DistributedDataParallel(
#         render,
#         device_ids=[args.local_rank],
#         output_device=args.local_rank
#     )
    
    
    loss_l2 = LossL2() #.cuda(args.local_rank)
    loss_ce = nn.CrossEntropyLoss() #.cuda(args.local_rank)
    optim = torch.optim.Adam(
                dfr.parameters(),
                lr=1e-4,
                weight_decay=0.00001 # config.e_wd
    )
    
    
    dataset = DatasetDFR(args.train_data)
    # makes sure that each process gets a different slice of the training data
    # during distributed training.
    #
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.cuda.device_count(),
#             num_replicas=4,
#             rank = args.local_rank,
        )
        
    
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
#         pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    
    
    idx_rigid_stop = args.iter_rigid
    modulo_save_imgs = args.iter_save_img
    modulo_save_model = args.iter_save_chkpt
    
    # Start optimization
    # -----------------------------
    pbar = tqdm(range(0, idx_rigid_stop), dynamic_ncols=True, smoothing=0.01)
#     pbar = range(0, idx_rigid_stop)
    k = 0
    for k in pbar:
        if args.distributed:
            sampler.set_epoch(k)
        
        
        for example in loader:
            latents = example['latents'].cuda(args.local_rank)
            landmarks_2d_gt = example['landmarks_2d_gt'].cuda(args.local_rank)
            images = example['images'].cuda(args.local_rank)
            image_masks = example['image_masks'].cuda(args.local_rank)
#             print("2 step in rank: ", args.local_rank, flush=True)
#             print(latents.shape)
#             print(landmarks_2d_gt.shape)
#             print(images.shape)
#             print(image_masks.shape)


            shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            vertices, landmarks2d, landmarks3d = flame(shape_params=shape,
                                                       expression_params=expression,
                                                       pose_params=pose)


            trans_vertices = util.batch_orth_proj(vertices, cam); 
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]


            losses = {}
            losses['landmark_2d'] = loss_l2(
                landmarks2d[:, 17:, :2],
                landmarks_2d_gt[:, 17:, :2]
            ) * config.w_lmks


            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
#                 losses_to_plot[key].append(losses[key].item()) # Store for plotting later.


            losses['all_loss'] = all_loss
# #             losses_to_plot['all_loss'].append(losses['all_loss'].item())


            optim.zero_grad()
            all_loss.backward()
            optim.step()


#             if args.local_rank == 0:
            pbar.set_description(
                (
                    f"total: {losses['all_loss']:.4f}; landmark_2d: {losses['landmark_2d']:.4f}; "
                )
            )

            if args.local_rank == 0:
                if (k % modulo_save_imgs == 0):
#                     try:
#                         grids = {}
#                         grids['images'] = torchvision.utils.make_grid(images.detach().cpu())
#                         grids['landmarks_2d_gt'] = torchvision.utils.make_grid(
#                             util.tensor_vis_landmarks(images, landmarks_2d_gt))
#                         grids['landmarks2d'] = torchvision.utils.make_grid(
#                             util.tensor_vis_landmarks(images, landmarks2d))
#                         grids['landmarks3d'] = torchvision.utils.make_grid(
#                             util.tensor_vis_landmarks(images, landmarks3d))

#                         grid = torch.cat(list(grids.values()), 1)
#                         grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
#                         grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
#                         cv2.imwrite('{}/{}.jpg'.format(savefolder, str(k).zfill(6)), grid_image)
#                     except:
#                         print("Error saving images... continuing")
#                         continue
                    grids = {}
                    bsize = range(0,10)
                    grids['images'] = torchvision.utils.make_grid(images.detach().cpu()[bsize])
                    grids['landmarks_2d_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[bsize], landmarks_2d_gt))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[bsize], landmarks2d))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[bsize], landmarks3d))

                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                    cv2.imwrite('{}/{}.jpg'.format(savefolder, str(k).zfill(6)), grid_image)

                
#             if args.local_rank == 0:
#                 if k % modulo_save_model == 0:
#                     save_checkpoint(path=savefolder,
#                                     epoch=k+1,
# #                                     losses=losses_to_plot,
#                                     losses=None,
#                                     model=dfr)   
    
    
    train_render(args, dfr, render, flame, flametex,
                 texture_mean=texture_mean,
                 optim=optim,
                 savefolder=savefolder,
                 dataloader=loader)
        
        
def train_render(args, dfr, render, flame, flametex, texture_mean, optim, savefolder, dataloader):
#     # Save final epoch for rigid fitting.
#     #
#     if k > 0:
#         save_checkpoint(path=savefolder,
#                         epoch=k+1,
#                         losses=losses_to_plot,
#                         model=dfr)
    
    
    loss_mse = nn.MSELoss()
    
    
    idx_rigid_stop = args.iter_rigid
    modulo_save_imgs = args.iter_save_img
    modulo_save_model = args.iter_save_chkpt
    
    # Second stage training. Adding in photometric loss.
    #
    pbar = tqdm(range(idx_rigid_stop, args.iter), dynamic_ncols=True, smoothing=0.01)
    for k in pbar:
        for example in dataloader:
            latents = example['latents'].cuda(args.local_rank)
            landmarks_2d_gt = example['landmarks_2d_gt'].cuda(args.local_rank)
            landmarks_3d_gt = example['landmarks_3d_gt'].to(args.local_rank)
            images = example['images'].cuda(args.local_rank)
            image_masks = example['image_masks'].cuda(args.local_rank)


#             shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            vertices, landmarks2d, landmarks3d = flame(shape_params=shape,
                                                       expression_params=expression,
                                                       pose_params=pose)


            trans_vertices = util.batch_orth_proj(vertices, cam);
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]


            losses = {}
            
#             if k < 250:
#                 losses['landmark_2d'] = util.l2_distance(landmarks2d[:, 17:, :2],
#                                                       landmarks_2d_gt[:, 17:, :2]) * 2.0 #config.w_lmks
#             else:
#                 losses['landmark_2d'] = util.l2_distance(landmarks2d[:, :, :2],
#                                                       landmarks_2d_gt[:, :, :2]) * 2.0
            losses['landmark_2d'] = util.l2_distance(landmarks2d[:, :, :2],
                                                      landmarks_2d_gt[:, :, :2]) * 2.0
    
            losses['landmark_3d'] = util.l2_distance(landmarks3d[:, :, :2],
                                                      landmarks_3d_gt[:, :, :2]) * 1.0
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(expression ** 2) / 2) * config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * config.w_pose_reg


            ## render
            albedos = flametex(tex) / 255.
            losses['texture_reg'] = loss_mse(albedos, texture_mean.repeat(args.batch_size, 1, 1, 1)) #* 1e-3 # Regularize learned texture.
            ops = render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['photometric_texture'] = (image_masks * (predicted_images - images).abs()).mean() \
                                            * config.w_pho


            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
#                 losses_to_plot[key].append(losses[key].item()) # Store for plotting later.

            losses['all_loss'] = all_loss
#             losses_to_plot['all_loss'].append(losses['all_loss'].item())


            optim.zero_grad()
            all_loss.backward()
            optim.step()
#             scheduler.step(all_loss)


            pbar.set_description(
                (
                    f"total: {losses['all_loss']:.4f}; lmk_2d: {losses['landmark_2d']:.4f}; "
                    f"lmk_3d: {losses['landmark_3d']:.4f}; "
                    f"shape: {losses['shape_reg']:.4f}; express: {losses['expression_reg']:.4f}; "
                    f"photo: {losses['photometric_texture']:.4f}; "
                )
            )


            # visualize
            if k % modulo_save_imgs == 0:
                bsize = range(0,10)
                shape_images = render.render_shape(vertices, trans_vertices, images)
                save_rendered_imgs(savefolder, k, images[bsize], predicted_images[bsize], shape_images[bsize],
                                   albedos[bsize], ops['albedo_images'].detach().cpu()[bsize],
                                   landmarks_2d_gt, landmarks2d, landmarks3d)
#                 try:
# #                     grids = {}
# #     #                 visind = range(bz)  # [0]
# #                     grids['images'] = torchvision.utils.make_grid(images).detach().cpu()
# #                     grids['landmarks_gt'] = torchvision.utils.make_grid(
# #                         util.tensor_vis_landmarks(images.clone().detach(), landmarks_gt))
# #                     grids['landmarks2d'] = torchvision.utils.make_grid(
# #                         util.tensor_vis_landmarks(images, landmarks2d))
# #                     grids['landmarks3d'] = torchvision.utils.make_grid(
# #                         util.tensor_vis_landmarks(images, landmarks3d))
# #                     grids['albedoimage'] = torchvision.utils.make_grid(
# #                         (ops['albedo_images']).detach().cpu())
# #                     grids['render'] = torchvision.utils.make_grid(predicted_images.detach().float().cpu())
# #                     shape_images = render.render_shape(vertices, trans_vertices, images)
# #                     grids['shape'] = torchvision.utils.make_grid(
# #                         F.interpolate(shape_images, [224, 224])).detach().float().cpu()


# #                     grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos, [224, 224])).detach().cpu()
# #                     grid = torch.cat(list(grids.values()), 1)
# #                     grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
# #                     grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

# #                     cv2.imwrite('{}/{}.jpg'.format(savefolder, str(k).zfill(6)), grid_image) 

#                     shape_images = render.render_shape(vertices, trans_vertices, images)
#                     save_rendered_imgs(savefolder, k, images, predicted_images, shape_images, albedos, ops,
#                                        landmarks_gt, landmarks2d, landmarks3d)
#                 except:
#                     print("Error saving images and renderings... continuing")
#                     continue
        
            
#             if k % modulo_save_model == 0:
#                 save_checkpoint(path=savefolder,
#                                 epoch=k+1,
#                                 losses=losses_to_plot,
#                                 model=dfr)
    
    
    # Save final epoch renderings and checkpoints.
    #
    bsize = range(0,10)
    shape_images = render.render_shape(vertices, trans_vertices, images)
    save_rendered_imgs(savefolder, k, images[bsize], predicted_images[bsize], shape_images[bsize],
                       albedos[bsize], ops['albedo_images'].detach().cpu()[bsize],
                       landmarks_2d_gt, landmarks2d, landmarks3d)
    
    
#     save_checkpoint(path=savefolder,
#                     epoch=k+1,
#                     losses=losses_to_plot,
#                     model=dfr)
    
    if args.local_rank == 0:
        print("cam: ", cam)
        print("landmarks3d.mean: ", landmarks3d.mean())
        print("landmarks3d.min: ", landmarks3d.min())
        print("landmarks3d.max: ", landmarks3d.max())
    
    
    
    
    

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=500)
    parser.add_argument("--iter_rigid", type=int,
                        help="epochs for training only landmaorks, pose",
                        default=100)
    parser.add_argument("--iter_save_img", type=int, 
                        help="modulo epoch value to save output images from model during training",
                        default=50)
    parser.add_argument("--iter_save_chkpt", type=int,
                        help="modulo epoch value to checkpoint model to disk during training",
                        default=100)
    parser.add_argument("--workers", type=int, help="number of dataloader workers", default=4)
#     parser.add_argument("--size", type=int, default=256)d
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--flame_data", type=str, default=None)
    parser.add_argument("--train_data", type=str, default='/home/jupyter/training_data_dfr')

    args = parser.parse_args()
    assert args.iter >= args.iter_rigid
    
    
    device = 'cuda'
    config = get_config()
    config['flame_model_path'] = f'{args.flame_data}/generic_model.pkl'
    config['flame_lmk_embedding_path'] = f'{args.flame_data}/landmark_embedding.npy'
    config['tex_space_path'] = f'{args.flame_data}/FLAME_texture.npz'
    config = dict2obj(config)
#     'flame_model_path': './data/generic_model.pkl',  # acquire it from FLAME project page
#         'flame_lmk_embedding_path': './data/landmark_embedding.npy',
#         'tex_space_path': './data/FLAME_texture.npz',  # acquire it from FLAME project page
    

    if args.distributed:
        print("Distributed training...")
#         from torch.multiprocessing import set_start_method
# #         try:
# #             set_start_method('spawn')
# #         except RuntimeError:
# #             pass
        
#         import multiprocessing as mp
#         mp.set_start_method('spawn')
#         q = mp.Queue()
#         p = mp.Process(train_distributed, args=(args, config))
#         p.start()
#         print(q.get())
#         p.join()

        train_distributed(args, config)
        dist.destroy_process_group()
    else:
        print("NON-distributed training...")
        config.batch_size = args.batch_size
        dfr = DFRParamRegressor(config)
        if args.ckpt is not None:
            dfr.load_state_dict(torch.load(args.ckpt)['dfr'], strict=False)
            dfr.train();
    
    
        flame = FLAME(config)
        flametex = FLAMETex(config)
        mesh_file = f'{args.flame_data}/head_template_mesh.obj'
        render = Renderer(config.image_size, obj_filename=mesh_file)

        
        dataset = DatasetDFR(args.train_data)
        sampler = data_sampler(dataset, shuffle=True, distributed=args.distributed)
        dataloader = data.DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     sampler=sampler,
                                     drop_last=True,
                                     num_workers=args.workers)
        
        
        texture_mean = flametex.get_texture_mean(args.batch_size) / 255.
        texture_mean = texture_mean.cuda()
            
            
        dfr.to(device)
        flame.to(device)
        render.to(device)
        flametex.to(device)

        
        train(args, config, dataloader, dfr, flame, flametex, render, texture_mean, device)
        
        
        
# Example commandline run:
#
# CUDA_VISIBLE_DEVICES=2,3 python train_dfr_regressor.py --iter=500 --iter_rigid=0 --iter_save_img=10 --iter_save_chkpt=50 --batch_size=50
#
# Distributed:
#
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_dfr_regressor.py --iter=10 --iter_rigid=10 --iter_save_img=1 --local_rank=0 --batch_size=4 --distributed --flame_data=/home/ec2-user/SageMaker/pretrained_models --train_data=/home/ec2-user/SageMaker/train_data_trunc0p2