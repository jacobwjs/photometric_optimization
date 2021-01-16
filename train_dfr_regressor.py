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
    
    return dict2obj(config)



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
    
    

def save_checkpoint(path, epoch, losses, model):
    epoch = str(epoch).zfill(6)
    torch.save(losses, f'{path}/losses_epoch{epoch}.pkl')
    
    model_data = {
        'dfr': model.state_dict(),
    }
    torch.save(model_data, f'{path}/dfr_ckpt_epoch{epoch}.pt')


    
def save_rendered_imgs(savefolder, epoch, images, predicted_images, shape_images, albedos, ops,
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
    grids['albedoimage'] = torchvision.utils.make_grid(
        (ops['albedo_images']).detach().cpu())
    grids['render'] = torchvision.utils.make_grid(predicted_images.detach().float().cpu())
#     shape_images = render.render_shape(vertices, trans_vertices, images)
    grids['shape'] = torchvision.utils.make_grid(
        F.interpolate(shape_images, [224, 224])).detach().float().cpu()


    grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos, [224, 224])).detach().cpu()
    grid = torch.cat(list(grids.values()), 1)
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

    cv2.imwrite('{}/{}.jpg'.format(savefolder, str(epoch).zfill(6)), grid_image)
    
    
    

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
#     optim = torch.optim.SGD(dfr.parameters(), lr=0.01, momentum=0.9) # Produces NaNs
#     optim = torch.optim.RMSprop(params, lr=0.01)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min') 
    
    
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
                        util.tensor_vis_landmarks(images, landmarks_gt))
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
# def train_parallel(rank, world_size, args, loader, dfr, flame, flametex, cam, optim):
def train_parallel(rank, world_size, args, loader, dfr, flame, flametex, optim):
# def train_parallel(rank, world_size, args, loader, dfr, optim):
    
    config = get_config()
    loader = sample_data(loader)
    pbar = range(args.iter)
    
    print("Rank: ", get_rank(), ", rank=", rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f"{rank + 1}/{world_size} process initialized.")
    
    kwargs_ddp = {'device_ids': [rank]}
    dfr = DDP(dfr, **kwargs_ddp)
    flame = DDP(flame, **kwargs_ddp)
    flametex = DDP(flametex, **kwargs_ddp)
#     flame = FLAME(config).cuda(rank)
#     flametex = FLAMETex(config).cuda(rank)
#     cam.cuda(rank)
    
    if rank == 0:
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
        
#     if args.distributed:
#         dfr_module = dfr.module
#         flame_module = flame.module
#         flametex_module = flametex.module
#     else:
#         dfr_module = dfr
#         flame_module = flame
#         flametex_module = flametex

    bz = args.batch_size
    tex = nn.Parameter(torch.zeros(bz, config.tex_params).float().cuda(rank))
    cam = torch.zeros(bz, config.camera_params); 
    cam[:, 0] = 5.0
    cam = nn.Parameter(cam.float().cuda(rank))
    lights = nn.Parameter(torch.zeros(bz, 9, 3).float().cuda(rank))
        
    
    for idx in pbar:
#         if i > args.iter:
#             print('Done training!')
#             break
#         print("Rank: ", get_rank(), " , epoch: ", idx)
        
#         example = next(loader)
        for example in loader:
            latents = example['latents'].cuda(rank)
            landmarks_gt = example['landmarks_gt'].cuda(rank)
            images = example['images'].cuda(rank)
            image_masks = example['image_masks'].cuda(rank)

            shape, expression, pose = dfr(latents.view(args.batch_size, -1))
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
            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2],
                                                  landmarks_gt[:, :, :2]) * 1#config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
    #                 losses_to_plot[key].append(losses[key].item()) # Store for plotting later.

            losses['all_loss'] = all_loss
    #             losses_to_plot['all_loss'].append(losses['all_loss'].item())


            optim.zero_grad()
            all_loss.backward()
            optim.step()
        
            if get_rank() == 0:
                pbar.set_description(
                    (
                        f"total: {losses['all_loss']:.4f}; landmark: {losses['landmark']:.4f};"
                    )
                )
    
    
    
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
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--distributed", type=int, default=0)
    parser.add_argument("--train_data", type=str, default='/home/jupyter/training_data_dfr')

    args = parser.parse_args()
    assert args.iter > args.iter_rigid
    
    
    device = 'cuda'
    
    
    if args.distributed:
        print('Distributed training...')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    config = get_config()
    config.batch_size = args.batch_size
#     print(config)
    
    
#     dfr = DFRParamRegressor(config).to(device)
#     flame = FLAME(config).to(device)
#     flametex = FLAMETex(config).to(device)
#     mesh_file = './data/head_template_mesh.obj'
#     render = Renderer(config.image_size, obj_filename=mesh_file).to(device)
    dfr = DFRParamRegressor(config)
    if args.ckpt is not None:
#         g_ema.load_state_dict(torch.load(checkpoint_sg2)['g_ema'], strict=False)
#         g_ema = g_ema.to(device)
#         g_ema.eval();
        dfr.load_state_dict(torch.load(args.ckpt)['dfr'], strict=False)
        dfr.eval();
    
    
    flame = FLAME(config)
    flametex = FLAMETex(config)
    mesh_file = './data/head_template_mesh.obj'
    render = Renderer(config.image_size, obj_filename=mesh_file)

#     # self._setup_renderer()
#     # mesh_file = './data/head_template_mesh.obj'
#     mesh_file = f'{path_photo_optim_repo}/data/head_template_mesh.obj'
#     render = Renderer(image_size, obj_filename=mesh_file).to(device)
    
#     if args.distributed:
#         dfr = nn.parallel.DistributedDataParallel(
#             dfr,
#             device_ids=[args.local_rank],
#             output_device=args.local_rank,
# #             broadcast_buffers=False
#         )
        
# #         flame = nn.parallel.DistributedDataParallel(
# #             flame,
# #             device_ids=[args.local_rank],
# #             output_device=args.local_rank,
# # #             broadcast_buffers=False
# #         )
        
# #         flametex = nn.parallel.DistributedDataParallel(
# #             flametex,
# #             device_ids=[args.local_rank],
# #             output_device=args.local_rank,
# # #             broadcast_buffers=False
# #         )

    
#     torch.multiprocessing.set_start_method("spawn")
    dataset = DatasetDFR(args.train_data)
    sampler = data_sampler(dataset, shuffle=True, distributed=args.distributed)
    dataloader = data.DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 sampler=sampler,
                                 drop_last=True,
                                 num_workers=args.workers)
    
    


    
    
    if args.distributed:
        #     lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(device))    
        e_optim = torch.optim.Adam(
                        list(dfr.parameters()), #+ [cam, tex, lights],
                        lr=config.e_lr,
                        weight_decay=config.e_wd
        )
        
        world_size = torch.cuda.device_count()
        print("GPUs found: ", world_size)
#         args = (world_size, args, dataloader, dfr, e_optim)
        args = (world_size, args, dataloader, dfr, flame, flametex, e_optim)
#         args = (rank, world_size, args, dataloader, dfr, flame, flametex, cam, e_optim)
        mp.spawn(train_parallel,
            args=args,
            nprocs=world_size,
            join=True)
#         dist.destroy_process_group()
    else:
        texture_mean = flametex.get_texture_mean(args.batch_size) / 255.
        texture_mean = texture_mean.cuda()
        
#         dfr = nn.DataParallel(dfr)
#         flame = nn.DataParallel(flame)
#         flametex = nn.DataParallel(flametex)
#         render = nn.DataParallel(render)
        
        dfr.to(device)
        flame.to(device)
        render.to(device)
        flametex.to(device)
        
        train(args, config, dataloader, dfr, flame, flametex, render, texture_mean, device)
        
        
        
# Example commandline run:
#
# CUDA_VISIBLE_DEVICES=2,3 python train_dfr_regressor.py --iter=500 --iter_rigid=0 --iter_save_img=10 --iter_save_chkpt=50 --batch_size=50