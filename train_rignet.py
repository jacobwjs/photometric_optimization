import argparse
import yaml
import os, sys
import numpy as np
import glob


from pathlib import Path


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
# from DFR_regressor import DFRParamRegressor
# from FLAME import FLAME, FLAMETex
# from renderer import Renderer
import util
import load


from tqdm import tqdm




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
  

def squeeze_dims(x, dims=3):
    '''Add in batch dim if doesn't exist.'''
    if len(x.shape) == dims:
        return x
    elif len(x.shape) > dims:
        return x.squeeze()
    else:
        return x.unsqueeze(0)
            
def collate_fn(batch):
    A = [x[0] for x in batch]
    B = [x[1] for x in batch]
#     return A, B

    w_A = torch.cat([squeeze_dims(x['latents'], dims=3) for x in A], dim=0)
    landmarks_2d_gt_A = torch.cat([squeeze_dims(x['landmarks_2d_gt'], dims=3) for x in A], dim=0)
    landmarks_3d_gt_A = torch.cat([squeeze_dims(x['landmarks_3d_gt'], dims=3) for x in A], dim=0)
    images_A = torch.cat([squeeze_dims(x['images'], dims=4) for x in A], dim=0)
    image_masks_A = torch.cat([squeeze_dims(x['image_masks'], dims=4) for x in A], dim=0)
    A_dict = {}
    A_dict['latents'] = w_A
    A_dict['landmarks_2d_gt'] = landmarks_2d_gt_A
    A_dict['landmarks_3d_gt'] = landmarks_3d_gt_A
    A_dict['images'] = images_A
    A_dict['image_masks'] = image_masks_A

    w_B = torch.cat([squeeze_dims(x['latents'], dims=3) for x in B], dim=0)
    landmarks_2d_gt_B = torch.cat([squeeze_dims(x['landmarks_2d_gt'], dims=3) for x in B], dim=0)
    landmarks_3d_gt_B = torch.cat([squeeze_dims(x['landmarks_3d_gt'], dims=3) for x in B], dim=0)
    images_B = torch.cat([squeeze_dims(x['images'], dims=4) for x in B], dim=0)
    image_masks_B = torch.cat([squeeze_dims(x['image_masks'], dims=4) for x in B], dim=0)
    B_dict = {}
    B_dict['latents'] = w_B
    B_dict['landmarks_2d_gt'] = landmarks_2d_gt_B
    B_dict['landmarks_3d_gt'] = landmarks_3d_gt_B
    B_dict['images'] = images_B
    B_dict['image_masks'] = image_masks_B

    return A_dict, B_dict

             
class DatasetSiamese(data.Dataset):
    def __init__(self, path_to_dir_A, path_to_dir_B):
        self.path_A = path_to_dir_A
        self.path_B = path_to_dir_B
        self.files_A = glob.glob1(self.path_A, '*.pkl')
        self.files_B = glob.glob1(self.path_B, '*.pkl')
        
    def __len__(self):
        len_A = len(glob.glob1(self.path_A, '*.pkl'))
        len_B = len(glob.glob1(self.path_A, '*.pkl'))
        assert len_A == len_B
        return len_A
    
    def __getitem__(self, index):
        filename = str(index).zfill(6) + '.pkl'
        example_A = torch.load(f'{self.path_A}/{filename}')
        example_B = torch.load(f'{self.path_B}/{filename}')
        return example_A, example_B
    
    

def save_checkpoint(path, epoch, model, optim, losses=None):
    epoch = str(epoch).zfill(6)
    
    if losses is not None:
        torch.save(losses, f'{path}/losses_epoch{epoch}.pkl')
    
    model_data = {
        'epoch': epoch,
        'rignet': model.state_dict(),
        'optim': optim.state_dict()
    }
    torch.save(model_data, f'{path}/dfr_ckpt_epoch{epoch}.pt')


    
def save_rendered_imgs(savefolder, 
                       epoch,
                       images,
                       landmarks_gt,
                       landmarks2d,
                       landmarks3d=None,
                       predicted_images=None,
                       shape_images=None,
                       albedos=None,
                       albedo_images=None,
                       tag=None):

    if not os.path.exists(savefolder):
        os.makedirs(savefolder, exist_ok=True)
    
    grids = {}
    grids['images'] = torchvision.utils.make_grid(images).detach().cpu()
    grids['landmarks_gt'] = torchvision.utils.make_grid(
        util.tensor_vis_landmarks(images.clone().detach(), landmarks_gt))
    grids['landmarks2d'] = torchvision.utils.make_grid(
        util.tensor_vis_landmarks(images, landmarks2d))
    
    if landmarks3d is not None:
        grids['landmarks3d'] = torchvision.utils.make_grid(
            util.tensor_vis_landmarks(images, landmarks3d))
    if albedo_images is not None:
        grids['albedoimage'] = torchvision.utils.make_grid(albedo_images)
    if predicted_images is not None:
        grids['render'] = torchvision.utils.make_grid(predicted_images)
    if shape_images is not None:
        grids['shape'] = torchvision.utils.make_grid(
            F.interpolate(shape_images, [224, 224])).detach().float().cpu()
    if albedos is not None:
        grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos, [224, 224])).detach().cpu()

    grid = torch.cat(list(grids.values()), 1)
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

    filename = str(epoch).zfill(6)
    if tag is not None:
        filename = filename + tag
        
    cv2.imwrite('{}/{}.jpg'.format(savefolder, filename), grid_image)
    


    
class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
    
    def forward(self, verts1, verts2):
        return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()
    
    
    
def latent_reconstruction_loss(rignet, w_A, params_A, w_B, params_B, labels_in, scale_recon=10.0):
    loss_recon = 0.0
    w_Ahat = rignet(w_A, params_A, labels_in) # Reconsruct self
    loss_recon = (w_Ahat - w_A).abs().square().mean() * scale_recon
    w_Bhat = rignet(w_B, params_B, labels_in) # Reconsruct self
    loss_recon += (w_Bhat - w_B).abs().square().mean() * scale_recon
    return loss_recon
   
    
def cycle_editing_loss(epoch, texture_mean, example, param,
                       _flame, _flametex, _render):
#     import util
    
    loss_mse = nn.MSELoss().cuda()


    latents = example['latents'].cuda()
    landmarks_2d_gt = example['landmarks_2d_gt'].cuda()
    landmarks_3d_gt = example['landmarks_3d_gt'].cuda()
    images = example['images'].cuda()
    image_masks = example['image_masks'].cuda()
    
    batch_size = latents.shape[0]
    
    # shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
    shape, expression, pose, tex, cam, lights = param
#     vertices, landmarks2d, landmarks3d = _flame(shape_params=shape,
#                                                expression_params=expression,
#                                                pose_params=pose)

    
    
#     trans_vertices = util.batch_orth_proj(vertices, cam);
#     trans_vertices[..., 1:] = - trans_vertices[..., 1:]
#     landmarks2d = util.batch_orth_proj(landmarks2d, cam);
#     landmarks2d[..., 1:] = - landmarks2d[..., 1:]
#     landmarks3d = util.batch_orth_proj(landmarks3d, cam);
#     landmarks3d[..., 1:] = - landmarks3d[..., 1:]
    
#     # render
#     #
#     albedos = _flametex(tex) / 255.
#     ops = _render(vertices, trans_vertices, albedos, lights)

    render_outputs = render_all(param, _flame, _flametex, _render)
    vertices = render_outputs['vertices']
    landmarks2d = render_outputs['landmarks2d']
    landmarks3d = render_outputs['landmarks3d']
    trans_vertices = render_outputs['trans_vertices']
    albedos = render_outputs['albedos']
    ops = render_outputs['ops']
    predicted_images = ops['images']
    
    
    losses = {}
#     print(landmarks2d.shape)
#     print(landmarks_2d_gt.shape)
    losses['landmark_2d'] = util.l2_distance(landmarks2d[:, :, :2],
                                              landmarks_2d_gt[:, :, :2]) * 10.0
    losses['landmark_3d'] = util.l2_distance(landmarks3d[:, :, :2],
                                              landmarks_3d_gt[:, :, :2]) * 10.0
    losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * 1e-4
    losses['expression_reg'] = (torch.sum(expression ** 2) / 2) * 1e-4
    losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * 1e-4 #config.w_pose_reg 
    losses['texture_reg'] = loss_mse(albedos, texture_mean.repeat(batch_size, 1, 1, 1)) # Regularize learned texture.
    losses['photometric_texture'] = (image_masks * (predicted_images - images).abs()).mean() \
                                    * 1.0 #config.w_pho


    all_loss = 0.
    for key in losses.keys():
        all_loss = all_loss + losses[key]
#         losses_to_plot[key].append(losses[key].item()) # Store for plotting later.


    losses['all_loss'] = all_loss
#     losses_to_plot['all_loss'].append(losses['all_loss'].item())


#     if savefolder is not None:
#         bsize = range(0, batch_size_save)
#         shape_images = render.render_shape(vertices, trans_vertices, images)
#         save_rendered_imgs(
#             savefolder,
#             epoch,
#             images[bsize].clone(),
#             landmarks_2d_gt.clone(),
#             landmarks2d.clone(),
#             landmarks3d.clone(),
#             predicted_images[bsize].detach().cpu().float().clone(),
#             shape_images[bsize].clone(),
#             albedos[bsize].clone(),
#             ops['albedo_images'].detach().cpu().clone()[bsize],                        
#         )

    return losses
    
    
def render_all(param, _flame, _flametex, _render):
    shape, expression, pose, tex, cam, lights = param
    vertices, landmarks2d, landmarks3d = _flame(shape_params=shape,
                                               expression_params=expression,
                                               pose_params=pose)
    

    trans_vertices = util.batch_orth_proj(vertices, cam);
    trans_vertices[..., 1:] = - trans_vertices[..., 1:]
    landmarks2d = util.batch_orth_proj(landmarks2d, cam);
    landmarks2d[..., 1:] = - landmarks2d[..., 1:]
    landmarks3d = util.batch_orth_proj(landmarks3d, cam);
    landmarks3d[..., 1:] = - landmarks3d[..., 1:] 
    
    # render
    #
    albedos = _flametex(tex) / 255.
    ops = _render(vertices, trans_vertices, albedos, lights)

    
    out = {}
    out['vertices'] = vertices
    out['landmarks2d'] = landmarks2d
    out['landmarks3d'] = landmarks3d
    out['trans_vertices'] = trans_vertices
    out['albedos'] = albedos
    out['ops'] = ops
    
    return out


    
def train(args, config, loader, rignet, dfr, flame, flametex, render, device):
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H.%M.%S")     # dd/mm/YY H:M:S
    savefolder = os.path.sep.join(['./test_results', f'{dt_string}'])
    if not os.path.exists(savefolder):
        os.makedirs(savefolder, exist_ok=True)

        
    optim = torch.optim.Adam(
                rignet.parameters(),
                lr=1e-3,
#                 weight_decay=0.00001 # config.e_wd
    )
    
    
#     loader = sample_data(loader)

    
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
    
    
    loss_mse = nn.MSELoss().cuda()
    
    
    modulo_save_imgs = args.iter_save_img
    modulo_save_model = args.iter_save_ckpt
    
    # shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
    # onehot_cam = F.one_hot(torch.tensor([4]), 6).unsqueeze(0)
    onehot_pose = F.one_hot(torch.tensor([2]), 6).unsqueeze(0)
    # labels_in = torch.cat((onehot_pose, onehot_expr), dim=0).cuda()
    labels_in = onehot_pose
    labels_in = labels_in.expand(args.batch_size, -1, -1)
    labels_in = labels_in.cuda()
    
    
    texture_mean = flametex.get_texture_mean(args.batch_size) / 255.
    texture_mean = texture_mean.cuda()
    
    save_batch_size = args.save_batch_size
    
    
    pbar = tqdm(range(0, args.iter), dynamic_ncols=True, smoothing=0.01)
    k = 0
    for k in pbar:
        for A, B in loader:
            try:
                w_A = A['latents'].cuda()
                landmarks_2d_gt_A = A['landmarks_2d_gt'].cuda()
                images_A = A['images'].cuda()
                image_masks_A = A['image_masks'].cuda()

                w_B = B['latents'].cuda()
                landmarks_2d_gt_B = B['landmarks_2d_gt'].cuda()
                images_B = B['images'].cuda()
                image_masks_B = B['image_masks'].cuda()
                
            except:
                print('Skipping A, B = loader')
                continue

                
#             print("w_A: ", w_A.shape)
#             print("w_B: ", w_B.shape)
#             print("landmarks_2d_gt_A: ", landmarks_2d_gt_A.shape)
#             print("landmarks_2d_gt_B: ", landmarks_2d_gt_B.shape)
#             print("images_A: ", images_A.shape)
#             print("images_B: ", images_B.shape)


            # 1) Calculate parameters from latents
            #
            # params_A = dfr(w_A.view(w_A.shape[0], -1))
            # params_B = dfr(w_B.view(w_B.shape[0], -1))
            params_A = dfr(nn.Flatten()(w_A))
            params_B = dfr(nn.Flatten()(w_B))
            
            
            # Enforce latents produced from RigNet are bound to the
            # original distribution.
            # That is, latents that create parameters, should be able
            # to take those parameters and re-generate the original latents.
            #
            # e.g. param = dfr(w); w_hat = rignet(w, param); MSE(w - w_hat)
            #
            loss_recon = latent_reconstruction_loss(rignet,
                                                    w_A, params_A,
                                                    w_B, params_B,
                                                    labels_in, scale_recon=10.0)


            # 2) Transfer semantic "style" (e.g. pose) from one latent to another.
            #
            w_Ahat = rignet(w_B, params_A, labels_in) # Transfer semantic params from A to latent B
            w_Bhat = rignet(w_A, params_B, labels_in) # Transfer semantic params from B to latent A
    #         w_Bhat = rignet(w_Ahat, params_B, labels_in) # Transfer semantic params from A to latent B
    #         w_Ahat = rignet(w_Bhat, params_A, labels_in) # Transfer semantic params from B to latent A


            # 3) Create the cycle, where transferred paramter to latent, and latent to parameter
            #    should match.
            #
            params_AB = dfr(nn.Flatten()(w_Ahat))
            params_BA = dfr(nn.Flatten()(w_Bhat))


            # shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            params_edit_A = [x.clone() for x in params_A]
            params_edit_B = [x.clone() for x in params_B]
#             params_edit_A = [x for x in params_A]
#             params_edit_B = [x for x in params_B]


            ### TODO:
            ### - Update below to use labels to dictate what is swapped in
            ###   for the editing loss.
            ###
            params_edit_A[2] = params_AB[2] # Ensure pose is maintained
            params_edit_B[2] = params_BA[2] # Ensure pose is maintained


            # Losses
            # ------------------------------------------------
            total_loss = 0.0


            loss_A = cycle_editing_loss(
                k, texture_mean, A, params_edit_A, flame, flametex, render,
            )
            loss_B = cycle_editing_loss(
                k, texture_mean, B, params_edit_B, flame, flametex, render,
            )


            loss_edit = loss_A['all_loss'] + loss_B['all_loss']


            total_loss += loss_recon
            total_loss += loss_edit 


            optim.zero_grad()
            total_loss.backward()
            optim.step()


            pbar.set_description(
                    (
                        f"total: {total_loss:.4f}; edit: {loss_edit:.4f}; recon: {loss_recon:.4f}; "
                    )
                )

#         if k % 100 == 0:
#             print("epoch: ", k, ", loss_edit: ", total_loss.item(), ", loss_recon: ", loss_recon.item())
    

        if k % modulo_save_imgs == 0:
#             savefolder = path_save
#             render_outputs = render_all(params_A, flame, flametex, render)
            render_outputs = render_all(params_edit_A, flame, flametex, render)
            vertices = render_outputs['vertices']
            landmarks2d = render_outputs['landmarks2d']
            landmarks3d = render_outputs['landmarks3d']
            trans_vertices = render_outputs['trans_vertices']
            albedos = render_outputs['albedos']
            ops = render_outputs['ops']
            predicted_images = ops['images']

            # Assign which set of data to write out
            #
            images = images_A
            landmarks_2d_gt = landmarks_2d_gt_A
        
            if savefolder is not None:
                bsize = range(0, save_batch_size)
                shape_images = render.render_shape(vertices, trans_vertices, images)
                save_rendered_imgs(
                    savefolder,
                    k,
                    images[bsize].clone(),
                    landmarks_2d_gt.clone(),
                    landmarks2d.clone(),
                    landmarks3d.clone(),
                    predicted_images[bsize].detach().cpu().float().clone(),
                    shape_images[bsize].clone(),
                    albedos[bsize].clone(),
                    ops['albedo_images'].detach().cpu().clone()[bsize],
                    tag='_verA'
                )

#             render_outputs = render_all(params_B, flame, flametex, render)
            render_outputs = render_all(params_edit_B, flame, flametex, render)
            vertices = render_outputs['vertices']
            landmarks2d = render_outputs['landmarks2d']
            landmarks3d = render_outputs['landmarks3d']
            trans_vertices = render_outputs['trans_vertices']
            albedos = render_outputs['albedos']
            ops = render_outputs['ops']
            predicted_images = ops['images']
            
            
            # Assign which set of data to write out
            #
            images = images_B
            landmarks_2d_gt = landmarks_2d_gt_B
            
            if savefolder is not None:
                bsize = range(0, save_batch_size)
                shape_images = render.render_shape(vertices, trans_vertices, images)
                save_rendered_imgs(
                    savefolder,
                    k,
                    images[bsize].clone(),
                    landmarks_2d_gt.clone(),
                    landmarks2d.clone(),
                    landmarks3d.clone(),
                    predicted_images[bsize].detach().cpu().float().clone(),
                    shape_images[bsize].clone(),
                    albedos[bsize].clone(),
                    ops['albedo_images'].detach().cpu().clone()[bsize],
                    tag='_verB'
                )


    

def train_distributed(args, config, savefolder, train_dir, dfr=None):
    
    if dfr is None:
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
    if args.local_rank == 0:
        print("training on ", train_dir)
    
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    #
    torch.cuda.set_device(args.local_rank)

    
    
    if dfr is None:
        # Have not trained on any data (i.e. first pass in).
        # Need to create.
        #
        print("creating DFR model...")
        dfr = DFRParamRegressor(config)
        print("done")
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
    
    
    # To help convergence we train on multiple stages of data generated 
    # with different truncation values. The idea being it is easier to
    # learn on many examples that are very similar (low truncation value e.g. 0.2)
    # and slowly incorporate more challenging and diverse examples 
    # (high truncation, e.g. 0.7).
    # 
    
    if args.local_rank == 0:
        print("Training on ", train_dir, "...")

    dataset = None
#     path= '/home/ec2-user/SageMaker/train_data/trunc02'
#     dataset = DatasetDFR(path)
#     print("train_dir: ", train_dir)
    dataset = DatasetDFR(train_dir)
    # makes sure that each process gets a different slice of the training data
    # during distributed training.
    #
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.cuda.device_count(),
# #             num_replicas=4,
#             rank = args.local_rank,
        )

#     loader = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
#         collate_fn=collate_fn,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )

    if args.local_rank == 0:
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


    idx_rigid_stop = args.iter_rigid
    modulo_save_imgs = args.iter_save_img
    modulo_save_model = args.iter_save_ckpt
    if args.save_batch_size < args.batch_size:
        save_batch_size = args.save_batch_size 
    else:
        save_batch_size = args.batch_size
    
    
    path_save = f'{savefolder}/{os.path.basename(train_dir)}'

    # Start optimization
    # -----------------------------
    #
    
    # Check if we start with "rigid training" (just facial landmarks).
    #
    if idx_rigid_stop > 0:
        k = 0
        vertices = None
        images = None
        trans_vertices = None
        landmarks2d = None
        landmarks3d = None
        pbar = tqdm(range(0, idx_rigid_stop), dynamic_ncols=True, smoothing=0.01)
        for k in pbar:
            if args.distributed:
                sampler.set_epoch(k)

    #         print("args.local_rank: ", args.local_rank)
            for example in loader:
                latents = example['latents'].cuda(args.local_rank)
                landmarks_2d_gt = example['landmarks_2d_gt'].cuda(args.local_rank)
                images = example['images'].cuda(args.local_rank)
                image_masks = example['image_masks'].cuda(args.local_rank)

    #             if args.local_rank == 0:
    #                 print("2 step in rank: ", args.local_rank, flush=True)
    #                 print(latents.shape)
    #                 print(landmarks_2d_gt.shape)
    #                 print(images.shape)
    #                 print(image_masks.shape)


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
                losses['all_loss'] = all_loss


                optim.zero_grad()
                all_loss.backward()
                optim.step()


                if args.local_rank == 0:
                    for key in losses.keys():
                        losses_to_plot[key].append(losses[key].item()) # Store for plotting later.
                    losses_to_plot['all_loss'].append(losses['all_loss'].item())  


    #             if args.local_rank == 0:
                pbar.set_description(
                    (
                        f"total: {losses['all_loss']:.4f}; landmark_2d: {losses['landmark_2d']:.4f}; "
                    )
                )

        
                # Visualize results.
                #
                if args.local_rank == 0:
                    if (k % modulo_save_imgs == 0):
    #                     print("saving images...", path_save)
                        bsize = range(0, save_batch_size)
                        save_rendered_imgs(
                            path_save,
                            k,
                            images[bsize].clone(),
                            landmarks_2d_gt.clone(),
                            landmarks2d.clone(),
                            landmarks3d.clone(),
                        )


                # Save current model.
                #
                if args.local_rank == 0:
                    if k % modulo_save_model == 0:
                        save_checkpoint(
                            path=path_save,
                            epoch=k,
                            losses=losses_to_plot,
                            model=dfr,
                            optim=optim)   


        dist.barrier()
                        
        # Save final epoch for rigid fitting.
        #
        if args.local_rank == 0:
            save_checkpoint(path=savefolder,
                        epoch=k,
                        losses=losses_to_plot,
                        model=dfr,
                        optim=optim)


        if args.local_rank == 0:
            print("DONE: rigid training...") 

            
        

        
        
    # Add in the rendering
    #
    
    train_render(args, dfr, render, flame, flametex,
                 texture_mean=texture_mean,
                 optim=optim,
                 savefolder=path_save,
                 sampler=sampler,
                 dataloader=loader,
                 losses_to_plot=losses_to_plot if args.local_rank == 0 else None)
    
    
    return dfr


        
def train_render(args, dfr, render, flame, flametex,
                 texture_mean, optim, savefolder, 
                 sampler, dataloader, losses_to_plot):
    
    
    loss_mse = nn.MSELoss()
    
    ### TODO:
    ### - make below a return value(s) from a function call
    ###   so it does not need to be udpated in multiple functions
    ###   by hand.
    ###
    idx_rigid_stop = args.iter_rigid
    modulo_save_imgs = args.iter_save_img
    modulo_save_model = args.iter_save_ckpt
    if args.save_batch_size < args.batch_size:
        save_batch_size = args.save_batch_size 
    else:
        save_batch_size = args.batch_size
    
    # Second stage training. Adding in photometric loss.
    #
    k = 0
    vertices = None
#     trans_vertices = None
#     images = None
#     landmarks2d = None
#     landmarks3d = None
#     predicted_images = None,
#     shape_images = None
#     albedos = None
#     ops = None
    pbar = tqdm(range(idx_rigid_stop, args.iter), dynamic_ncols=True, smoothing=0.01)
    for k in pbar:
        if args.distributed:
            sampler.set_epoch(k)
            
        for example in dataloader:
            latents = example['latents'].cuda(args.local_rank)
            landmarks_2d_gt = example['landmarks_2d_gt'].cuda(args.local_rank)
            landmarks_3d_gt = example['landmarks_3d_gt'].cuda(args.local_rank)
            images = example['images'].cuda(args.local_rank)
            image_masks = example['image_masks'].cuda(args.local_rank)


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
            losses['landmark_2d'] = util.l2_distance(landmarks2d[:, :, :2],
                                                      landmarks_2d_gt[:, :, :2]) * 1.0
            losses['landmark_3d'] = util.l2_distance(landmarks3d[:, :, :2],
                                                      landmarks_3d_gt[:, :, :2]) * 1.0
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(expression ** 2) / 2) * config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * config.w_pose_reg


            ## render
            albedos = flametex(tex) / 255.
            
            # Regularize learned texture.
            #
            losses['texture_reg'] = loss_mse(albedos, texture_mean.repeat(args.batch_size, 1, 1, 1)) 
            ops = render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['photometric_texture'] = (image_masks * (predicted_images - images).abs()).mean() \
                                            * config.w_pho


            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss


            optim.zero_grad()
            all_loss.backward()
            optim.step()
#             scheduler.step(all_loss)


            # Store for plotting later.
            #
            if (args.local_rank == 0) and (losses_to_plot is not None):
                for key in losses.keys():
                    losses_to_plot[key].append(losses[key].item()) 
                losses_to_plot['all_loss'].append(losses['all_loss'].item()) 
                

            pbar.set_description(
                (
                    f"all: {losses['all_loss']:.4f}; 2d: {losses['landmark_2d']:.4f}; "
                    f"3d: {losses['landmark_3d']:.4f}; "
                    f"shape: {losses['shape_reg']:.4f}; expr: {losses['expression_reg']:.4f}; "
                    f"photo: {losses['photometric_texture']:.4f}; "
                )
            )


            # visualize results
            #
            if args.local_rank == 0:
                if k % modulo_save_imgs == 0:
                    bsize = range(0, save_batch_size)
                    shape_images = render.render_shape(vertices, trans_vertices, images)
                    save_rendered_imgs(
                        savefolder,
                        k,
                        images[bsize].clone(),
                        landmarks_2d_gt.clone(),
                        landmarks2d.clone(),
                        landmarks3d.clone(),
                        predicted_images[bsize].detach().cpu().float().clone(),
                        shape_images[bsize].clone(),
                        albedos[bsize].clone(),
                        ops['albedo_images'].detach().cpu().clone()[bsize],                        
                    )
                    
                if k % modulo_save_model == 0:
                    save_checkpoint(path=savefolder,
                                    epoch=k,
                                    losses=losses_to_plot,
                                    model=dfr,
                                    optim=optim)
    
    
#     if args.local_rank == 0:
#         if (k+1) % args.iter == 0:
#             bsize = range(0, save_batch_size)
#             shape_images = render.render_shape(vertices, trans_vertices, images)
#             save_rendered_imgs(
#                 savefolder,
#                 k,
#                 images[bsize].clone(),
#                 landmarks_2d_gt.clone(),
#                 landmarks2d.clone(),
#                 landmarks3d.clone(),
#                 predicted_images[bsize].detach().cpu().float().clone(),
#                 shape_images[bsize].clone(),
#                 albedos[bsize].clone(),
#                 ops['albedo_images'].detach().cpu().clone()[bsize],                        
#             )

    dist.barrier()
    
    
    if args.local_rank == 0:
        save_checkpoint(path=savefolder,
                        epoch=k,
                        losses=losses_to_plot,
                        model=dfr,
                        optim=optim)
    
    if args.local_rank == 0:
        print("DONE: render training...")

    
#     # Save final epoch renderings and checkpoints.
#     #
#     if args.local_rank == 0:
#         bsize = range(0, save_batch_size)
#         shape_images = render.render_shape(vertices, trans_vertices, images)
#         save_rendered_imgs(
#             savefolder,
#             k,
#             images[bsize].clone(),
#             landmarks_2d_gt.clone(),
#             landmarks2d.clone(),
#             landmarks3d.clone(),
#             predicted_images[bsize].detach().cpu().float().clone(),
#             shape_images[bsize].clone(),
#             albedos[bsize].clone(),
#             ops['albedo_images'].detach().cpu().clone()[bsize],                        
#         )

        
#     if args.local_rank == 0:
#         save_checkpoint(path=savefolder,
#                         epoch=k,
#                         losses=losses_to_plot,
#                         model=dfr,
#                         optim=optim)
    
    
#     if args.local_rank == 0:
#         print("cam: ", cam)
#         print("landmarks3d.mean: ", landmarks3d.mean())
#         print("landmarks3d.min: ", landmarks3d.min())
#         print("landmarks3d.max: ", landmarks3d.max())
    
    
    

    
    
    



    
    

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=50)
    parser.add_argument("--iter_save_img", type=int, 
                        help="modulo epoch value to save output images from model during training",
                        default=50)
    parser.add_argument("--iter_save_ckpt", type=int,
                        help="modulo epoch value to checkpoint model to disk during training",
                        default=100)
    parser.add_argument("--workers", type=int, help="number of dataloader workers", default=4)
    parser.add_argument("--save_batch_size", type=int,
                        help="number of images to save to disk when triggered", default=8)
    parser.add_argument("--base_path", type=str,
                        help="top level directory where all repos, etc. live",
                        default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--distributed", action='store_true')
#     parser.add_argument("--flame_data", type=str, default=None)
#     parser.add_argument("--train_data", type=str, default='/home/jupyter/train_data')

    
    args = parser.parse_args()
    assert args.batch_size >= args.save_batch_size
    
    device = 'cuda'
    
#     train_dir_A = f'{args.base_path}/train_data/rignet_A'
#     train_dir_B = f'{args.base_path}/train_data/rignet_B'
    train_dir_A = f'{args.base_path}/train_data/testing_A'
    train_dir_B = f'{args.base_path}/train_data/testing_B'
    print("train_dir: ", train_dir_A)
    print("train_dir: ", train_dir_B)
    dataset = DatasetSiamese(train_dir_A, train_dir_B)

    batch_size = args.batch_size
    sampler = None
    # # if args.distributed:
    # #     sampler = torch.utils.data.distributed.DistributedSampler(
    # #         dataset,
    # #         num_replicas=torch.cuda.device_count(),
    # # # #             num_replicas=4,
    # # #             rank = args.local_rank,
    # #     )

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    

    
      
    with open(f'{args.base_path}/photometric_optimization/configs/config.yaml') as f:
        config = yaml.safe_load(f)


    model_params = config['model_params']
    path_pretrained = f'{args.base_path}/pretrained_models'
    model_params['flame_model_path'] = f'{path_pretrained}/generic_model.pkl'
    model_params['flame_lmk_embedding_path'] = f'{path_pretrained}/landmark_embedding.npy'
    model_params['tex_space_path'] = f'{path_pretrained}/FLAME_texture.npz'

    # Load flame and render models.
    #
    render = load.renderer(args.base_path, model_params).cuda()
    flame = load.flame(args.base_path, model_params).cuda()
    flametex = load.flametex(args.base_path, model_params).cuda()
    
    # Load DFR, and RigNet models.
    #
    one_hot = True
    rignet = load.rignet(args.base_path, one_hot=one_hot).cuda()
    dfr = load.dfr(args.base_path, load_weights=True).cuda()

    
    # Kick off the training.
    #
    device = 'cuda'
    train(args, model_params, loader, rignet, dfr, flame, flametex, render, device)
    
    

#     if args.distributed:
#         print("Distributed training...")
# #         from torch.multiprocessing import set_start_method
# # #         try:
# # #             set_start_method('spawn')
# # #         except RuntimeError:
# # #             pass
        
#         # Create unique directory to save results.
#         #
#         from datetime import datetime
#         now = datetime.now()
#         dt_string = now.strftime("%d-%m-%Y_%H.%M.%S")     # dd/mm/YY H:M:S
#         savefolder = os.path.sep.join(['./test_results', f'{dt_string}'])
#         if not os.path.exists(savefolder):
#             os.makedirs(savefolder, exist_ok=True)

#         # Sort the directories based on truncation value.
#         # 
#         truncation_dirs = glob.glob(args.train_data + "/*")
#         truncation_dirs.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
#         print("top: ", args.train_data)
#         print("training dirs: ", truncation_dirs)
        
#         dfr = None
#         for trunc_dir in truncation_dirs:
#             dfr = train_distributed(args, config, savefolder, train_dir=trunc_dir, dfr=dfr)

#         dist.destroy_process_group()
#     else:
#         print("NON-distributed training...")
#         config.batch_size = args.batch_size
#         dfr = DFRParamRegressor(config)
#         if args.ckpt is not None:
#             dfr.load_state_dict(torch.load(args.ckpt)['dfr'], strict=False)
#             dfr.train();
    
    
#         flame = FLAME(config)
#         flametex = FLAMETex(config)
#         mesh_file = f'{args.flame_data}/head_template_mesh.obj'
#         render = Renderer(config.image_size, obj_filename=mesh_file)

        
#         dataset = DatasetDFR(args.train_data)
#         sampler = data_sampler(dataset, shuffle=True, distributed=args.distributed)
#         dataloader = data.DataLoader(dataset,
#                                      batch_size=args.batch_size,
#                                      sampler=sampler,
#                                      drop_last=True,
#                                      num_workers=args.workers)
        
        
#         texture_mean = flametex.get_texture_mean(args.batch_size) / 255.
#         texture_mean = texture_mean.cuda()
            
            
#         dfr.to(device)
#         flame.to(device)
#         render.to(device)
#         flametex.to(device)

        
#         train(args, config, dataloader, dfr, flame, flametex, render, texture_mean, device)
        
        
        
# Example commandline run:
#
# CUDA_VISIBLE_DEVICES=0 python train_rignet.py --save_batch_size=5 --batch_size=5 --iter=3 --iter_save_img=1 --workers=6 --base_path=/home/ec2-user/SageMaker
#
# Distributed:
# - example with 2 GPUs on 1 node.
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_dfr_regressor.py --iter=100 --iter_rigid=0 --iter_save_img=10 --iter_save_ckpt=10 --local_rank=0 --batch_size=32 --flame_data=/home/ec2-user/SageMaker/pretrained_models --train_data=/home/ec2-user/SageMaker/train_data --distributed --workers=4

# Debug:
# You can see what all the threads are doing “sudo gdb -p … thread apply all bt"