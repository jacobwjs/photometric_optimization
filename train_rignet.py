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

    dist.barrier()
    
    
    
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
    
    

def save_checkpoint(savefolder, epoch, model, optim, losses=None, distributed=False):
        
    epoch = str(epoch).zfill(6)
    
    
    if losses is not None:
        torch.save(losses, f'{savefolder}/losses_epoch{epoch}.pt')
    
    
    model_data = {}
    model_data['epoch'] = epoch
    model_data['optim'] = optim.state_dict()
    model_data['rignet'] =  model.state_dict()
    
    ### Comment:
    ### - Only needed if using nn.DataParallel
    ###
#     if distributed:
#         model_data['rignet'] =  model.module.state_dict()
#     else:
#         model_data['rignet'] =  model.state_dict()
        
        
    torch.save(model_data, f'{savefolder}/ringnet_ckpt_epoch{epoch}.pt')


    
    
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
    
    
    
def latent_reconstruction_loss(rignet,
                               w_A, p_A,
                               w_B, p_B, 
                               labels_in, scale_recon=10.0):
    loss_recon = 0.0
    w_reconA = rignet(w_A, p_A, labels_in) # Reconsruct self
    loss_recon = (w_reconA - w_A).abs().square().mean() * scale_recon
    w_reconB = rignet(w_B, p_B, labels_in) # Reconsruct self
    loss_recon += (w_reconB - w_B).abs().square().mean() * scale_recon
    
    return loss_recon

   
    
def rendering_loss(args, epoch, texture_mean, example, param,
                       _flame, _flametex, _render):
    
    loss_mse = nn.MSELoss().cuda(args.local_rank)


    latents = example['latents'].cuda(args.local_rank)
    landmarks_2d_gt = example['landmarks_2d_gt'].cuda(args.local_rank)
    landmarks_3d_gt = example['landmarks_3d_gt'].cuda(args.local_rank)
    images = example['images'].cuda(args.local_rank)
    image_masks = example['image_masks'].cuda(args.local_rank)
    
    batch_size = latents.shape[0]
    
    # shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
    shape, expression, pose, tex, cam, lights = param


    render_outputs = render_all(param, _flame, _flametex, _render)
    vertices = render_outputs['vertices']
    landmarks2d = render_outputs['landmarks2d']
    landmarks3d = render_outputs['landmarks3d']
    trans_vertices = render_outputs['trans_vertices']
    albedos = render_outputs['albedos']
    ops = render_outputs['ops']
    predicted_images = ops['images']
    
    
    losses = {}
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

    losses['all_loss'] = all_loss


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

        
#     optim = torch.optim.Adam(rignet.parameters(), lr=1e-4)
#     optim = torch.optim.Adadelta(rignet.parameters(), lr=0.001) # default Adadelta
#     optim = torch.optim.SGD(rignet.parameters(), lr=0.01, momentum=.9) # default SGD # Trains
#     optim = torch.optim.RMSprop(rignet.parameters()) # default RMSprop
    optim = torch.optim.Adamax(rignet.parameters()) # default Adamax
    
    
#     loader = sample_data(loader)
    args.local_rank = 0

    
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
#     onehot_pose = F.one_hot(torch.tensor([2]), 6)#.unsqueeze(0)
#     # labels_in = torch.cat((onehot_pose, onehot_expr), dim=0).cuda()
#     labels_in = onehot_pose
#     labels_in = labels_in.expand(args.batch_size, -1, -1)
#     labels_in = labels_in.cuda()
    total_params = 6
    semantic_labels = np.array([0, 1, 2, 3, 4, 5]) # only using subset of [shape, expression, pose, tex, cam, lights]


    texture_mean = flametex.get_texture_mean(args.batch_size) / 255.
    texture_mean = texture_mean.cuda()
    
    
    save_batch_size = args.save_batch_size
    batch_size = args.batch_size
    
    
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


            # 1) Calculate parameters from latents
            #
            # params_A = dfr(w_A.view(w_A.shape[0], -1))
            # params_B = dfr(w_B.view(w_B.shape[0], -1))
            params_A = dfr(nn.Flatten()(w_A))
            params_B = dfr(nn.Flatten()(w_B))
            
            
            # Randomly choose one of the semantic parameters to use as a label,
            # and later to be swapped out for rendering in the edit loss.
            #
            semantic_idx = np.random.randint(low=0, high=len(semantic_labels), size=(1,)) 
            semantic_param = semantic_labels[semantic_idx] # This will be used to swap edits to match the labels.
            labels_in = F.one_hot(torch.tensor([semantic_param]), total_params)
            labels_in = labels_in.expand(batch_size, -1, -1).cuda()
            
            
            # Enforce latents produced from RigNet are bound to the
            # original distribution.
            # That is, latents that create parameters, should be able
            # to take those parameters and re-generate the original latents.
            #
            # e.g. param = dfr(w); w_hat = rignet(w, param); MSE(w - w_hat)
            #
            labels_null = torch.zeros((batch_size, 1, total_params)).cuda()
            loss_recon = latent_reconstruction_loss(rignet,
                                                    w_A, params_A,
                                                    w_B, params_B,
                                                    labels_in=labels_null,
                                                    scale_recon=5.0)


            # 2) Transfer semantic "style" (e.g. pose) from one latent to another.
            #
            w_AB = rignet(w_A, params_B, labels_in) # Transfer semantic params from B to latent A
#             w_BA = rignet(w_B, params_A, labels_in) # Transfer semantic params from A to latent B


            # 3) Create the cycle, where transferred parameter to latent, and latent to parameter
            #    should match.
            #
            params_AB = dfr(nn.Flatten()(w_AB))
            params_AB = np.array(params_AB)
#             params_BA = dfr(nn.Flatten()(w_BA))
#             params_BA = np.array(params_BA)
            
            
            # shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            params_edit_A = None
            params_edit_B = None
            params_edit_A = np.array([x.detach().clone() for x in params_A])
#             params_edit_B = np.array([x.detach().clone() for x in params_B])


            params_edit_A[semantic_param] = params_AB[semantic_param] # Ensure pose is maintained
#             params_edit_B[semantic_param] = params_AB[semantic_param] # Ensure pose is maintained
        
#             params_A[semantic_param] = params_BA[semantic_param] # Ensure pose is maintained
#             params_B[semantic_param] = params_AB[semantic_param] # Ensure pose is maintained


            loss_edit_A = rendering_loss(
                args, k, texture_mean, A, params_edit_A, flame, flametex, render,
            )
#             loss_edit_B = rendering_loss(
#                 args, k, texture_mean, B, params_edit_B, flame, flametex, render,
#             )

#             loss_A = rendering_loss(
#                 args, k, texture_mean, A, params_edit_B, flame, flametex, render,
#             )
#             loss_B = rendering_loss(
#                 args, k, texture_mean, B, params_edit_A, flame, flametex, render,
#             )

    
            # 4) Consistency loss
            # Assign all parameters that should not have been changed back to original
            # and rerender.
            #
            params_consistency_A = np.array([x.detach().clone() for x in params_A])
            
            # Only want to assign what should not have changed, which are all other semantic params
            # other than the label that was chosen.
            #
            for label_idx in [x for x in semantic_labels if x != semantic_param]:
                params_consistency_A[label_idx] = params_AB[label_idx]
                
            loss_consistency_A = rendering_loss(
                args, k, texture_mean, A, params_consistency_A, flame, flametex, render
            )
            
            
            


            # Losses
            # ------------------------------------------------
            total_loss = 0.0
            

#             loss_edit = loss_edit_A['all_loss'] + loss_edit_B['all_loss']
            loss_edit = loss_edit_A['all_loss']
            loss_edit *= 10.0
        
            loss_consistency = loss_consistency_A['all_loss']
            loss_consistency *= 10.0


            total_loss += loss_recon
            total_loss += loss_edit 
            total_loss += loss_consistency


            optim.zero_grad()
            total_loss.backward()
            optim.step()

            
            all_loss = 0.0
#             for key in loss_A.keys():
#                 all_loss = all_loss + loss_edit_A[key] + loss_edit_B[key]
#                 losses_to_plot[key].append(loss_edit_[key].item() + loss_edit_B[key].item()) # Store for plotting later.
            for key in loss_edit_A.keys():
                all_loss = all_loss + loss_edit_A[key]
                losses_to_plot[key].append(loss_edit_A[key].item()) # Store for plotting later.

            losses_to_plot['all_loss'].append(all_loss)

            
            pbar.set_description(
                    (
                        f"total: {total_loss:.4f}; edit: {loss_edit:.4f}; recon: {loss_recon:.4f}; "
                        f"consistency: {loss_consistency:.4f}; "
                    )
                )

#         if k % 100 == 0:
#             print("epoch: ", k, ", loss_edit: ", total_loss.item(), ", loss_recon: ", loss_recon.item())
    

        if k % modulo_save_imgs == 0:
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
            
            if params_edit_B is not None:
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

        # Save current model.
        #
        if k % modulo_save_model == 0:
            save_checkpoint(
                savefolder=savefolder,
                epoch=k,
                losses=losses_to_plot,
                model=rignet,
                optim=optim,
                distributed=args.distributed)
            
    # Save current model.
    #
    save_checkpoint(
        savefolder=savefolder,
        epoch=k,
        losses=losses_to_plot,
        model=rignet,
        optim=optim,
        distributed=args.distributed)


                
                
def train_distributed(args, config, savefolder):
    
    if not os.path.exists(savefolder):
        os.makedirs(savefolder, exist_ok=True)
    
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    
    
#     synchronize()
    args.local_rank = get_rank()
    
    
    # this is the total # of GPUs across all nodes
    # if using 2 nodes with 4 GPUs each, world size is 8
    args.world_size = torch.distributed.get_world_size()
    print("### global rank of curr node: {} of {}".format(get_rank(), get_world_size()))

        
    
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    #
    torch.cuda.set_device(args.local_rank)
    
    
    # Load flame and render models.
    #
    render = load.renderer(args.base_path, model_params).cuda(args.local_rank)
    flame = load.flame(args.base_path, model_params).cuda(args.local_rank)
    flametex = load.flametex(args.base_path, model_params).cuda(args.local_rank)

    
    # Load DFR, and RigNet models.
    #
    one_hot = True
    rignet = load.rignet(args.base_path, one_hot=one_hot, training=True).cuda(args.local_rank)
    dfr = load.dfr(args.base_path, load_weights=True).cuda(args.local_rank)
    
    
    # Used to regularize textures.
    #
    texture_mean = flametex.get_texture_mean(args.batch_size) / 255.
    texture_mean = texture_mean.cuda(args.local_rank)
    
    
    save_batch_size = args.save_batch_size
    batch_size = args.batch_size
    workers = args.workers
    modulo_save_imgs = args.iter_save_img
    modulo_save_model = args.iter_save_ckpt
    epochs = args.iter
    
    
    loss_mse = nn.MSELoss().cuda(args.local_rank)
    optim = torch.optim.Adam(
                rignet.parameters(),
                lr=1e-3,
    )
    
#     train_dir_A = f'{args.base_path}/train_data/testing_A'
#     train_dir_B = f'{args.base_path}/train_data/testing_B'
    train_dir_A = f'{args.base_path}/train_data/rignet_A'
    train_dir_B = f'{args.base_path}/train_data/rignet_B'
    if args.local_rank == 0:
        print("train_dir: ", train_dir_A)
        print("train_dir: ", train_dir_B)
        
        
    dataset = DatasetSiamese(train_dir_A, train_dir_B)
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.cuda.device_count(),
#             num_replicas=4,
            rank = args.local_rank,
        )
    
    
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
    
    
    pbar = tqdm(range(0, epochs), dynamic_ncols=True, smoothing=0.01)
    for k in pbar:
        if args.distributed:
            sampler.set_epoch(k)
            
            
        for A, B in loader:
            try:
                w_A = A['latents'].cuda(args.local_rank)
                landmarks_2d_gt_A = A['landmarks_2d_gt'].cuda(args.local_rank)
                images_A = A['images'].cuda(args.local_rank)
                image_masks_A = A['image_masks'].cuda(args.local_rank)

                w_B = B['latents'].cuda(args.local_rank)
                landmarks_2d_gt_B = B['landmarks_2d_gt'].cuda(args.local_rank)
                images_B = B['images'].cuda(args.local_rank)
                image_masks_B = B['image_masks'].cuda(args.local_rank)

            except:
                print('Skipping A, B = loader')
                continue
                
                
            ### TODO:
            ### - Once this is verified to properly transfer pose, alter below to randomly
            ###   decide with semantic parameter is choosen, that we can conditionally
            ###   decide what to transfer using calls to RigNet that go beyond pose.
            ###
            # Create a labels to properly map the latent space with a label, and what is
            # being "edited" below based on the output of the DFR network.
            #
            # e.g. shape, expression, pose, tex, cam, lights = dfr(latents.view(args.batch_size, -1))
            #
            # onehot_cam = F.one_hot(torch.tensor([4]), 6).unsqueeze(0)
            onehot_pose = F.one_hot(torch.tensor([2]), 6).unsqueeze(0)
            # labels_in = torch.cat((onehot_pose, onehot_expr), dim=0).cuda()
            labels_in = onehot_pose
            labels_in = labels_in.expand(batch_size, -1, -1).cuda(args.local_rank)
    
    
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
                args, k, texture_mean, A, params_edit_A, flame, flametex, render,
            )
            loss_B = cycle_editing_loss(
                args, k, texture_mean, B, params_edit_B, flame, flametex, render,
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
            
        # At the end of each epoch, check if we should save rendering/output
        # and/or checkpoint.
        #
        if args.local_rank == 0:    
            if k % modulo_save_imgs == 0:
#                 synchronize()
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
                    
            # Save current model.
            #
            if k % modulo_save_model == 0:
                save_checkpoint(
                    savefolder=savefolder,
                    epoch=k,
                    losses=losses_to_plot,
                    model=rignet,
                    optim=optim,
                    distributed=args.distributed)

                
                
    if args.local_rank == 0:
        if k % modulo_save_model == 0:
            save_checkpoint(
                savefolder=savefolder,
                epoch=k,
                losses=losses_to_plot,
                model=dfr,
                optim=optim,
                distributed=args.distributed)



    




        
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


    
    args = parser.parse_args()
    assert args.batch_size >= args.save_batch_size
    
    
    device = 'cuda'
    
    
    with open(f'{args.base_path}/photometric_optimization/configs/config.yaml') as f:
        config = yaml.safe_load(f)

        
    path_pretrained = f'{args.base_path}/pretrained_models'
    model_params = config['model_params']
    model_params['flame_model_path'] = f'{path_pretrained}/generic_model.pkl'
    model_params['flame_lmk_embedding_path'] = f'{path_pretrained}/landmark_embedding.npy'
    model_params['tex_space_path'] = f'{path_pretrained}/FLAME_texture.npz'
    

    if args.distributed:
        print("Distributed training...")
#         from torch.multiprocessing import set_start_method
# #         try:
# #             set_start_method('spawn')
# #         except RuntimeError:
# #             pass
        
        # Create unique directory to save results.
        #
        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H.%M.%S")     # dd/mm/YY H:M:S
        savefolder = os.path.sep.join(['./test_results', f'{dt_string}'])
        if not os.path.exists(savefolder):
            os.makedirs(savefolder, exist_ok=True)

#         dfr = None
#         for trunc_dir in truncation_dirs:
#             dfr = train_distributed(args, config, savefolder, train_dir=trunc_dir, dfr=dfr)

        train_distributed(args, model_params, savefolder)
        dist.destroy_process_group()
    else:
        print("Beginning non-distributed training...")
    #     train_dir_A = f'{args.base_path}/train_data/rignet_A'
    #     train_dir_B = f'{args.base_path}/train_data/rignet_B'
        train_dir_A = f'{args.base_path}/train_data/testing_A'
        train_dir_B = f'{args.base_path}/train_data/testing_B'
        print("train_dir: ", train_dir_A)
        print("train_dir: ", train_dir_B)
        dataset = DatasetSiamese(train_dir_A, train_dir_B)

        batch_size = args.batch_size
        sampler = None
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

        # Load flame and render models.
        #
        render = load.renderer(args.base_path, model_params).cuda()
        flame = load.flame(args.base_path, model_params).cuda()
        flametex = load.flametex(args.base_path, model_params).cuda()

        # Load DFR, and RigNet models.
        #
        one_hot = True
        rignet = load.rignet(args.base_path, one_hot=one_hot, training=True).cuda()
        dfr = load.dfr(args.base_path, load_weights=True, training=False).cuda()


        # Kick off the training.
        #
        device = 'cuda'
        train(args, model_params, loader, rignet, dfr, flame, flametex, render, device)        
        
    
    
# --------------------------------------------------
#
# Example commandline run:
#
# CUDA_VISIBLE_DEVICES=0 python train_rignet.py --save_batch_size=5 --batch_size=5 --iter=3 --iter_save_img=1 --workers=4 --base_path=/home/ec2-user/SageMaker

# Distributed:
# - example with 4 GPUs on 1 node.
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_rignet.py --save_batch_size=5 --batch_size=32 --iter=1000 --iter_save_img=25 --iter_save_ckpt=25 --base_path=/home/ec2-user/SageMaker --workers=7 --distributed

# Debug:
# You can see what all the threads are doing “sudo gdb -p … thread apply all bt"