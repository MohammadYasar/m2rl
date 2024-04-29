from arm.optim.lamb import Lamb
from arm.utils import stack_on_channel, quaternion_to_discrete_euler, normalize_quaternion
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from agents.mdeit import MoveDit
from agents.qfunction import QFunction
from agents.mdeit_utils import _preprocess_inputs

class DiffuserActorAgent():
    def __init__(self,
                coordinate_bounds: list,
                perceiver_encoder: nn.Module,
                camera_names: list,
                batch_size: int,
                voxel_size: int,
                voxel_feature_size: int,
                num_rotation_classes: int,
                rotation_resolution: float,
                lr: float = 0.0001,
                image_resolution: list = None,
                lambda_weight_l2: float = 0.0,
                transform_augmentation: bool = True,
                transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                transform_augmentation_rot_resolution: int = 5,
                optimizer_type: str = 'lamb'):

        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._camera_names = camera_names
        self._batch_size = batch_size
        self._voxel_size = voxel_size
        self._voxel_feature_size = voxel_feature_size
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = transform_augmentation_xyz
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type

        self.vae, self.scheduler, self.unet = self.load_model('stabilityai/stable-diffusion-2-base')
        self.vae = self.vae.cuda()

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._mse_loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
       
        self._q = QFunction(self._perceiver_encoder,                            
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._optimizer_type == 'lamb':
            # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
            self._optimizer = Lamb(
                self._q.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif self._optimizer_type == 'adam':
            self._optimizer = torch.optim.Adam(
                self._q.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        elif self._optimizer_type == 'adamW':
            self._optimizer = torch.optim.AdamW(
            params=self._q.parameters(),
            lr=1e-4, weight_decay=1e-6)

            # Cosine LR schedule with linear warmup
            lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=500,
                num_training_steps=len(dataloader) * num_epochs
            )
        else:
            raise Exception('Unknown optimizer')

    def _softmax_q(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _get_one_hot_expert_actions(self,  # You don't really need this function since GT labels are already in the right format. This is some leftover code from my experiments with label smoothing.
                                    batch_size,
                                    action_trans,
                                    action_rot_grip,
                                    action_ignore_collisions,
                                    device):
        bs = batch_size

        # initialize with zero tensors
        action_trans_one_hot = torch.zeros((bs, self._voxel_size, self._voxel_size, self._voxel_size), dtype=int, device=device)
        action_rot_x_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_y_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_z_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_grip_one_hot  = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
          
          # ignore collision
          gt_ignore_collisions = action_ignore_collisions[b, :]
          action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        # flatten trans
        action_trans_one_hot = action_trans_one_hot.view(bs, -1)

        return action_trans_one_hot, \
               action_rot_x_one_hot, \
               action_rot_y_one_hot, \
               action_rot_z_one_hot, \
               action_grip_one_hot,  \
               action_collision_one_hot
    
    def load_model(self, model_id):
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae")
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet")
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]//3
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])  # (bs, 4, 4, 64, 64)

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    def _get_img_pcd(self, obs):
        images = [img for (img, pcd) in obs]
        pcds = [pcd for (img, pcd) in obs]
        images = torch.cat(images, 0)
        pcds = torch.cat(pcds, 0)
        return images, pcds

    
    def update(self, step: int, replay_sample: dict, backprop: bool = True) -> dict:
        # sample
        action_trans = replay_sample['trans_action_indicies'][:, -1, :3]#.int()
        action_rot_grip = replay_sample['rot_grip_action_indicies'][:, -1]#.int()
        action_ignore_collisions = replay_sample['ignore_collisions'][:, -1].int()
        action_gripper_pose = replay_sample['gripper_pose'][:, -1]        
        prev_gripper_pose = replay_sample['prev_gripper_pose'].squeeze(1)
        # print ("prev gripper pose ", prev_gripper_pose.shape, prev_gripper_pose[:, 3:])
        quat = prev_gripper_pose[:, 3:]
        # for quat_index in range(quat.shape[0]):
        #     if quat[quat_index].all() == 0:
        #         quat[quat_index,0] = 1e-2

        # quat = normalize_quaternion(quat.cpu().numpy())
        # quat = prev_gripper_pose[:, 3:].cpu().numpy()
        # print ("quat ", quat)
        # negative_mask = quat[:, -1] < 0
        # quat[negative_mask, :] *= -1

        # if quat[-1] < 0:
        #     quat = -quat
        # disc_rot = quaternion_to_discrete_euler(quat, resolution=5)
        # disc_rot = torch.from_numpy(disc_rot).cuda().unsqueeze(0)
        # quat = torch.from_numpy(quat).cuda()#.unsqueeze(0)
        proprio_new = torch.cat((prev_gripper_pose[:, :3], quat), dim=-1)
        proprio_new = torch.cat((proprio_new, torch.randn((proprio_new.shape[0], 2)).cuda()), dim=-1)
        
        # metric scene bounds
        bounds = bounds_tp1 = self._coordinate_bounds
        # inputs        
        # proprio = stack_on_channel(proprio_new)
        # proprio = torch.cat((proprio, proprio), dim=-1)
        
        obs, pcd = _preprocess_inputs(replay_sample)
        # TODO: data augmentation by applying SE(3) pertubations to pcd and actions
        # see https://github.com/peract/peract/blob/main/voxel/augmentation.py#L68 for reference

        # retrive batched image and depth
        images, pcds = self._get_img_pcd(obs)
        latent_images = self.encode_image(images, self.vae)
        latent_depth = self.encode_image(pcds, self.vae)

        # Q function
        q_trans, rot_grip_q, collision_q = self._q(latent_images,
                                                               proprio_new,
                                                               latent_depth,
                                                               bounds)
        # one-hot expert actions
        bs = self._batch_size
        

        action_trans_one_hot, action_rot_x_one_hot, \
        action_rot_y_one_hot, action_rot_z_one_hot, \
        action_grip_one_hot, action_collision_one_hot = self._get_one_hot_expert_actions(bs,
                                                                                         action_trans,
                                                                                         action_rot_grip,
                                                                                         action_ignore_collisions,
                                                                                         device=self._device)
        total_loss = 0.
        if backprop:
            # cross-entropy loss
            trans_loss = self._mse_loss(q_trans.view(bs, -1).float(),
                                                  action_trans.float())

            rot_grip_loss = 0.
            # print ("prediction: ",q_trans[:, :3].view(bs, -1).float(), "gt: ", action_trans.float())

            rot_grip_loss += self._mse_loss(rot_grip_q[:, :3].view(bs, -1).float(),
                                                  action_rot_grip[:, :3].float(),)
            rot_grip_loss += self._mse_loss(rot_grip_q[:, -1].float(), action_grip_one_hot.argmax(-1).float())#.item()


            collision_loss = self._cross_entropy_loss(collision_q,
                                                      action_collision_one_hot.argmax(-1))

            total_loss = trans_loss  + rot_grip_loss #+ collision_loss
            total_loss = total_loss.mean()
            # backprop
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

            total_loss = total_loss.item()
            

        # choose best action through argmax
        coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                                                          rot_grip_q,
                                                                                                          collision_q)
        # self._q.choose_highest_action(q_trans, rot_grip_q, collision_q)


        return {
            'total_loss': total_loss,
            'pred_action': {
                'trans': coords_indicies,                
                'rot_and_grip': rot_and_grip_indicies,
                'collision': ignore_collision_indicies
            },
            'expert_action': {
                'action_trans': action_trans
            }
        }


