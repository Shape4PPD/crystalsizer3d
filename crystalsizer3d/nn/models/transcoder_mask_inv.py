import math

import torch
import torch.nn as nn
from torch.nn import Parameter

from crystalsizer3d.crystal import ROTATION_MODE_AXISANGLE, ROTATION_MODE_QUATERNION
from crystalsizer3d.nn.dataset import Dataset
from crystalsizer3d.nn.models.transcoder import Transcoder


class TranscoderMaskInv(Transcoder):
    mask: torch.Tensor

    def __init__(
            self,
            latent_size: int,
            param_size: int,
            ds: Dataset,
            latent_activation: str = 'none',
            normalise_latents: bool = True,
    ):
        super().__init__(latent_size, param_size, latent_activation)
        self.ds = ds
        self.normalise_latents = normalise_latents
        self.register_buffer('mask', None)
        self.weight = Parameter(torch.zeros(latent_size, param_size))
        self._build_model()

    def _build_model(self):
        """
        Initialise the weight matrix and mask.
        """
        self._init_params()
        self._init_mask()

    def _init_params(self):
        """
        Initialise the weight matrix with orthogonal rows.
        """
        W = self.weight
        nn.init.orthogonal_(W)

    def _init_mask(self):
        """
        Build a mask to ensure independent parameter groups don't overlap in the latent space.
        """
        n_latents_per_param = math.floor(self.latent_size / self.param_size)
        mask = torch.zeros(self.latent_size, self.param_size)
        row_idx = 0
        col_idx = 0

        def update_mask(n_params):
            nonlocal row_idx, col_idx, mask
            n_rows = n_latents_per_param * n_params
            mask[row_idx:row_idx + n_rows, col_idx:col_idx + n_params] = 1
            row_idx += n_rows
            col_idx += n_params

        if self.ds.ds_args.train_zingg:
            update_mask(2)

        if self.ds.ds_args.train_distances:
            n_dist_params = len(self.ds.labels_distances_active)
            if self.ds.ds_args.use_distance_switches:
                n_dist_params *= 2
            update_mask(n_dist_params)

        if self.ds.ds_args.train_transformation:
            n_trans_params = len(self.ds.labels_transformation)
            if self.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
                n_trans_params += len(self.ds.labels_rotation_quaternion)
            else:
                assert self.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE
                n_trans_params += len(self.ds.labels_rotation_axisangle)
            update_mask(n_trans_params)

        if self.ds.ds_args.train_material:
            update_mask(len(self.ds.labels_material))

        if self.ds.ds_args.train_light:
            n_light_params = len(self.ds.labels_light_active)
            update_mask(n_light_params)

        self.mask = mask

    def to_parameters(self, z: torch.Tensor, **kwargs):
        """
        Convert a latent vector to a parameter vector.
        """
        z = self.latent_activation_fn(z)  # Assume input z is logits
        self.latents_in = z
        if self.normalise_latents:
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        W = self.weight * self.mask
        p = torch.einsum('ij,bi->bj', W, z)
        return p

    def to_latents(self, p: torch.Tensor, activate: bool = True, **kwargs):
        """
        Convert a parameter vector to a latent vector.
        """
        W = self.weight * self.mask
        Z = torch.einsum('ji,bj->bi', torch.linalg.pinv(W), p)
        if activate:
            Z = self.latent_activation_fn(Z)
        self.latents_out = Z
        if self.normalise_latents:
            Z = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)
        return Z
