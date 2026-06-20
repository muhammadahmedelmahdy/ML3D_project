from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from .. import models as trellis_models
from ..modules import sparse as sp


class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    # ──────────────────────────────────────────────────────────────────────────
    # RePaint inpainting helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_repaint_sampler(self, sigma_min: float) -> samplers.FlowEulerRepaintSampler:
        """Return a cached FlowEulerRepaintSampler for the given sigma_min."""
        if not hasattr(self, '_repaint_sampler_cache'):
            self._repaint_sampler_cache: dict = {}
        if sigma_min not in self._repaint_sampler_cache:
            self._repaint_sampler_cache[sigma_min] = samplers.FlowEulerRepaintSampler(sigma_min=sigma_min)
        return self._repaint_sampler_cache[sigma_min]

    def load_sparse_structure_encoder(self, path: str) -> None:
        """
        Load the sparse-structure VAE encoder needed for inpainting.

        The encoder is not part of the default generation pipeline, so it must
        be loaded explicitly before calling run_inpaint().

        Args:
            path: Local path or HF hub path to the encoder checkpoint
                  (without .json / .safetensors extension), e.g.
                  "microsoft/TRELLIS-text-large/ckpts/ss_enc_conv3d_16l8_fp16".
        """
        encoder = trellis_models.from_pretrained(path)
        encoder.eval().to(self.device)
        self.models['sparse_structure_encoder'] = encoder

    @torch.no_grad()
    def encode_sparse_structure_latent(self, occupancy_vol: torch.Tensor) -> torch.Tensor:
        """
        Encode a binary occupancy volume to the sparse-structure latent space.

        Args:
            occupancy_vol: Float tensor [N, 1, 64, 64, 64] with values in {0, 1}.

        Returns:
            Latent tensor [N, C, reso, reso, reso] where C and reso match the
            flow model's in_channels and resolution (typically 8 and 16).
        """
        assert 'sparse_structure_encoder' in self.models, (
            "Sparse-structure encoder not loaded. "
            "Call load_sparse_structure_encoder(path) first."
        )
        return self.models['sparse_structure_encoder'](
            occupancy_vol.float().to(self.device), sample_posterior=False
        )

    def coords_to_occupancy(self, coords: torch.Tensor, reso: int = 64) -> torch.Tensor:
        """
        Convert sparse voxel coordinates to a dense binary occupancy volume.

        Args:
            coords: Int tensor [M, 3] (x, y, z) or [M, 4] (batch, x, y, z).
            reso: Side length of the output volume.

        Returns:
            Float tensor [1, 1, reso, reso, reso].
        """
        vol = torch.zeros(1, 1, reso, reso, reso, dtype=torch.float32, device=self.device)
        if coords.shape[1] == 4:
            vol[0, 0, coords[:, 1], coords[:, 2], coords[:, 3]] = 1.0
        else:
            vol[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        return vol

    def downsample_mask(self, mask_3d: torch.Tensor, target_reso: int) -> torch.Tensor:
        """
        Downsample a binary 3D mask to the latent resolution using max pooling.

        A latent cell is considered *known* if any of the input voxels it covers
        is marked as known.

        Args:
            mask_3d: Float tensor [1, 1, H, W, D] with values in {0, 1}.
            target_reso: Target resolution (e.g. 16 for the SS latent space).

        Returns:
            Float tensor [1, 1, target_reso, target_reso, target_reso].
        """
        src_reso = mask_3d.shape[2]
        kernel = src_reso // target_reso
        if kernel <= 1:
            return mask_3d.float()
        return F.max_pool3d(mask_3d.float(), kernel_size=kernel, stride=kernel)

    def coords_to_slat_mask(
        self,
        coords: torch.Tensor,
        known_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a per-voxel boolean mask indicating which entries in *coords* are
        present in *known_coords* (i.e. belong to the known region).

        Args:
            coords: Int tensor [M, 4] (batch, x, y, z) — full voxel set.
            known_coords: Int tensor [K, 3] (x, y, z) — known voxels.

        Returns:
            Bool tensor [M] — True where the voxel is known.
        """
        # Convert known_coords to a set of (x,y,z) tuples for O(1) look-up
        known_set = set(map(tuple, known_coords.cpu().tolist()))
        mask = torch.tensor(
            [tuple(c[1:].tolist()) in known_set for c in coords.cpu()],
            dtype=torch.bool,
            device=self.device,
        )
        return mask

    # ──────────────────────────────────────────────────────────────────────────
    # RePaint sampling — Stage 1: sparse structure
    # ──────────────────────────────────────────────────────────────────────────

    def sample_sparse_structure_repaint(
        self,
        cond: dict,
        x0_known: torch.Tensor,
        structure_mask: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structure with RePaint conditioning on known regions.

        Args:
            cond: Conditioning dict from get_cond().
            x0_known: Known structure in *latent* space [1, C, reso, reso, reso].
            structure_mask: Binary mask at latent resolution [1, 1, reso, reso, reso],
                            1 = keep this cell.
            num_samples: Number of independent samples.
            sampler_params: Extra kwargs forwarded to FlowEulerRepaintSampler.sample()
                            (e.g. num_resample_steps, cfg_strength, …).

        Returns:
            Sparse coordinates [M, 4] (batch, x, y, z).
        """
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)

        if num_samples > 1:
            x0_known = x0_known.expand(num_samples, -1, -1, -1, -1)
            structure_mask = structure_mask.expand(num_samples, -1, -1, -1, -1)

        sampler = self._get_repaint_sampler(self.sparse_structure_sampler.sigma_min)
        params = {**self.sparse_structure_sampler_params, **sampler_params}

        z_s = sampler.sample(
            flow_model,
            noise,
            x0_known=x0_known,
            mask=structure_mask,
            **cond,
            **params,
            verbose=True,
        ).samples

        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        return coords

    # ──────────────────────────────────────────────────────────────────────────
    # RePaint sampling — Stage 2: structured latent
    # ──────────────────────────────────────────────────────────────────────────

    def sample_slat_repaint(
        self,
        cond: dict,
        coords: torch.Tensor,
        slat0_known: sp.SparseTensor,
        slat_mask: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample SLAT with RePaint conditioning on known voxel features.

        Args:
            cond: Conditioning dict from get_cond().
            coords: Voxel coordinates [M, 4] (batch, x, y, z).
            slat0_known: Known SLAT as a *normalized* SparseTensor (8 channels),
                         sharing the same coords as *coords*.  Typically obtained
                         from a prior TRELLIS generation before denormalization.
            slat_mask: Bool tensor [M] — True for voxels whose features are known.
            sampler_params: Extra kwargs for FlowEulerRepaintSampler.sample().

        Returns:
            Decoded (denormalized) SLAT SparseTensor ready for decode_slat().
        """
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )

        sampler = self._get_repaint_sampler(self.slat_sampler.sigma_min)
        params = {**self.slat_sampler_params, **sampler_params}

        slat = sampler.sample(
            flow_model,
            noise,
            x0_known=slat0_known,
            mask=slat_mask,
            **cond,
            **params,
            verbose=True,
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat

    # ──────────────────────────────────────────────────────────────────────────
    # Top-level inpainting entry point
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run_inpaint(
        self,
        prompt: str,
        mesh: o3d.geometry.TriangleMesh,
        structure_mask_3d: torch.Tensor,
        slat_known: Optional[sp.SparseTensor] = None,
        slat_mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Inpaint a 3D asset using RePaint-style conditioning on a known region.

        Stage 1 always runs RePaint on the sparse-structure flow model, preserving
        the voxel occupancy of the known region while generating new geometry
        elsewhere.

        Stage 2 optionally runs RePaint on the SLAT flow model when slat_known
        and slat_mask are provided (e.g. from a previous TRELLIS generation).
        Otherwise the SLAT is generated from scratch at the new coordinates.

        Args:
            prompt: Text description of the desired output.
            mesh: Known partial mesh.  Its voxelized occupancy defines x0_known
                  for Stage 1.
            structure_mask_3d: Binary volume [1, 1, 64, 64, 64] in float32,
                               1 = keep this voxel's latent cell.
            slat_known: Optional normalized SLAT SparseTensor (8 channels) from a
                        prior generation.  Required when slat_mask is given.
            slat_mask: Optional bool tensor [M] marking known voxels in the SLAT.
                       Required when slat_known is given.
            num_samples: Number of independent outputs to generate.
            seed: RNG seed.
            sparse_structure_sampler_params: Forwarded to sample_sparse_structure_repaint()
                                             (e.g. num_resample_steps=10, steps=250).
            slat_sampler_params: Forwarded to sample_slat_repaint() or sample_slat().
            formats: Output representation formats for decode_slat().

        Returns:
            dict with keys matching *formats* (e.g. 'mesh', 'gaussian', …).
        """
        assert 'sparse_structure_encoder' in self.models, (
            "Sparse-structure encoder not loaded. "
            "Call load_sparse_structure_encoder(path) before run_inpaint()."
        )

        cond = self.get_cond([prompt])
        torch.manual_seed(seed)

        # ── Stage 1: RePaint on sparse structure ──────────────────────────────
        known_voxels = self.voxelize(mesh)                         # [K, 3]
        known_occ = self.coords_to_occupancy(known_voxels)         # [1,1,64,64,64]
        x0_known_latent = self.encode_sparse_structure_latent(known_occ)

        flow_reso = self.models['sparse_structure_flow_model'].resolution
        structure_mask_latent = self.downsample_mask(
            structure_mask_3d.to(self.device), flow_reso
        )

        coords = self.sample_sparse_structure_repaint(
            cond,
            x0_known_latent,
            structure_mask_latent,
            num_samples,
            sparse_structure_sampler_params,
        )

        # ── Stage 2: SLAT RePaint or unconditional generation ────────────────
        if slat_known is not None and slat_mask is not None:
            slat = self.sample_slat_repaint(
                cond, coords, slat_known, slat_mask, slat_sampler_params
            )
        else:
            slat = self.sample_slat(cond, coords, slat_sampler_params)

        return self.decode_slat(slat, formats)
