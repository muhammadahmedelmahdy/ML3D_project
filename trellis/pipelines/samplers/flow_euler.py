from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from ...modules import sparse as sp


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)


class FlowEulerRepaintSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    RePaint-style conditioned sampler for flow matching models.

    At each timestep t the known region is replaced by x0_known noised to that
    time level via the flow-matching forward process:
        x_t^known = (1 - t) * x0 + (sigma_min + (1 - sigma_min) * t) * eps

    Optionally repeats num_resample_steps times per timestep: after each Euler
    step the sample is re-noised back to t from the predicted x_0, giving the
    model more chances to harmonize known and unknown regions (RePaint §4.2).

    Works for both dense tensors (Stage 1, sparse-structure flow) and
    SparseTensors (Stage 2, SLAT flow).
    """

    def _forward_to_t(self, feats: torch.Tensor, t: float) -> torch.Tensor:
        """Flow-matching forward process: noise a clean tensor to timestep t."""
        eps = torch.randn_like(feats)
        return (1.0 - t) * feats + (self.sigma_min + (1.0 - self.sigma_min) * t) * eps

    def _apply_repaint_mask(self, x_t, x0_known, mask, t: float):
        """
        Paste the known region (noised to t) into x_t.

        Dense path  – x_t: [N,C,H,W,D], x0_known: same, mask: [N,1,H,W,D]
        Sparse path – x_t: SparseTensor, x0_known: SparseTensor,
                      mask: [V] bool/float (one entry per voxel)
        """
        if isinstance(x_t, sp.SparseTensor):
            noised = self._forward_to_t(x0_known.feats, t)
            m = mask.float().unsqueeze(-1)          # [V, 1]
            return x_t.replace(m * noised + (1.0 - m) * x_t.feats)
        else:
            noised = self._forward_to_t(x0_known, t)
            return mask * noised + (1.0 - mask) * x_t

## do animated generation 
## play around with repaint (see something) --> have some kind of change on repaint (a bit of modifications (if you use softmax,stop masking at some point, repaint always does from start to end (play around with it)))
## baseline is repaint (think about the masking (early stop masking), actual bounded generation)
## crop out the voxels for each part (split them back after denoising)
## you have parts to extract (show that on visuals)
## improvement on traditional repaint


# 1- do animated generatopm (Nikola)
# 2- the masking of trellis( play with it ) -->Mahdi
# 3- in the pipeline, after generation, we should split the parts back to show that the layout is correct (jonas)
# we show the parts to extract them form the layout

# we compare against traditional repaint. (Mahdi)
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        x0_known,
        mask,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        num_resample_steps: int = 1,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate inpainted samples using the RePaint loop.

        Args:
            model: The flow model.
            noise: Initial noise — dense [N,C,H,W,D] or SparseTensor.
            cond: Positive conditioning.
            neg_cond: Negative conditioning for CFG.
            x0_known: Clean known region at t=0, same type/shape as noise.
            mask: Binary mask, 1 = known.
                  Dense: [N,1,H,W,D] (broadcast over channels).
                  Sparse: [V] bool or float (one value per voxel).
            steps: Euler denoising steps.
            rescale_t: Timestep rescaling factor.
            cfg_strength: CFG guidance scale.
            cfg_interval: Timestep range over which CFG is applied.
            num_resample_steps: RePaint U — resample iterations per timestep.
                                 1 = standard RePaint (no resampling loop).
            verbose: Show tqdm progress bar.

        Returns:
            EasyDict with keys:
              'samples'   – final inpainted tensor.
              'pred_x_t'  – list of x_{t-1} predictions per outer step.
              'pred_x_0'  – list of x_0 predictions per outer step.
        """
        is_sparse = isinstance(noise, sp.SparseTensor)
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="Sampling [RePaint]", disable=not verbose):
            for u in range(num_resample_steps):
                # ── 1. Inject known region noised to current level t ──────────
                sample = self._apply_repaint_mask(sample, x0_known, mask, t)

                # ── 2. One Euler denoising step (CFG via _inference_model) ────
                out = self.sample_once(
                    model, sample, t, t_prev, cond,
                    neg_cond=neg_cond,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                    **kwargs,
                )
                sample = out.pred_x_prev

                # ── 3. Re-noise from t_prev → t for next resample iteration ──
                if u < num_resample_steps - 1:
                    if is_sparse:
                        eps = torch.randn_like(out.pred_x_0.feats)
                        re_noised = (
                            (1.0 - t) * out.pred_x_0.feats
                            + (self.sigma_min + (1.0 - self.sigma_min) * t) * eps
                        )
                        sample = sample.replace(re_noised)
                    else:
                        eps = torch.randn_like(out.pred_x_0)
                        sample = (
                            (1.0 - t) * out.pred_x_0
                            + (self.sigma_min + (1.0 - self.sigma_min) * t) * eps
                        )

            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(out.pred_x_0)

        # Final paste: set known region exactly to x0_known (t=0, noise ≈ 0)
        sample = self._apply_repaint_mask(sample, x0_known, mask, 0.0)
        ret.samples = sample
        return ret
