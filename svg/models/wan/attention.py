import json
import sys
import warnings
from typing import Optional

import flashinfer
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch.nn.attention.flex_attention import (
    flex_attention,
)

from ...kernels.triton.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
from ...kernels.triton.rmsnorm import triton_rmsnorm_forward
from ...kmeans_utils import (
    batch_kmeans_Euclid,
    density_calculation,
    dynamic_block_sparse_fwd_flashinfer,
    identify_dynamic_map,
)
from ...logger import logger
from ...timer import time_logging_decorator
from ...utils.misc import Color
from .placement import (
    ref_wan_hidden_states_placement,
    ref_wan_sparse_head_placement,
    wan_hidden_states_placement,
    wan_sparse_head_placement,
)
from .utils import (
    create_block_mask_cached,
    flashinfer_sparse_attn_forward,
    gen_temporal_mask,
    generate_temporal_head_mask_mod,
)

try:
    # raise ImportError  # TODO: Remove this
    sys.path.append("svg/kernels/build/")
    import _kernels

    def apply_rotary_emb(query: torch.Tensor, key: torch.Tensor, freqs: torch.Tensor):
        freqs_real, freqs_imag = freqs
        _kernels.apply_qk_rope_inplace_cossin_complex(query, key, freqs_real, freqs_imag, 0)  # len_text_prompt = 0
        return query, key

    ENABLE_FAST_KERNEL = True

    logger.info(f"{Color.green}Using Fast CUDA and Triton Kernels{Color.reset}")


except ImportError:
    warnings.warn("Could not import RoPE / Norm kernels! Falling back to PyTorch implementation.")

    def apply_rotary_emb(query: torch.Tensor, key: torch.Tensor, freqs: torch.Tensor):
        def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
            x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
            x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
            return x_out.type_as(hidden_states)

        query = _apply_rotary_emb(query, freqs)
        key = _apply_rotary_emb(key, freqs)
        return query, key

    ENABLE_FAST_KERNEL = False

    logger.info(f"{Color.red}Disable Fast CUDA and Triton Kernels{Color.reset}")

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


class WanAttn_SVGAttn_Processor2_0:
    version = None
    context_length = 0
    num_frame = 0
    frame_size = 0

    first_layers_fp = 0
    first_times_fp = 0

    num_sampled_rows = 32
    attention_masks = None
    sparsity = 0

    block_mask = None
    temporal_mask_metadata = None

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    @time_logging_decorator("Level 2 - qkv")
    def get_qkv(self, attn, hidden_states, encoder_hidden_states):
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        return query, key, value

    @time_logging_decorator("Level 2 - qk_norm")
    def get_qk_norm(self, attn, query, key):
        if attn.norm_q is not None:
            if isinstance(attn.norm_q, torch.nn.RMSNorm) or isinstance(attn.norm_q, DiffusersRMSNorm):
                # query = attn.norm_q(query)
                query = triton_rmsnorm_forward(query, attn.norm_q.weight, attn.norm_q.eps)
            else:
                raise ValueError(f"Unsupported norm type: {type(attn.norm_q)}")

        if attn.norm_k is not None:
            if isinstance(attn.norm_k, torch.nn.RMSNorm) or isinstance(attn.norm_k, DiffusersRMSNorm):
                # key = attn.norm_k(key)
                key = triton_rmsnorm_forward(key, attn.norm_k.weight, attn.norm_k.eps)
            else:
                raise ValueError(f"Unsupported norm type: {type(attn.norm_k)}")
        return query, key

    @time_logging_decorator("Level 2 - transpose")
    def get_transpose_qkv(self, attn, query, key, value):
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        return query, key, value

    @time_logging_decorator("Level 2 - rotary_emb")
    def get_rotary_emb(self, query, key, rotary_emb):

        if rotary_emb is not None:
            query, key = apply_rotary_emb(query, key, rotary_emb)

        return query, key

    @time_logging_decorator("Level 2 - output")
    def get_o(self, attn, query, hidden_states, hidden_states_img):
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query, key, value = self.get_qkv(attn, hidden_states, encoder_hidden_states)

        query, key = self.get_qk_norm(attn, query, key)

        query, key, value = self.get_transpose_qkv(attn, query, key, value)

        query, key = self.get_rotary_emb(query, key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # # ============================== Save QKV ==============================
        # save_flag = timestep[0] % 4 == 0 and self.layer_idx % 4 == 0
        # print(f"save_flag: {save_flag}, timestep: {timestep[0]}, layer_idx: {self.layer_idx}")
        # save_dir = f"assets/svg_tensors"
        # if save_flag:
        #     save_qkvx(query, key, value, hidden_states, save_dir, self.layer_idx, timestep[0].item())

        # ========================================================================
        if timestep is None:  # Cross Attention in Wan
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:  # The main attention
            hidden_states = self.attention_core_logic(query, key, value, timestep)
        # ========================================================================

        hidden_states = self.get_o(attn, query, hidden_states, hidden_states_img)

        return hidden_states

    @time_logging_decorator("Level 3 - sample mse")
    def sample_mse(self, query, key, value):
        assert len(self.attention_masks) == 2

        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(self.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=self.sample_mse_max_row, size=(num_sampled_rows,))
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)

        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = torch.zeros(len(self.attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(self.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses

    @time_logging_decorator("Level 3 - sparse flex attention")
    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    @time_logging_decorator("Level 3 - sparse flashinfer attention")
    def sparse_flashinfer_attention(self, query, key, value, temporal_mask_metadata):
        return flashinfer_sparse_attn_forward(query, key, value, temporal_mask_metadata)

    @time_logging_decorator("Level 3 - sparse head placement")
    def sparse_head_placement(
        self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
    ):
        query_out, key_out, value_out = ref_wan_sparse_head_placement(
            query, key, value, best_mask_idx, context_length, num_frame, frame_size
        )
        return query_out, key_out, value_out

    @time_logging_decorator("Level 3 - fast sparse head placement")
    def fast_sparse_head_placement(
        self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
    ):
        wan_sparse_head_placement(
            query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
        )
        return query_out, key_out, value_out

    @time_logging_decorator("Level 3 - hidden states placement")
    def hidden_states_placement(
        self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
    ):
        ref_wan_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

    @time_logging_decorator("Level 3 - fast hidden states placement")
    def fast_hidden_states_placement(
        self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
    ):
        wan_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

    @time_logging_decorator("Level 3 - Dense Flash Attention")
    def flash_attention(self, query, key, value):
        output_hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        return output_hidden_states

    @time_logging_decorator("Level 2 - attention core logic")
    def attention_core_logic(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestep,
    ):
        cfg, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep[0] > self.first_times_fp:
            full_attention_flag = True

        if full_attention_flag:
            output_hidden_states = self.flash_attention(query, key, value)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
        else:
            sampled_mses = self.sample_mse(query, key, value)
            best_mask_idx = torch.argmin(sampled_mses, dim=0)

            output_hidden_states = torch.zeros_like(query)
            query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

            query_out, key_out, value_out = self.fast_sparse_head_placement(
                query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
            )

            hidden_states = self.sparse_flex_attention(query_out, key_out, value_out, block_mask=self.block_mask)
            # hidden_states = self.sparse_flashinfer_attention(query_out, key_out, value_out, temporal_mask_metadata=self.temporal_mask_metadata)

            self.fast_hidden_states_placement(
                hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
            )

            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)


def prepare_flexattention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

    seq_len = context_length + num_frame * frame_size
    query, key, value = [
        torch.zeros((cfg_size, num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)
    ]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)
    _ = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask


def prepare_flashinfer_attention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

    temporal_mask_metadata = gen_temporal_mask(num_frame, frame_size, multiplier)

    return temporal_mask_metadata


# ---- Semantic Aware Permutation Processor ----
class WanAttn_SAPAttn_Processor(WanAttn_SVGAttn_Processor2_0):
    num_layers = 0
    num_q_centroids = 0
    num_k_centroids = 0
    top_p_kmeans = 0
    min_kc_ratio = 0

    centroids_init = False
    q_centroids = None
    k_centroids = None

    kmeans_iter_init = 0
    kmeans_iter_step = 0
    zero_step_kmeans_init = False

    logging_file = None

    @time_logging_decorator("Level 3.7 - kmeans init")
    def kmeans_init(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_q_centroids, max_iters=self.kmeans_iter_init
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_k_centroids, max_iters=self.kmeans_iter_init
        )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3.7 - kmeans step")
    def kmeans_step(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids,
        )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3.5 - kmeans clustering")
    def kmeans_clustering(self, query, key, layer_idx):
        if not self.centroids_init:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_init(
                query, key, layer_idx
            )
            self.centroids_init = True
            print(f"Centroids initialized at layer {layer_idx}. Init step: {self.kmeans_iter_init}")
        else:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_step(
                query, key, layer_idx
            )

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3 - semantic aware permutation")
    def semantic_aware_permutation(self, query, key, value):
        cfg, num_heads, seq_len, dim = query.size()

        # 1. Kmeans clustering
        qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_clustering(
            query, key, self.layer_idx
        )

        # 2. Identify dynamic map
        q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)

        dynamic_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        # 3. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, v_sorted_indices = permute_tensor_by_labels_triton(
            value, klabels, dim=2, sorted_indices=k_sorted_indices
        )

        return q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices

    @time_logging_decorator("Level 3 - Dense Flashinfer Attention")
    def flashinfer_attention(self, query, key, value):

        cfg, num_heads, seq_len, dim = query.size()

        query = query.flatten(0, 1).permute(1, 0, 2)
        key = key.flatten(0, 1).permute(1, 0, 2)
        value = value.flatten(0, 1).permute(1, 0, 2)

        o, o_lse = flashinfer.single_prefill_with_kv_cache(
            query,
            key,
            value,
            causal=False,
            return_lse=True,
        )

        o = o.permute(1, 0, 2).reshape(cfg, num_heads, seq_len, dim)

        return o

    @time_logging_decorator("Level 2 - attention core logic")
    def attention_core_logic(self, query, key, value, timestep):
        cfg, num_heads, seq_len, dim = query.size()
        assert cfg == 1, "Batch size must be 1 for kmeans block sparse attention"

        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep[0] > self.first_times_fp:
            full_attention_flag = True

        if full_attention_flag:
            if self.zero_step_kmeans_init:
                video_length = self.num_frame * self.frame_size
                query_video = query[:, :, :video_length, :].contiguous()
                key_video = key[:, :, :video_length, :].contiguous()
                self.kmeans_clustering(query_video, key_video, self.layer_idx)

            output_hidden_states = self.flash_attention(query, key, value)
            # output_hidden_states = self.flashinfer_attention(query, key, value)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)

        else:
            q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.semantic_aware_permutation(
                query, key, value
            )

            output_permuted = dynamic_block_sparse_fwd_flashinfer(
                q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False
            )

            attn_output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)

            # Save time, layer, density information to logging file
            if self.logging_file is not None:
                with time_logging_decorator("Level 3 - density calculation and logging"):
                    # 4. Calculate density
                    densities = density_calculation(dyn_map, qc_sz_s, kc_sz_s)

                    avg_density = densities.mean().item()
                    log_entry = {
                        "timestep": timestep[0].item(),
                        "layer": self.layer_idx,
                        "avg_density": avg_density,
                        "density": densities.tolist(),
                    }

                    # print(f"Time Step: {timestep[0].item()} Layer: {self.layer_idx} Density: {avg_density}")

                    with open(self.logging_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

            return attn_output.reshape(cfg, num_heads, seq_len, dim)


# ---- Selective Step Skipping (SSS) Processor ----
class WanAttn_SSSAttn_Processor(WanAttn_SVGAttn_Processor2_0):
    """
    SSS: Selective Step Skipping Attention Processor

    This processor implements training-free acceleration by:
    1. Identifying keyframes based on frame similarity after warmup
    2. Denoising keyframes every step, normal frames every skip_n steps
    3. Interpolating normal frames at non-denoising steps (provide K/V only)
    4. Using sparse attention mask: interpolated frames don't compute query output
    """

    # SSS configuration
    warmup_steps = 10
    num_keyframes = 12
    skip_n = 2
    similarity_window = 5
    keyframe_strategy = "cosine"  # or "fixed"
    keyframe_similarity_threshold = 0.98
    min_keyframe_gap = 1

    # Progressive SSS schedule. Values in sss_transition_points can be absolute
    # step offsets or ratios in (0, 1].
    sss_schedule = "progressive"
    sss_min_skip = 2
    sss_max_skip = 3
    sss_transition_points = None
    total_inference_steps = None

    # Runtime state
    keyframe_indices = None  # Class variable: shared across all layers
    logging_file = None
    step_counter = 0  # Class variable: track actual inference steps (not timesteps)
    last_timestep = None
    current_branch = 0
    branch_counter = 0

    # SVG sparse attention support (for SSS v2 optimization)
    use_svg_sparse = False  # Enable SVG sparse attention in denoise steps
    svg_block_mask = None  # FlexAttention block mask (reused from SVG)

    def __init__(self, layer_idx):
        super().__init__(layer_idx)
        # Note: keyframe_indices is a class variable shared across all layers
        # normal_frame_cache is per-layer and per CFG branch.
        self.normal_frame_cache = {}

    def get_normal_frame_cache(self) -> dict:
        return self.normal_frame_cache.setdefault(WanAttn_SSSAttn_Processor.current_branch, {})

    def update_normal_frame_cache(self, output: torch.Tensor, current_step: int):
        if WanAttn_SSSAttn_Processor.keyframe_indices is None:
            return

        branch_cache = self.get_normal_frame_cache()
        for frame_idx in range(self.num_frame):
            if frame_idx in WanAttn_SSSAttn_Processor.keyframe_indices:
                continue

            frame_tokens = slice(
                self.context_length + frame_idx * self.frame_size,
                self.context_length + (frame_idx + 1) * self.frame_size,
            )
            old_cache = branch_cache.get(frame_idx)
            if old_cache is not None and "latent_after" in old_cache:
                latent_before = old_cache["latent_after"]
                step_before = old_cache.get("step_after", current_step)
            else:
                latent_before = output[:, :, frame_tokens, :].clone()
                step_before = current_step

            branch_cache[frame_idx] = {
                "latent_before": latent_before,
                "latent_after": output[:, :, frame_tokens, :].clone(),
                "step_before": step_before,
                "step_after": current_step,
            }

    @staticmethod
    def get_progressive_skip_n(
        step_offset: int,
        total_sss_steps: int = None,
        min_skip: int = None,
        max_skip: int = None,
        transition_points: list = None,
    ) -> int:
        """
        Compute progressive skip_n based on current step offset.

        Implements progressive sparsity: start with moderate skipping, gradually become more sparse.
        This allows for better quality in early steps (when noise is high)
        while maximizing speedup in later steps (when noise is low).

        Args:
            step_offset: Current step minus warmup_steps (0, 1, 2, ...)
            total_sss_steps: Total number of SSS steps (default 40 for 50 total steps with 10 warmup)
            min_skip: Starting skip_n (default 2, skip 1 step in early phase)
            max_skip: Final skip_n (default 3, skip 2 steps in late phase)
            transition_points: List of [early_end, middle_end] in step_offset coordinates
                              Default [12, 28] means:
                                - offset 1-12: skip_n=2 (early phase, skip 1 step)
                                - offset 13-28: skip_n=3 (middle phase, skip 2 steps)
                                - offset 29+: skip_n=3 (late phase, skip 2 steps)

        Returns:
            current_skip_n: Integer in range [min_skip, max_skip]

        Examples:
            >>> WanAttn_SSSAttn_Processor.get_progressive_skip_n(5)   # Early phase
            2
            >>> WanAttn_SSSAttn_Processor.get_progressive_skip_n(15)  # Middle phase
            3
            >>> WanAttn_SSSAttn_Processor.get_progressive_skip_n(35)  # Late phase
            3
        """
        cls = WanAttn_SSSAttn_Processor
        if cls.sss_schedule == "fixed":
            return max(1, int(cls.skip_n))

        if total_sss_steps is None:
            if cls.total_inference_steps is not None:
                total_sss_steps = max(1, int(cls.total_inference_steps) - int(cls.warmup_steps))
            else:
                total_sss_steps = 40

        min_skip = int(cls.sss_min_skip if min_skip is None else min_skip)
        max_skip = int(cls.sss_max_skip if max_skip is None else max_skip)
        min_skip = max(1, min_skip)
        max_skip = max(min_skip, max_skip)

        if transition_points is None:
            transition_points = cls.sss_transition_points
        if transition_points is None:
            transition_points = [0.3, 0.7]

        # Validate transition points
        if len(transition_points) != 2:
            transition_points = [0.3, 0.7]

        early_end, middle_end = transition_points
        if 0 < early_end <= 1:
            early_end = int(round(early_end * total_sss_steps))
        if 0 < middle_end <= 1:
            middle_end = int(round(middle_end * total_sss_steps))

        # Ensure transition points are within valid range
        early_end = max(0, min(int(early_end), total_sss_steps))
        middle_end = max(early_end, min(int(middle_end), total_sss_steps))

        if step_offset < 0:
            # Should not happen, but return min_skip for safety
            return min_skip
        elif step_offset <= early_end:
            # Early phase: denoise every step for maximum quality
            return min_skip
        elif step_offset <= middle_end:
            # Middle phase: balanced approach
            return min_skip + 1
        else:
            # Late phase: maximize speedup
            return max_skip

    @time_logging_decorator("Level 3 - identify keyframes")
    def identify_keyframes_from_value(self, value: torch.Tensor) -> list:
        """
        Identify keyframes from value tensor at the end of warmup.

        Args:
            value: [cfg, num_heads, seq_len, dim]

        Returns:
            List of keyframe indices
        """
        from ...utils.keyframe_detection import (
            extract_frame_representations_from_value,
            identify_keyframes_cosine_similarity,
            identify_keyframes_fixed_interval,
            identify_keyframes_improved_distribution,
            identify_keyframes_adaptive_distribution,
            identify_keyframes_sequential_similarity,
            identify_keyframes_random,
            identify_keyframes_first,
            visualize_keyframe_selection,
        )

        # Extract per-frame representations
        frame_reps = extract_frame_representations_from_value(
            value, num_frames=self.num_frame, frame_size=self.frame_size, context_length=self.context_length
        )

        # Select keyframes based on strategy
        if self.keyframe_strategy == "adaptive":
            # Use new adaptive distribution algorithm for optimal uniformity
            keyframe_indices = identify_keyframes_adaptive_distribution(
                frame_reps, num_keyframes=self.num_keyframes, similarity_window=self.similarity_window
            )
        elif self.keyframe_strategy == "uniform":
            keyframe_indices = identify_keyframes_fixed_interval(
                num_frames=self.num_frame, num_keyframes=self.num_keyframes
            )
        elif self.keyframe_strategy in ("sequential", "semantic"):
            keyframe_indices = identify_keyframes_sequential_similarity(
                frame_reps,
                num_keyframes=self.num_keyframes,
                similarity_threshold=self.keyframe_similarity_threshold,
                min_gap=self.min_keyframe_gap,
            )
        elif self.keyframe_strategy == "cosine":
            # Use improved distribution algorithm for better uniformity
            keyframe_indices = identify_keyframes_improved_distribution(
                frame_reps, num_keyframes=self.num_keyframes, similarity_window=self.similarity_window
            )
        elif self.keyframe_strategy == "cosine_original":
            # Original cosine similarity algorithm (for comparison)
            keyframe_indices = identify_keyframes_cosine_similarity(
                frame_reps, num_keyframes=self.num_keyframes, similarity_window=self.similarity_window
            )
        elif self.keyframe_strategy == "fixed":
            keyframe_indices = identify_keyframes_fixed_interval(
                num_frames=self.num_frame, num_keyframes=self.num_keyframes
            )
        elif self.keyframe_strategy == "random":
            keyframe_indices = identify_keyframes_random(
                num_frames=self.num_frame, num_keyframes=self.num_keyframes
            )
        elif self.keyframe_strategy == "first":
            keyframe_indices = identify_keyframes_first(
                num_frames=self.num_frame, num_keyframes=self.num_keyframes
            )
        else:
            raise ValueError(f"Unknown keyframe strategy: {self.keyframe_strategy}")

        # Log keyframe selection
        logger.info(f"[SSS] Layer {self.layer_idx}: Identified {len(keyframe_indices)} keyframes: {keyframe_indices}")
        vis = visualize_keyframe_selection(keyframe_indices, self.num_frame)
        logger.info(f"[SSS] Keyframe visualization:\n{vis}")

        # Save to logging file if specified
        if self.logging_file is not None:
            log_entry = {"event": "keyframe_identification", "layer": self.layer_idx, "keyframes": keyframe_indices}
            with open(self.logging_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return keyframe_indices

    @time_logging_decorator("Level 3 - interpolate normal frames")
    def interpolate_normal_frames(
        self, key: torch.Tensor, value: torch.Tensor, current_step: int, denoising_frames: set
    ) -> tuple:
        """
        Interpolate K/V for normal frames that are not denoising at this step.

        Args:
            key: [cfg, num_heads, seq_len, dim]
            value: [cfg, num_heads, seq_len, dim]
            current_step: Current diffusion timestep
            denoising_frames: Set of frame indices that are denoising this step

        Returns:
            Modified (key, value) with interpolated frames
        """
        for frame_idx in range(self.num_frame):
            # Skip keyframes and currently denoising frames
            if frame_idx in WanAttn_SSSAttn_Processor.keyframe_indices or frame_idx in denoising_frames:
                continue

            # This is a normal frame that needs interpolation
            branch_cache = self.get_normal_frame_cache()
            if frame_idx not in branch_cache:
                logger.warning(f"[SSS] Frame {frame_idx} not in cache, skipping interpolation")
                continue

            cache = branch_cache[frame_idx]
            latent_before = cache["latent_before"]
            latent_after = cache["latent_after"]
            step_before = cache["step_before"]
            step_after = cache["step_after"]

            # Calculate interpolation alpha
            # Linear interpolation between step_before and step_after
            alpha = (step_before - current_step) / (step_before - step_after + 1e-8)
            alpha = torch.clamp(torch.tensor(alpha), 0.0, 1.0).item()

            # Interpolate
            interpolated = (1 - alpha) * latent_before + alpha * latent_after

            # Update K/V for this frame
            frame_tokens = slice(self.context_length + frame_idx * self.frame_size,
                                self.context_length + (frame_idx + 1) * self.frame_size)
            key[:, :, frame_tokens, :] = interpolated
            value[:, :, frame_tokens, :] = interpolated

        return key, value

    @time_logging_decorator("Level 3 - create SSS attention mask")
    def create_sss_attention_mask(self, denoising_frames: set, device: torch.device) -> torch.Tensor:
        """
        Create sparse attention mask for SSS.

        Mask pattern:
        - Denoising frames (query rows): True (can attend to all frames)
        - Interpolated frames (query rows): False (masked out, don't compute output)

        Args:
            denoising_frames: Set of frame indices that are denoising
            device: Device to create mask on

        Returns:
            attention_mask: [seq_len, seq_len] boolean tensor
        """
        seq_len = self.context_length + self.num_frame * self.frame_size
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Context tokens (if any) can always attend
        if self.context_length > 0:
            mask[:self.context_length, :] = True

        # Frame tokens: only denoising frames can attend
        for frame_idx in denoising_frames:
            q_start = self.context_length + frame_idx * self.frame_size
            q_end = self.context_length + (frame_idx + 1) * self.frame_size
            # These query rows can attend to all frames
            mask[q_start:q_end, :] = True

        # Interpolated frames' query rows remain False (masked out)

        return mask

    @time_logging_decorator("Level 2 - attention core logic")
    def attention_core_logic(self, query, key, value, timestep):
        """
        SSS attention core logic with selective step skipping.
        """
        cfg, num_heads, seq_len, dim = query.shape
        current_timestep = timestep[0].item()  # This is the diffusion timestep (e.g., 956)

        # Increment step counter only when a new scheduler timestep starts. With
        # classifier-free guidance, the pipeline calls the transformer twice with
        # the same timestep; those two branches must share the same denoising
        # step number but keep separate SSS caches.
        if self.layer_idx == 0:
            timestep_key = int(current_timestep)
            if WanAttn_SSSAttn_Processor.last_timestep != timestep_key:
                WanAttn_SSSAttn_Processor.last_timestep = timestep_key
                WanAttn_SSSAttn_Processor.step_counter += 1
                WanAttn_SSSAttn_Processor.branch_counter = 0
            else:
                WanAttn_SSSAttn_Processor.branch_counter += 1
            WanAttn_SSSAttn_Processor.current_branch = WanAttn_SSSAttn_Processor.branch_counter

        current_step = WanAttn_SSSAttn_Processor.step_counter  # This is the actual step number (e.g., 1, 2, 3...)

        assert (
            seq_len == self.context_length + self.num_frame * self.frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {self.context_length} + {self.num_frame} * {self.frame_size}"

        full_attention_flag = False
        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep[0] > self.first_times_fp:
            full_attention_flag = True

        # ===== Phase 1: Warmup / configured dense fallback =====
        if current_step <= self.warmup_steps or full_attention_flag:
            output_hidden_states = self.flash_attention(query, key, value)

            # At the end of warmup, identify keyframes (only once, in layer 0)
            if (
                current_step == self.warmup_steps
                and self.layer_idx == 0
                and WanAttn_SSSAttn_Processor.keyframe_indices is None
            ):
                WanAttn_SSSAttn_Processor.keyframe_indices = self.identify_keyframes_from_value(value)

            if current_step >= self.warmup_steps and WanAttn_SSSAttn_Processor.keyframe_indices is not None:
                self.update_normal_frame_cache(output_hidden_states, current_step)

            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)

        # ===== Phase 2: SSS - Progressive Selective Step Skipping =====
        # Key optimization: Only compute attention for denoising frames, interpolate others
        # Progressive strategy: Early dense, middle balanced, late sparse

        # Determine which frames denoise at this step
        step_offset = current_step - self.warmup_steps

        # ===== PROGRESSIVE SKIP_N IMPLEMENTATION =====
        # Compute progressive skip_n based on current step offset
        current_skip_n = self.get_progressive_skip_n(step_offset)

        denoising_frames = set(WanAttn_SSSAttn_Processor.keyframe_indices)  # Keyframes always denoise

        # Normal frames denoise based on progressive skip_n
        is_normal_denoise_step = (step_offset % current_skip_n == 0)
        if is_normal_denoise_step:
            for frame_idx in range(self.num_frame):
                if frame_idx not in WanAttn_SSSAttn_Processor.keyframe_indices:
                    denoising_frames.add(frame_idx)

        # Log progressive skipping behavior
        if self.layer_idx == 0 and WanAttn_SSSAttn_Processor.current_branch == 0:  # Only log once per step
            logger.info(f"[SSS] Step {current_step} (offset={step_offset}): "
                       f"skip_n={current_skip_n}, "
                       f"normal_denoise={is_normal_denoise_step}, "
                       f"denoising_frames={len(denoising_frames)}/{self.num_frame}")

        # Prepare output tensor
        output = torch.zeros(cfg, num_heads, seq_len, dim, device=query.device, dtype=query.dtype)

        # ===== STEP 1.4: Use SVG sparse in denoise steps =====
        use_svg = self.use_svg_sparse and is_normal_denoise_step and self.svg_block_mask is not None

        if use_svg:
            # Denoise step with SVG spatial sparse attention
            # Use parent class (SVG) sparse attention method
            output = self.sparse_flex_attention(query, key, value, block_mask=self.svg_block_mask)

            if self.layer_idx == 0 and WanAttn_SSSAttn_Processor.current_branch == 0:
                logger.info(f"[SSS v2] Denoise step: Using SVG spatial sparse attention")
        else:
            # Skip step or SVG disabled: use simple per-frame dense attention
            for frame_idx in denoising_frames:
                frame_tokens = slice(self.context_length + frame_idx * self.frame_size,
                                   self.context_length + (frame_idx + 1) * self.frame_size)
                q_frame = query[:, :, frame_tokens, :]

                # Use full K/V to preserve context (no cat/copy to avoid OOM)
                attn_output = F.scaled_dot_product_attention(
                    q_frame, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                output[:, :, frame_tokens, :] = attn_output

        # ===== Interpolate output for non-denoising frames =====
        non_denoising_frames = set(range(self.num_frame)) - denoising_frames
        for frame_idx in non_denoising_frames:
            frame_tokens = slice(self.context_length + frame_idx * self.frame_size,
                               self.context_length + (frame_idx + 1) * self.frame_size)

            # Get cached outputs from previous denoising steps
            branch_cache = self.get_normal_frame_cache()
            cache = branch_cache.get(frame_idx)
            if cache is not None:
                output_before = cache.get("latent_after")  # Use "after" from last denoise as "before" for interpolation
                output_after = cache.get("latent_after")   # Same value (will be updated on next denoise)
                step_before = cache.get("step_after")
                step_after = cache.get("step_after")

                # For now, just use the cached output directly (will be properly interpolated after next denoise)
                interpolated_output = output_after
                output[:, :, frame_tokens, :] = interpolated_output

        # ===== Update cache for frames that just denoised =====
        # Update cache for normal frames (rolling cache strategy)
        if is_normal_denoise_step:
            self.update_normal_frame_cache(output, current_step)

        # ===== TODO-2 FIX: Removed keyframe cache update =====
        # Keyframes recompute every step, no need to cache them!
        # This saves 12 frames × 30 layers × 88MB × 50 steps = ~79GB memory

        # Log denoising schedule
        if (
            self.logging_file is not None
            and self.layer_idx == 0
            and WanAttn_SSSAttn_Processor.current_branch == 0
        ):
            log_entry = {
                "event": "denoising_step",
                "step": current_step,
                "timestep": current_timestep,
                "step_offset": step_offset,
                "denoising_frames": sorted(list(denoising_frames)),
                "num_denoising": len(denoising_frames),
            }
            with open(self.logging_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return output.reshape(cfg, num_heads, seq_len, dim)
