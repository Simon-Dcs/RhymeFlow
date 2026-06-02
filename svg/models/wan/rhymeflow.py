import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils import is_torch_xla_available

from ...logger import logger
from ...utils.keyframe_detection import (
    identify_keyframes_adaptive_distribution,
    identify_keyframes_fixed_interval,
    identify_keyframes_random,
    identify_keyframes_first,
    identify_keyframes_sequential_similarity,
    visualize_keyframe_selection,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


@dataclass
class RhymeFlowConfig:
    warmup_steps: int = 5
    num_keyframes: int = 5
    keyframe_strategy: str = "semantic"
    keyframe_similarity_threshold: float = 0.98
    min_keyframe_gap: int = 1
    sss_schedule: str = "progressive"
    sss_min_skip: int = 2
    sss_max_skip: int = 3
    sss_transition_points: Optional[Sequence[float]] = None
    projection_space: str = "sigma"
    projection_mode: str = "linear"
    foca_blend: float = 0.5
    foca_min_skip: int = 4
    context_mode: str = "latent"
    async_context_mode: str = "full"
    no_keyframes: bool = False
    solver: str = "euler"
    logging_file: Optional[str] = None


def _json_log(path: Optional[str], entry: Dict[str, Any]) -> None:
    if path is None:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _progressive_skip_n(step_offset: int, total_steps: int, config: RhymeFlowConfig) -> int:
    if config.sss_schedule == "fixed":
        return max(1, int(config.sss_min_skip))

    min_skip = max(1, int(config.sss_min_skip))
    max_skip = max(min_skip, int(config.sss_max_skip))
    transition_points = config.sss_transition_points
    if transition_points is None or len(transition_points) != 2:
        transition_points = [0.3, 0.7]

    early_end, middle_end = transition_points
    if 0 < early_end <= 1:
        early_end = int(round(early_end * total_steps))
    if 0 < middle_end <= 1:
        middle_end = int(round(middle_end * total_steps))

    early_end = max(0, min(int(early_end), total_steps))
    middle_end = max(early_end, min(int(middle_end), total_steps))

    if step_offset <= early_end:
        return min_skip
    if step_offset <= middle_end:
        return min(min_skip + 1, max_skip)
    return max_skip


def _frame_representations_from_latents(latents: torch.Tensor) -> torch.Tensor:
    # latents: [B, C, F, H, W]. Current experiments use B=1.
    reps = latents.detach().float().mean(dim=0).permute(1, 0, 2, 3).flatten(1)
    return reps


def _select_keyframes(clean_latents: torch.Tensor, config: RhymeFlowConfig) -> List[int]:
    frame_reps = _frame_representations_from_latents(clean_latents)
    num_frames = frame_reps.shape[0]
    strategy = config.keyframe_strategy

    if strategy in ("sequential", "semantic", "cosine"):
        keyframes = identify_keyframes_sequential_similarity(
            frame_reps,
            num_keyframes=config.num_keyframes,
            similarity_threshold=config.keyframe_similarity_threshold,
            min_gap=config.min_keyframe_gap,
        )
    elif strategy in ("uniform", "fixed", "adaptive"):
        # Adaptive distribution from the previous SSS experiments is a uniform
        # budgeted baseline. It is useful for isolating scheduler effects.
        if strategy == "adaptive":
            keyframes = identify_keyframes_adaptive_distribution(frame_reps, config.num_keyframes)
        else:
            keyframes = identify_keyframes_fixed_interval(num_frames, config.num_keyframes)
    elif strategy == "random":
        keyframes = identify_keyframes_random(num_frames, config.num_keyframes)
    elif strategy == "first":
        keyframes = identify_keyframes_first(num_frames, config.num_keyframes)
    else:
        raise ValueError(f"Unsupported RhymeFlow keyframe strategy: {strategy}")

    keyframes = sorted({int(idx) for idx in keyframes if 0 <= int(idx) < num_frames})
    if not keyframes:
        keyframes = [0]
    return keyframes


def _frame_token_indices(
    frame_indices: Sequence[int],
    frame_size: int,
    device: torch.device,
) -> torch.Tensor:
    indices = []
    for frame_idx in frame_indices:
        start = int(frame_idx) * frame_size
        indices.append(torch.arange(start, start + frame_size, device=device, dtype=torch.long))
    return torch.cat(indices, dim=0)


def _apply_rotary_emb_subset(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
    return x_out.type_as(hidden_states)


def _active_self_attention(
    attn,
    query_states: torch.Tensor,
    key_value_states: torch.Tensor,
    rotary_q: torch.Tensor,
    rotary_k: torch.Tensor,
) -> torch.Tensor:
    if attn.add_k_proj is not None:
        raise NotImplementedError("RhymeFlow active Wan forward currently targets T2V self-attention only.")

    query = attn.to_q(query_states)
    key = attn.to_k(key_value_states)
    value = attn.to_v(key_value_states)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
    key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
    value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

    query = _apply_rotary_emb_subset(query, rotary_q)
    key = _apply_rotary_emb_subset(key, rotary_k)

    hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
    hidden_states = hidden_states.type_as(query_states)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def _active_block_forward(
    block,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    active_token_indices: torch.Tensor,
    non_active_token_indices: torch.Tensor,
    key_value_token_indices: Optional[torch.Tensor] = None,
    context_cache: Optional[List[torch.Tensor]] = None,
    layer_idx: int = 0,
) -> torch.Tensor:
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        block.scale_shift_table + temb.float()
    ).chunk(6, dim=1)

    context_hidden_states = hidden_states
    if context_cache is not None and non_active_token_indices.numel() > 0:
        context_hidden_states = hidden_states.clone()
        cached_hidden_states = context_cache[layer_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
        if cached_hidden_states.shape[1] == non_active_token_indices.numel():
            cached_non_active = cached_hidden_states
        else:
            cached_non_active = cached_hidden_states.index_select(1, non_active_token_indices)
        context_hidden_states.index_copy_(
            1,
            non_active_token_indices,
            cached_non_active,
        )

    norm_hidden_states = (block.norm1(context_hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
        hidden_states
    )
    norm_active = norm_hidden_states.index_select(1, active_token_indices)

    rotary_q = rotary_emb.index_select(2, active_token_indices)
    if key_value_token_indices is None:
        key_value_states = norm_hidden_states
        rotary_k = rotary_emb
    else:
        key_value_states = norm_hidden_states.index_select(1, key_value_token_indices)
        rotary_k = rotary_emb.index_select(2, key_value_token_indices)
    attn_output = _active_self_attention(block.attn1, norm_active, key_value_states, rotary_q, rotary_k)

    active_hidden_states = hidden_states.index_select(1, active_token_indices)
    active_hidden_states = (active_hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

    norm_active = block.norm2(active_hidden_states.float()).type_as(active_hidden_states)
    attn_output = block.attn2(hidden_states=norm_active, encoder_hidden_states=encoder_hidden_states)
    active_hidden_states = active_hidden_states + attn_output

    norm_active = (block.norm3(active_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
        active_hidden_states
    )
    ff_output = block.ffn(norm_active)
    active_hidden_states = (active_hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

    hidden_states = hidden_states.clone()
    hidden_states.index_copy_(1, active_token_indices, active_hidden_states)
    return hidden_states


def wan_transformer_forward_active_frames(
    transformer,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    active_frames: Sequence[int],
    kv_frames: Optional[Sequence[int]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    context_cache: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        logger.warning("RhymeFlow active forward ignores attention_kwargs['scale'] without PEFT backend.")

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    if p_t != 1:
        raise NotImplementedError("RhymeFlow active frame forward currently assumes temporal patch_size == 1.")

    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    frame_size = post_patch_height * post_patch_width
    active_token_indices = _frame_token_indices(active_frames, frame_size, hidden_states.device)
    active_frame_set = set(int(idx) for idx in active_frames)
    key_value_token_indices = None
    if kv_frames is not None:
        key_value_token_indices = _frame_token_indices(kv_frames, frame_size, hidden_states.device)
    non_active_frames = [idx for idx in range(post_patch_num_frames) if idx not in active_frame_set]
    if non_active_frames:
        non_active_token_indices = _frame_token_indices(non_active_frames, frame_size, hidden_states.device)
    else:
        non_active_token_indices = torch.empty(0, device=hidden_states.device, dtype=torch.long)

    rotary_emb = transformer.rope(hidden_states)

    hidden_states = transformer.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
        timestep, encoder_hidden_states, None
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    for layer_idx, block in enumerate(transformer.blocks):
        hidden_states = _active_block_forward(
            block,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            active_token_indices,
            non_active_token_indices,
            key_value_token_indices=key_value_token_indices,
            context_cache=context_cache,
            layer_idx=layer_idx,
        )

    shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    active_hidden_states = hidden_states.index_select(1, active_token_indices)
    active_hidden_states = (transformer.norm_out(active_hidden_states.float()) * (1 + scale) + shift).type_as(
        active_hidden_states
    )
    active_hidden_states = transformer.proj_out(active_hidden_states)

    out_channels = transformer.config.out_channels
    output_seq = hidden_states.new_zeros(
        batch_size, post_patch_num_frames * post_patch_height * post_patch_width, out_channels * p_t * p_h * p_w
    )
    output_seq.index_copy_(1, active_token_indices, active_hidden_states)

    output = output_seq.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    output = output.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return output.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def wan_transformer_forward_full_with_cache(
    transformer,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    cache_token_indices: Optional[torch.Tensor] = None,
    cache_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        logger.warning("RhymeFlow cached full forward ignores attention_kwargs['scale'] without PEFT backend.")

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = transformer.rope(hidden_states)
    hidden_states = transformer.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
        timestep, encoder_hidden_states, None
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    cache: List[torch.Tensor] = []
    for block in transformer.blocks:
        if cache_token_indices is None:
            cached_hidden_states = hidden_states.detach()
        else:
            cached_hidden_states = hidden_states.index_select(1, cache_token_indices).detach()
        if cache_device is None:
            cache.append(cached_hidden_states.clone())
        else:
            cache.append(cached_hidden_states.to(device=cache_device, copy=True))
        try:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, timestep=timestep)
        except TypeError:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = transformer.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return output, cache


def _flow_euler_update(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigmas: torch.Tensor,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    dt = (sigmas[end_idx] - sigmas[start_idx]).to(device=sample.device, dtype=torch.float32)
    while dt.ndim < sample.ndim:
        dt = dt.view(*dt.shape, *([1] * (sample.ndim - dt.ndim)))
    return sample.float() + dt * model_output.float()


def _scheduler_step(pipe, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    return pipe.scheduler.step(model_output, timestep, sample, return_dict=False)[0]


def _flow_velocity_between(
    start: torch.Tensor,
    end: torch.Tensor,
    sigmas: torch.Tensor,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    dt = (sigmas[end_idx] - sigmas[start_idx]).to(device=start.device, dtype=torch.float32)
    while dt.ndim < start.ndim:
        dt = dt.view(*dt.shape, *([1] * (start.ndim - dt.ndim)))
    return (end.float() - start.float()) / dt


def _projection_delta(
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    start_idx: int,
    end_idx: int,
    projection_space: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if projection_space == "timestep":
        start_value = timesteps[start_idx].float()
        end_value = timesteps[end_idx].float()
    else:
        start_value = sigmas[start_idx].float()
        end_value = sigmas[end_idx].float()
    dt = (end_value - start_value).to(device=reference.device, dtype=torch.float32)
    while dt.ndim < reference.ndim:
        dt = dt.view(*dt.shape, *([1] * (reference.ndim - dt.ndim)))
    return dt


def _foca_project_from_history(
    start: torch.Tensor,
    prev_full: Optional[torch.Tensor],
    current_velocity: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    prev_full_idx: Optional[int],
    start_idx: int,
    target_idx: int,
    projection_space: str,
    blend: float,
) -> torch.Tensor:
    if target_idx <= start_idx:
        return start

    # The Wan flow prediction is defined over sigma, so the FoCa-style
    # predictor-corrector always integrates in sigma space.
    delta_space = "sigma"
    euler = start.float() + _projection_delta(
        sigmas,
        timesteps,
        start_idx,
        target_idx,
        delta_space,
        start,
    ) * current_velocity.float()

    if prev_full is None or prev_full_idx is None or prev_full_idx >= start_idx:
        return euler

    prev_dt = _projection_delta(
        sigmas,
        timesteps,
        prev_full_idx,
        start_idx,
        delta_space,
        start,
    )
    target_dt = _projection_delta(
        sigmas,
        timesteps,
        start_idx,
        target_idx,
        delta_space,
        start,
    )
    ratio = target_dt / (prev_dt + 1.0e-8)
    forecast = (
        start.float()
        + (ratio / 3.0) * (start.float() - prev_full.float())
        + (2.0 / 3.0) * target_dt * current_velocity.float()
    )
    blend = float(max(0.0, min(1.0, blend)))
    return blend * forecast + (1.0 - blend) * euler


def _foca_heun_correct(
    start: torch.Tensor,
    start_velocity: torch.Tensor,
    end_velocity: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    start_idx: int,
    target_idx: int,
) -> torch.Tensor:
    dt = _projection_delta(
        sigmas,
        timesteps,
        start_idx,
        target_idx,
        "sigma",
        start,
    )
    return start.float() + 0.5 * dt * (start_velocity.float() + end_velocity.float())


def _effective_foca_blend(skip_span: int, base_blend: float, min_skip: int) -> float:
    if skip_span < max(1, int(min_skip)):
        return 0.0
    ramp = min(1.0, float(skip_span - int(min_skip) + 1) / float(max(1, int(min_skip))))
    return float(max(0.0, min(1.0, base_blend))) * ramp


def _project_between(
    start: torch.Tensor,
    end: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    start_idx: int,
    end_idx: int,
    current_idx: int,
    projection_space: str,
) -> torch.Tensor:
    if current_idx <= start_idx:
        return start
    if current_idx >= end_idx:
        return end

    if projection_space == "timestep":
        start_value = timesteps[start_idx].float()
        end_value = timesteps[end_idx].float()
        current_value = timesteps[current_idx].float()
    else:
        start_value = sigmas[start_idx].float()
        end_value = sigmas[end_idx].float()
        current_value = sigmas[current_idx].float()

    alpha = (start_value - current_value) / (start_value - end_value + 1.0e-8)
    alpha = torch.clamp(alpha, 0.0, 1.0).to(device=start.device, dtype=torch.float32)
    while alpha.ndim < start.ndim:
        alpha = alpha.view(*alpha.shape, *([1] * (start.ndim - alpha.ndim)))
    return (1.0 - alpha) * start.float() + alpha * end.float()


def _project_skip_state(
    start: torch.Tensor,
    end: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    start_idx: int,
    end_idx: int,
    current_idx: int,
    projection_space: str,
    projection_mode: str,
    current_velocity: Optional[torch.Tensor] = None,
    prev_full: Optional[torch.Tensor] = None,
    prev_full_idx: Optional[int] = None,
    foca_blend: float = 0.5,
) -> torch.Tensor:
    if projection_mode == "foca":
        if foca_blend <= 0.0:
            return _project_between(
                start,
                end,
                sigmas,
                timesteps,
                start_idx,
                end_idx,
                current_idx,
                projection_space,
            )
        return _project_between(
            start,
            end,
            sigmas,
            timesteps,
            start_idx,
            end_idx,
            current_idx,
            projection_space,
        )
    return _project_between(
        start,
        end,
        sigmas,
        timesteps,
        start_idx,
        end_idx,
        current_idx,
        projection_space,
    )


def _set_frames(base: torch.Tensor, frame_indices: Sequence[int], values: torch.Tensor) -> torch.Tensor:
    if len(frame_indices) == 0:
        return base
    frame_tensor = torch.tensor(frame_indices, device=base.device, dtype=torch.long)
    base = base.clone()
    base.index_copy_(2, frame_tensor, values)
    return base


def _predict_noise(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    guidance_scale: float,
    active_frames: Optional[Sequence[int]],
    kv_frames: Optional[Sequence[int]],
    attention_kwargs: Optional[Dict[str, Any]],
    collect_full_cache: bool = False,
    context_caches: Optional[Dict[str, List[torch.Tensor]]] = None,
    cache_token_indices: Optional[torch.Tensor] = None,
    cache_device: Optional[torch.device] = None,
):
    transformer_dtype = pipe.transformer.dtype
    latent_model_input = latents.to(transformer_dtype)
    timestep_batch = timestep.expand(latents.shape[0])

    if active_frames is None:
        caches: Dict[str, List[torch.Tensor]] = {}
        if collect_full_cache:
            noise_pred, caches["cond"] = wan_transformer_forward_full_with_cache(
                pipe.transformer,
                latent_model_input,
                timestep_batch,
                prompt_embeds,
                attention_kwargs=attention_kwargs,
                cache_token_indices=cache_token_indices,
                cache_device=cache_device,
            )
        else:
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
        if pipe.do_classifier_free_guidance:
            if collect_full_cache:
                noise_uncond, caches["uncond"] = wan_transformer_forward_full_with_cache(
                    pipe.transformer,
                    latent_model_input,
                    timestep_batch,
                    negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    cache_token_indices=cache_token_indices,
                    cache_device=cache_device,
                )
            else:
                noise_uncond = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        if collect_full_cache:
            return noise_pred.float(), caches
        return noise_pred.float()

    noise_pred = wan_transformer_forward_active_frames(
        pipe.transformer,
        latent_model_input,
        timestep_batch,
        prompt_embeds,
        active_frames=active_frames,
        kv_frames=kv_frames,
        attention_kwargs=attention_kwargs,
        context_cache=None if context_caches is None else context_caches.get("cond"),
    )
    if pipe.do_classifier_free_guidance:
        noise_uncond = wan_transformer_forward_active_frames(
            pipe.transformer,
            latent_model_input,
            timestep_batch,
            negative_prompt_embeds,
            active_frames=active_frames,
            kv_frames=kv_frames,
            attention_kwargs=attention_kwargs,
            context_cache=None if context_caches is None else context_caches.get("uncond"),
        )
        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
    return noise_pred.float()


@torch.no_grad()
def rhymeflow_wan_generate(
    pipe,
    prompt: Any = None,
    negative_prompt: Any = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: str = "latent",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    config: Optional[RhymeFlowConfig] = None,
) -> Tuple[WanPipelineOutput, Dict[str, Any]]:
    if output_type != "latent":
        raise ValueError("rhymeflow_wan_generate returns latents; decode outside the sampler.")
    config = config or RhymeFlowConfig()

    pipe.check_inputs(prompt, negative_prompt, height, width, prompt_embeds, negative_prompt_embeds, ["latents"])

    if num_frames % pipe.vae_scale_factor_temporal != 1:
        logger.warning(
            f"`num_frames - 1` has to be divisible by {pipe.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False

    device = pipe._execution_device
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)

    num_channels_latents = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    latent_frame_count = latents.shape[2]
    warmup_steps = max(0, min(int(config.warmup_steps), num_inference_steps - 1))
    total_after_warmup = max(1, num_inference_steps - warmup_steps)
    keyframes: Optional[List[int]] = None
    non_keyframes: List[int] = []
    clean_proxy = None
    full_forward_steps = 0
    active_forward_steps = 0
    skipped_frame_updates = 0
    schedule_events: List[Dict[str, Any]] = []
    analysis_keyframes: Optional[List[int]] = None
    last_context_caches: Optional[Dict[str, List[torch.Tensor]]] = None
    use_layer_context_cache = (
        config.context_mode in ("last_full", "last_full_nonkey", "last_full_nonkey_cpu")
        and config.async_context_mode == "full"
        and not config.no_keyframes
    )
    cache_device = torch.device("cpu") if config.context_mode == "last_full_nonkey_cpu" else None
    prev_full_latents: Optional[torch.Tensor] = None
    prev_full_idx: Optional[int] = None

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        i = 0
        while i < num_inference_steps:
            if pipe.interrupt:
                i += 1
                progress_bar.update()
                continue

            t = timesteps[i]
            pipe._current_timestep = t

            if i < warmup_steps or keyframes is None:
                collect_cache = False
                if collect_cache:
                    noise_pred, last_context_caches = _predict_noise(
                        pipe,
                        latents,
                        t,
                        prompt_embeds,
                    negative_prompt_embeds,
                    guidance_scale,
                    active_frames=None,
                    kv_frames=None,
                    attention_kwargs=attention_kwargs,
                    collect_full_cache=True,
                    cache_device=cache_device,
                    )
                else:
                    noise_pred = _predict_noise(
                        pipe,
                        latents,
                        t,
                        prompt_embeds,
                        negative_prompt_embeds,
                        guidance_scale,
                        active_frames=None,
                        kv_frames=None,
                        attention_kwargs=attention_kwargs,
                    )
                full_forward_steps += 1
                if i == warmup_steps - 1 or warmup_steps == 0:
                    clean_proxy = latents - sigmas[i].to(latents.device) * noise_pred
                    analysis_keyframes = _select_keyframes(clean_proxy, config)
                    analysis_keyframes = sorted(set(analysis_keyframes + [0, latent_frame_count - 1]))
                    analysis_keyframes = sorted(idx for idx in analysis_keyframes if 0 <= idx < latent_frame_count)
                    if config.no_keyframes:
                        keyframes = []
                        non_keyframes = list(range(latent_frame_count))
                    else:
                        keyframes = list(analysis_keyframes)
                        non_keyframes = [idx for idx in range(latent_frame_count) if idx not in set(keyframes)]
                    logger.info(f"[RhymeFlow] Analysis keyframes: {analysis_keyframes}")
                    if keyframes:
                        logger.info(f"[RhymeFlow] Active keyframes: {keyframes}")
                    else:
                        logger.info("[RhymeFlow] Active keyframes disabled for no-keyframe ablation.")
                    logger.info(
                        f"[RhymeFlow] Keyframe visualization:\n"
                        f"{visualize_keyframe_selection(analysis_keyframes, latent_frame_count)}"
                    )
                    _json_log(
                        config.logging_file,
                        {
                            "event": "keyframe_identification",
                            "step": i + 1,
                            "timestep": int(t.item()),
                            "analysis_keyframes": analysis_keyframes,
                            "active_keyframes": keyframes,
                            "num_latent_frames": latent_frame_count,
                            "no_keyframes": config.no_keyframes,
                        },
                    )

                current_full_latents = latents.detach().clone()
                if config.solver in ("scheduler", "scheduler_approx"):
                    latents = _scheduler_step(pipe, noise_pred, t, latents)
                else:
                    latents = _flow_euler_update(latents, noise_pred, sigmas, i, i + 1)
                prev_full_latents = current_full_latents
                prev_full_idx = i
                schedule_events.append(
                    {
                        "step": i + 1,
                        "timestep": int(t.item()),
                        "mode": "warmup_full",
                        "solver": config.solver,
                        "end_step": i + 2,
                    }
                )
                _json_log(config.logging_file, schedule_events[-1])
                i += 1
                progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()
                continue

            step_offset = i - warmup_steps + 1
            skip_n = _progressive_skip_n(step_offset, total_after_warmup, config)
            end_i = min(num_inference_steps, i + skip_n)

            keyframe_set: Set[int] = set(keyframes)
            non_keyframes = [idx for idx in range(latent_frame_count) if idx not in keyframe_set]
            cache_token_indices = None
            if use_layer_context_cache and non_keyframes:
                _, _, _, latent_h, latent_w = latents.shape
                _, p_h, p_w = pipe.transformer.config.patch_size
                frame_size = (latent_h // p_h) * (latent_w // p_w)
                cache_token_indices = _frame_token_indices(non_keyframes, frame_size, latents.device)

            if use_layer_context_cache:
                noise_pred, last_context_caches = _predict_noise(
                    pipe,
                    latents,
                    t,
                    prompt_embeds,
                    negative_prompt_embeds,
                    guidance_scale,
                    active_frames=None,
                    kv_frames=None,
                    attention_kwargs=attention_kwargs,
                    collect_full_cache=True,
                    cache_token_indices=cache_token_indices,
                    cache_device=cache_device,
                )
            else:
                noise_pred = _predict_noise(
                    pipe,
                    latents,
                    t,
                    prompt_embeds,
                    negative_prompt_embeds,
                    guidance_scale,
                    active_frames=None,
                    kv_frames=None,
                    attention_kwargs=attention_kwargs,
                )
            full_forward_steps += 1

            if config.solver in ("scheduler", "scheduler_approx") and end_i == i + 1:
                current_full_latents = latents.detach().clone()
                latents = _scheduler_step(pipe, noise_pred, t, latents)
                prev_full_latents = current_full_latents
                prev_full_idx = i
                event = {
                    "step": i + 1,
                    "timestep": int(t.item()),
                    "mode": "rhythmic_full",
                    "solver": config.solver,
                    "skip_n": 1,
                    "end_step": int(end_i + 1),
                    "keyframes": keyframes,
                    "num_non_keyframes": len(non_keyframes),
                }
                schedule_events.append(event)
                _json_log(config.logging_file, event)
                i = end_i
                progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()
                continue

            group_start_latents = latents.detach().clone()
            group_skip_span = int(end_i - i)
            group_foca_blend = _effective_foca_blend(group_skip_span, config.foca_blend, config.foca_min_skip)
            if config.projection_mode == "foca" and group_foca_blend > 0.0:
                group_end_latents = _foca_project_from_history(
                    start=group_start_latents,
                    prev_full=prev_full_latents,
                    current_velocity=noise_pred,
                    sigmas=sigmas,
                    timesteps=timesteps,
                    prev_full_idx=prev_full_idx,
                    start_idx=i,
                    target_idx=end_i,
                    projection_space=config.projection_space,
                    blend=group_foca_blend,
                )
                if end_i < num_inference_steps:
                    end_noise_pred = _predict_noise(
                        pipe,
                        group_end_latents,
                        timesteps[end_i],
                        prompt_embeds,
                        negative_prompt_embeds,
                        guidance_scale,
                        active_frames=None,
                        kv_frames=None,
                        attention_kwargs=attention_kwargs,
                    )
                    full_forward_steps += 1
                    group_end_latents = _foca_heun_correct(
                        start=group_start_latents,
                        start_velocity=noise_pred,
                        end_velocity=end_noise_pred,
                        sigmas=sigmas,
                        timesteps=timesteps,
                        start_idx=i,
                        target_idx=end_i,
                    )
            else:
                group_end_latents = _flow_euler_update(group_start_latents, noise_pred, sigmas, i, end_i)
            group_velocity = _flow_velocity_between(group_start_latents, group_end_latents, sigmas, i, end_i)

            # At the rhythmic point, all frames are available. Keyframes still
            # advance one step at a time; non-keyframes jump to the group end.
            if config.solver == "scheduler_approx":
                scheduler_step_latents = _scheduler_step(pipe, noise_pred, t, group_start_latents)
                key_values = scheduler_step_latents[:, :, keyframes]
            else:
                key_values = _flow_euler_update(
                    group_start_latents[:, :, keyframes],
                    noise_pred[:, :, keyframes],
                    sigmas,
                    i,
                    i + 1,
                )
            latents = _set_frames(group_end_latents, keyframes, key_values)
            skipped_frame_updates += max(0, end_i - i - 1) * len(non_keyframes)
            event = {
                "step": i + 1,
                "timestep": int(t.item()),
                "mode": "rhythmic_full",
                "solver": config.solver,
                "skip_n": int(end_i - i),
                "end_step": int(end_i + 1),
                "keyframes": keyframes,
                "num_non_keyframes": len(non_keyframes),
            }
            schedule_events.append(event)
            _json_log(config.logging_file, event)
            progress_bar.update()

            if config.no_keyframes:
                for j in range(i + 1, end_i):
                    t_inner = timesteps[j]
                    pipe._current_timestep = t_inner
                    projected_event = {
                        "step": j + 1,
                        "timestep": int(t_inner.item()),
                        "mode": "no_keyframes_projected",
                        "solver": config.solver,
                        "projected_frames": latent_frame_count,
                    }
                    schedule_events.append(projected_event)
                    _json_log(config.logging_file, projected_event)
                    progress_bar.update()
                    if XLA_AVAILABLE:
                        xm.mark_step()
                latents = group_end_latents
                prev_full_latents = group_start_latents
                prev_full_idx = i
                i = end_i
                if XLA_AVAILABLE:
                    xm.mark_step()
                continue

            for j in range(i + 1, end_i):
                t_inner = timesteps[j]
                pipe._current_timestep = t_inner
                projected = latents.clone()
                if non_keyframes:
                    projected_non_key = _project_skip_state(
                        start=group_start_latents[:, :, non_keyframes],
                        end=group_end_latents[:, :, non_keyframes],
                        sigmas=sigmas,
                        timesteps=timesteps,
                        start_idx=i,
                        end_idx=end_i,
                        current_idx=j,
                        projection_space=config.projection_space,
                        projection_mode=config.projection_mode,
                        current_velocity=noise_pred[:, :, non_keyframes],
                        prev_full=None if prev_full_latents is None else prev_full_latents[:, :, non_keyframes],
                        prev_full_idx=prev_full_idx,
                        foca_blend=group_foca_blend,
                    )
                    projected = _set_frames(projected, non_keyframes, projected_non_key)

                noise_key = _predict_noise(
                    pipe,
                    projected,
                    t_inner,
                    prompt_embeds,
                    negative_prompt_embeds,
                    guidance_scale,
                    active_frames=keyframes,
                    kv_frames=None if config.async_context_mode == "full" else keyframes,
                    attention_kwargs=attention_kwargs,
                    context_caches=last_context_caches if use_layer_context_cache else None,
                )
                active_forward_steps += 1
                if config.solver == "scheduler_approx":
                    scheduler_model_output = group_velocity.clone()
                    scheduler_model_output = _set_frames(
                        scheduler_model_output,
                        keyframes,
                        noise_key[:, :, keyframes],
                    )
                    scheduler_step_latents = _scheduler_step(pipe, scheduler_model_output, t_inner, projected)
                    key_values = scheduler_step_latents[:, :, keyframes]
                else:
                    key_values = _flow_euler_update(
                        latents[:, :, keyframes],
                        noise_key[:, :, keyframes],
                        sigmas,
                        j,
                        j + 1,
                    )
                latents = _set_frames(latents, keyframes, key_values)
                event = {
                    "step": j + 1,
                    "timestep": int(t_inner.item()),
                    "mode": "async_keyframes",
                    "solver": config.solver,
                    "projected_non_keyframes": len(non_keyframes),
                    "keyframes": keyframes,
                }
                schedule_events.append(event)
                _json_log(config.logging_file, event)
                progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

            # At the group end all frames should be synchronized at the same
            # scheduler index. Non-keyframes are already at the endpoint.
            if end_i > i + 1 and non_keyframes:
                latents = _set_frames(latents, non_keyframes, group_end_latents[:, :, non_keyframes])

            prev_full_latents = group_start_latents
            prev_full_idx = i
            i = end_i
            if XLA_AVAILABLE:
                xm.mark_step()

    pipe._current_timestep = None
    pipe.maybe_free_model_hooks()

    metadata = {
        "rhymeflow_keyframes": keyframes,
        "rhymeflow_analysis_keyframes": analysis_keyframes,
        "rhymeflow_num_latent_frames": latent_frame_count,
        "rhymeflow_full_forward_steps": full_forward_steps,
        "rhymeflow_active_forward_steps": active_forward_steps,
        "rhymeflow_skipped_frame_updates": skipped_frame_updates,
        "rhymeflow_schedule_events": schedule_events,
        "rhymeflow_projection_space": config.projection_space,
        "rhymeflow_projection_mode": config.projection_mode,
        "rhymeflow_foca_blend": config.foca_blend,
        "rhymeflow_foca_min_skip": config.foca_min_skip,
        "rhymeflow_context_mode": config.context_mode,
        "rhymeflow_async_context_mode": config.async_context_mode,
        "rhymeflow_no_keyframes": config.no_keyframes,
        "rhymeflow_solver": config.solver,
    }
    output = WanPipelineOutput(frames=latents)
    if not return_dict:
        return (latents,), metadata
    return output, metadata
