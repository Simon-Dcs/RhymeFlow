"""FoCa-style feature caching for Wan transformer blocks.

This is a pragmatic in-repo reproduction of FoCa's training-free
Forecast-then-Calibrate idea for Wan2.1. It wraps transformer blocks and
predicts skipped block outputs from cached hidden-feature history with a
BDF2-style forecast plus a damped Heun-style calibration.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from ...logger import logger


@dataclass
class FoCaCacheConfig:
    warmup_steps: int = 2
    cache_interval: int = 3
    start_layer: int = 0
    end_layer: Optional[int] = None
    blend: float = 0.75
    cache_device: str = "cuda"


@dataclass
class _FeatureHistory:
    prev: Optional[torch.Tensor] = None
    prev_step: Optional[int] = None
    curr: Optional[torch.Tensor] = None
    curr_step: Optional[int] = None
    full_updates: int = 0
    predicted_updates: int = 0


class FoCaRuntime:
    def __init__(self, transformer, config: FoCaCacheConfig):
        self.transformer = transformer
        self.config = config
        self.step_index = -1
        self.current_branch = 0
        self.current_timestep: Optional[int] = None
        self.current_full_step = True
        self.full_transformer_calls = 0
        self.predicted_transformer_calls = 0
        self.full_block_calls = 0
        self.predicted_block_calls = 0
        self.fallback_block_calls = 0
        self.histories: Dict[Tuple[int, int], _FeatureHistory] = {}

        num_layers = len(transformer.blocks)
        end_layer = config.end_layer if config.end_layer is not None else num_layers - 1
        self.start_layer = max(0, int(config.start_layer))
        self.end_layer = min(num_layers - 1, int(end_layer))
        if self.end_layer < self.start_layer:
            raise ValueError(f"Invalid FoCa layer range: {self.start_layer}..{self.end_layer}")

    def begin_forward(self, timestep: torch.Tensor) -> None:
        timestep_value = int(timestep.flatten()[0].detach().item())
        if self.current_timestep != timestep_value:
            self.step_index += 1
            self.current_branch = 0
            self.current_timestep = timestep_value
        else:
            self.current_branch += 1

        self.current_full_step = self._is_full_step(self.step_index)
        if self.current_full_step:
            self.full_transformer_calls += 1
        else:
            self.predicted_transformer_calls += 1

    def _is_full_step(self, step_index: int) -> bool:
        warmup_steps = max(0, int(self.config.warmup_steps))
        cache_interval = max(1, int(self.config.cache_interval))
        if step_index < warmup_steps:
            return True
        return (step_index - warmup_steps) % cache_interval == 0

    def _cache_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.config.cache_device == "cpu":
            return tensor.detach().to("cpu")
        return tensor.detach()

    def _history_key(self, layer_idx: int) -> Tuple[int, int]:
        return int(self.current_branch), int(layer_idx)

    def _store_full(self, layer_idx: int, output: torch.Tensor) -> None:
        key = self._history_key(layer_idx)
        history = self.histories.setdefault(key, _FeatureHistory())
        history.prev = history.curr
        history.prev_step = history.curr_step
        history.curr = self._cache_tensor(output)
        history.curr_step = self.step_index
        history.full_updates += 1
        self.full_block_calls += 1

    def _store_prediction(self, layer_idx: int, output: torch.Tensor) -> None:
        key = self._history_key(layer_idx)
        history = self.histories.setdefault(key, _FeatureHistory())
        # Predictions are used for the current skipped forward only. Keeping
        # the two latest fully computed features as the BDF2 anchors avoids
        # turning FoCa into an unbounded Taylor chain across a cache window.
        history.predicted_updates += 1
        self.predicted_block_calls += 1

    def _can_predict(self, layer_idx: int) -> bool:
        if layer_idx < self.start_layer or layer_idx > self.end_layer:
            return False
        history = self.histories.get(self._history_key(layer_idx))
        return (
            history is not None
            and history.curr is not None
            and history.prev is not None
            and history.curr_step is not None
            and history.prev_step is not None
            and history.curr_step < self.step_index
        )

    def _predict_feature(self, layer_idx: int, reference: torch.Tensor) -> torch.Tensor:
        history = self.histories[self._history_key(layer_idx)]
        assert history.curr is not None
        assert history.prev is not None
        assert history.curr_step is not None
        assert history.prev_step is not None

        curr = history.curr.to(device=reference.device, dtype=reference.dtype, non_blocking=True)
        prev = history.prev.to(device=reference.device, dtype=reference.dtype, non_blocking=True)

        prev_delta = max(1, int(history.curr_step) - int(history.prev_step))
        target_delta = max(1, int(self.step_index) - int(history.curr_step))
        ratio = float(target_delta) / float(prev_delta)

        # Eq. 7 style BDF2 predictor. The derivative term is approximated from
        # the most recent feature difference so the path stays training-free.
        feature_delta = curr - prev
        forecast = (4.0 / 3.0) * curr - (1.0 / 3.0) * prev + (2.0 / 3.0) * ratio * feature_delta

        # Damped Heun-style calibration. With only cached tensors available, the
        # end slope is estimated from the forecast. The blend controls how much
        # of that corrected jump is accepted, which dampens large-interval
        # overshoot in the same spirit as FoCa's calibrator.
        pred_slope = forecast - curr
        heun = curr + 0.5 * (ratio * feature_delta + pred_slope)
        blend = max(0.0, min(1.0, float(self.config.blend)))
        prediction = curr + blend * (heun - curr) + (1.0 - blend) * (forecast - curr)
        return prediction.to(dtype=reference.dtype)

    def block_forward(self, layer_idx, original_forward, block, hidden_states, encoder_hidden_states, temb, rotary_emb):
        if self.current_full_step or not self._can_predict(layer_idx):
            output = original_forward(hidden_states, encoder_hidden_states, temb, rotary_emb)
            if layer_idx < self.start_layer or layer_idx > self.end_layer:
                self.full_block_calls += 1
            else:
                self._store_full(layer_idx, output)
                if not self.current_full_step:
                    self.fallback_block_calls += 1
            return output

        prediction = self._predict_feature(layer_idx, hidden_states)
        self._store_prediction(layer_idx, prediction)
        return prediction

    def metadata(self) -> Dict[str, object]:
        cached = sum(1 for history in self.histories.values() if history.curr is not None)
        return {
            "foca_enabled": True,
            "foca_warmup_steps": int(self.config.warmup_steps),
            "foca_cache_interval": int(self.config.cache_interval),
            "foca_start_layer": int(self.start_layer),
            "foca_end_layer": int(self.end_layer),
            "foca_blend": float(self.config.blend),
            "foca_cache_device": self.config.cache_device,
            "foca_steps_seen": int(self.step_index + 1),
            "foca_full_transformer_calls": int(self.full_transformer_calls),
            "foca_predicted_transformer_calls": int(self.predicted_transformer_calls),
            "foca_full_block_calls": int(self.full_block_calls),
            "foca_predicted_block_calls": int(self.predicted_block_calls),
            "foca_fallback_block_calls": int(self.fallback_block_calls),
            "foca_cached_histories": int(cached),
        }


def apply_foca_cache(transformer, config: FoCaCacheConfig) -> FoCaRuntime:
    runtime = FoCaRuntime(transformer, config)
    transformer._foca_runtime = runtime

    original_transformer_forward = transformer.forward

    def wrapped_transformer_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image=None,
        return_dict: bool = True,
        attention_kwargs=None,
    ):
        runtime.begin_forward(timestep)
        return original_transformer_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )

    transformer.forward = types.MethodType(wrapped_transformer_forward, transformer)

    for layer_idx, block in enumerate(transformer.blocks):
        original_block_forward = block.forward

        def make_block_forward(idx, original):
            def wrapped_block_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
                return runtime.block_forward(
                    idx,
                    original,
                    self,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    rotary_emb,
                )

            return wrapped_block_forward

        block.forward = types.MethodType(make_block_forward(layer_idx, original_block_forward), block)

    logger.info(
        "[FoCa] Enabled block feature caching: "
        f"warmup={config.warmup_steps}, interval={config.cache_interval}, "
        f"layers={runtime.start_layer}-{runtime.end_layer}, blend={config.blend}, cache_device={config.cache_device}"
    )
    return runtime


def collect_foca_metadata(transformer) -> Dict[str, object]:
    runtime = getattr(transformer, "_foca_runtime", None)
    if runtime is None:
        return {"foca_enabled": False}
    return runtime.metadata()


def clear_foca_cache(transformer) -> None:
    runtime = getattr(transformer, "_foca_runtime", None)
    if runtime is None:
        return
    runtime.histories.clear()
