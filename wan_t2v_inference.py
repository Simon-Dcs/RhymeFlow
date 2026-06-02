import argparse
import gc
import json
import os
import math
import time
from copy import deepcopy

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from termcolor import colored

from dataloader import load_prompt_or_image
from svg.models.wan.foca import FoCaCacheConfig, apply_foca_cache, clear_foca_cache, collect_foca_metadata
from svg.models.wan.inference import replace_wan_attention
from svg.models.wan.rhymeflow import RhymeFlowConfig, rhymeflow_wan_generate
from svg.utils.seed import seed_everything
from svg.timer import print_operator_log_data

from svg.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="Model ID to use for generation")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Wan_VBench", "T2V_Xingyang_VBench"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")

    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--logging_file", type=str, default=None, help="Path to the logging file.")
    parser.add_argument("--summary_file", type=str, default=None, help="Path to write run summary JSON.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generating existing output files")

    parser.add_argument("--pattern", type=str, default="dense", choices=["SVG", "dense", "SAP", "SSS", "RHYME", "RHYME_SVG", "RHYME_SAP", "FOCA"])
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="Only works for best config. Leave the first 10% timestep in FP")
    
    # SVG related
    parser.add_argument("--num_sampled_rows", type=int, default=64, help="The number of sampled rows")
    parser.add_argument("--sample_mse_max_row", type=int, default=10000, help="The maximum number of rows in attention mask. Prevent OOM.")
    parser.add_argument("--sparsity", type=float, default=0.25, help="The sparsity of the striped attention pattern. Accepts one or two float values.")

    # SAP related
    parser.add_argument("--num_q_centroids", "--qc", type=int, default=50, help="Number of query centroids for KMEANS_BLOCK.")
    parser.add_argument("--num_k_centroids", "--kc", type=int, default=200, help="Number of key centroids for KMEANS_BLOCK.")
    parser.add_argument("--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in KMEANS_BLOCK.")
    parser.add_argument("--min_kc_ratio", type=float, default=0, help="At least this proportion of key blocks to keep per query block in KMEANS_BLOCK.")
    parser.add_argument("--kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for initialization in KMEANS_BLOCK.")
    parser.add_argument("--kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for other diffusion steps in KMEANS_BLOCK.")

    # SSS related
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps with dense attention for SSS.")
    parser.add_argument("--num_keyframes", type=int, default=12, help="Number of keyframes to identify for SSS.")
    parser.add_argument("--skip_n", type=int, default=2, help="Normal frames denoise every skip_n steps in SSS.")
    parser.add_argument("--similarity_window", type=int, default=5, help="Window size for computing frame similarity in SSS.")
    parser.add_argument("--keyframe_strategy", type=str, default="cosine", choices=["cosine", "cosine_original", "fixed", "uniform", "adaptive", "sequential", "semantic", "random", "first"], help="Keyframe identification strategy for SSS.")
    parser.add_argument("--keyframe_similarity_threshold", type=float, default=0.98, help="Cosine threshold for sequential/semantic SSS keyframe selection.")
    parser.add_argument("--min_keyframe_gap", type=int, default=1, help="Minimum latent-frame gap for sequential/semantic SSS keyframes.")
    parser.add_argument("--sss_schedule", type=str, default="progressive", choices=["progressive", "fixed"], help="SSS skip schedule mode.")
    parser.add_argument("--sss_min_skip", type=int, default=2, help="Minimum skip interval for progressive SSS.")
    parser.add_argument("--sss_max_skip", type=int, default=3, help="Maximum skip interval for progressive SSS.")
    parser.add_argument("--sss_transition_points", type=str, default=None, help="Comma-separated SSS transition offsets or ratios, e.g. '0.3,0.7' or '12,28'.")
    parser.add_argument("--rhyme_projection_space", type=str, default="sigma", choices=["sigma", "timestep"], help="Interpolation coordinate for RhymeFlow latent projection.")
    parser.add_argument("--rhyme_projection_mode", type=str, default="linear", choices=["linear", "foca"], help="Skipped-state estimation mode for RhymeFlow.")
    parser.add_argument("--rhyme_foca_blend", type=float, default=0.5, help="Blend weight for the FoCa predictor in skipped-state calibration.")
    parser.add_argument("--rhyme_foca_min_skip", type=int, default=4, help="Minimum group skip span before FoCa calibration activates.")
    parser.add_argument("--rhyme_context_mode", type=str, default="latent", choices=["latent", "last_full", "last_full_nonkey", "last_full_nonkey_cpu"], help="Context source for RhymeFlow async keyframe updates.")
    parser.add_argument("--rhyme_async_context_mode", type=str, default="full", choices=["full", "keyframe_only"], help="Whether async keyframe updates attend to projected full context or keyframes only.")
    parser.add_argument("--rhyme_no_keyframes", action="store_true", help="Disable keyframe updates and use group-level skip only.")
    parser.add_argument("--rhyme_solver", type=str, default="euler", choices=["euler", "scheduler", "scheduler_approx"], help="RhymeFlow latent update rule. 'scheduler' is intended for no-skip/full-frame validation; 'scheduler_approx' advances scheduler state during skipped groups with projected non-keyframe velocities.")
    parser.add_argument("--foca_warmup_steps", type=int, default=2, help="Full transformer steps before standalone FoCa block-cache prediction starts.")
    parser.add_argument("--foca_cache_interval", type=int, default=3, help="Standalone FoCa full-compute interval after warmup. Interval 3 means one full step followed by two cached steps.")
    parser.add_argument("--foca_start_layer", type=int, default=0, help="First transformer block to cache for standalone FoCa.")
    parser.add_argument("--foca_end_layer", type=int, default=-1, help="Last transformer block to cache for standalone FoCa. -1 means the final block.")
    parser.add_argument("--foca_blend", type=float, default=0.75, help="Damping weight for standalone FoCa Heun-style calibration.")
    parser.add_argument("--foca_cache_device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device used to keep standalone FoCa feature history.")
    parser.add_argument("--decode_after_cache_clear", action="store_true", help="Return latents from the pipeline, clear SSS caches, then decode manually.")
    parser.add_argument("--offload_transformer_before_decode", action="store_true", help="Move transformer to CPU before manual latent decode.")
    parser.add_argument("--vae_tiling", action="store_true", help="Enable Wan VAE spatial tiled decode.")
    parser.add_argument("--vae_slicing", action="store_true", help="Enable Wan VAE batch sliced decode.")
    parser.add_argument("--vae_stream_cpu", action="store_true", help="Decode Wan VAE frames sequentially and accumulate decoded frames on CPU.")
    parser.add_argument("--latent_only", action="store_true", help="Stop after denoising latents and skip VAE decode/video export.")
    parser.add_argument("--allow_decode_failure", action="store_true", help="Write summary and exit successfully if VAE decode OOMs.")

    args = parser.parse_args()

    def parse_transition_points(value):
        if value is None or value == "":
            return None
        points = [float(item.strip()) for item in value.split(",") if item.strip()]
        if len(points) != 2:
            raise ValueError("--sss_transition_points must contain exactly two comma-separated values.")
        return points

    def clear_sss_caches(pipe):
        if args.pattern != "SSS":
            return
        logger.info("Clearing SSS caches...")
        for block in pipe.transformer.blocks:
            if hasattr(block.attn1, 'processor'):
                processor = block.attn1.processor
                if hasattr(processor, 'normal_frame_cache'):
                    processor.normal_frame_cache.clear()
        from svg.models.wan.attention import WanAttn_SSSAttn_Processor
        WanAttn_SSSAttn_Processor.keyframe_indices = None
        WanAttn_SSSAttn_Processor.step_counter = 0
        WanAttn_SSSAttn_Processor.last_timestep = None
        WanAttn_SSSAttn_Processor.current_branch = 0
        WanAttn_SSSAttn_Processor.branch_counter = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def log_cuda_memory(label):
        if not torch.cuda.is_available():
            return
        logger.info(
            f"{label}: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB, "
            f"peak_allocated={torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )

    def decode_wan_latents(pipe, latents):
        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        with torch.no_grad():
            if args.vae_stream_cpu and not getattr(pipe.vae, "use_tiling", False):
                logger.info("Using streaming Wan VAE decode with CPU accumulation.")
                pipe.vae.clear_cache()
                _, _, num_frame, _, _ = latents.shape
                x = pipe.vae.post_quant_conv(latents)
                decoded_chunks = []
                for i in range(num_frame):
                    pipe.vae._conv_idx = [0]
                    decoded = pipe.vae.decoder(
                        x[:, :, i : i + 1, :, :],
                        feat_cache=pipe.vae._feat_map,
                        feat_idx=pipe.vae._conv_idx,
                    )
                    decoded = torch.clamp(decoded, min=-1.0, max=1.0)
                    decoded_chunks.append(decoded.detach().cpu())
                    del decoded
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                pipe.vae.clear_cache()
                del x
                video = torch.cat(decoded_chunks, dim=2)
            else:
                video = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.video_processor.postprocess_video(video, output_type="np")

    seed_everything(args.seed)

    # In some cases it will raise RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
    torch.backends.cuda.preferred_linalg_library(backend="magma")
    
    if args.skip_existing:
        if os.path.exists(args.output_file):
            logger.info(f"Output file {args.output_file} already exists. Skipping generation.")
            exit(0)

    #########################################################
    # Load the model
    #########################################################
    # Default release target: Wan-AI/Wan2.1-T2V-1.3B-Diffusers.
    model_id = args.model_id
    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    pipe.scheduler = scheduler
    pipe.to("cuda")

    if args.vae_tiling:
        pipe.vae.enable_tiling()
        logger.info("Wan VAE tiling enabled.")
    if args.vae_slicing:
        pipe.vae.enable_slicing()
        logger.info("Wan VAE slicing enabled.")
        
    config = pipe.transformer.config
    
    #########################################################
    # Translate the percentage of warmup of layers and timesteps to the actual layers and timesteps
    #########################################################
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)
    ref_timesteps = ref_scheduler.timesteps
    
    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * config.num_layers)
    if num_fp_timesteps > 0:
        args.first_times_fp = ref_scheduler.timesteps[num_fp_timesteps - 1] - 1
    else:
        args.first_times_fp = 1001 # 1000 is the first timestep
    args.first_layers_fp = num_fp_layers
    
    logger.info(f"Warmup of Timesteps: {num_fp_timesteps} / {args.num_inference_steps} || {args.first_times_fp} / 1000 use FP")
    logger.info(f"Warmup of Layers: {num_fp_layers} / {config.num_layers} use FP")
    
    #########################################################
    # Load the prompt
    #########################################################
    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)

    if args.prompt is None:
        logger.info(colored("Using default prompt", "red"))
        args.prompt = "A cat walks on the grass, realistic"

    if args.negative_prompt is None:
        args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    print("=" * 20 + " Prompts " + "=" * 20)
    print(f"Prompt: {args.prompt}\n\n" + f"Negative Prompt: {args.negative_prompt}")

    #########################################################
    # Replace the attention & time logger
    #########################################################
    if args.pattern == "SVG":
        replace_wan_attention(
            pipe, 
            args.height, 
            args.width, 
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SVG specific
            num_sampled_rows=args.num_sampled_rows,
            sample_mse_max_row=args.sample_mse_max_row,
            sparsity=args.sparsity,
        )
    elif args.pattern == "RHYME_SVG":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern="SVG",
            # SVG specific
            num_sampled_rows=args.num_sampled_rows,
            sample_mse_max_row=args.sample_mse_max_row,
            sparsity=args.sparsity,
        )
    elif args.pattern == "RHYME_SAP":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern="SAP",
            # SAP specific
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            kmeans_iter_init=args.kmeans_iter_init,
            kmeans_iter_step=args.kmeans_iter_step,
        )
    elif args.pattern == "SAP":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SAP specific
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            kmeans_iter_init=args.kmeans_iter_init,
            kmeans_iter_step=args.kmeans_iter_step,
        )
    elif args.pattern == "SSS":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SSS specific
            warmup_steps=args.warmup_steps,
            num_keyframes=args.num_keyframes,
            skip_n=args.skip_n,
            similarity_window=args.similarity_window,
            keyframe_strategy=args.keyframe_strategy,
            keyframe_similarity_threshold=args.keyframe_similarity_threshold,
            min_keyframe_gap=args.min_keyframe_gap,
            sss_schedule=args.sss_schedule,
            sss_min_skip=args.sss_min_skip,
            sss_max_skip=args.sss_max_skip,
            sss_transition_points=parse_transition_points(args.sss_transition_points),
            total_inference_steps=args.num_inference_steps,
            logging_file=args.logging_file,
            sparsity=args.sparsity,
        )
    elif args.pattern == "FOCA":
        apply_foca_cache(
            pipe.transformer,
            FoCaCacheConfig(
                warmup_steps=args.foca_warmup_steps,
                cache_interval=args.foca_cache_interval,
                start_layer=args.foca_start_layer,
                end_layer=None if args.foca_end_layer < 0 else args.foca_end_layer,
                blend=args.foca_blend,
                cache_device=args.foca_cache_device,
            ),
        )

    # Print time logger
    for block in pipe.transformer.blocks:
        block.register_forward_hook(print_operator_log_data)

    #########################################################
    # Generate the video
    #########################################################
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    status = "success"
    error_message = None
    output = None
    denoise_time = None
    decode_time = None
    rhymeflow_metadata = {}
    foca_metadata = {}

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps,
    )

    if args.pattern in ("RHYME", "RHYME_SVG", "RHYME_SAP"):
        rhyme_config = RhymeFlowConfig(
            warmup_steps=args.warmup_steps,
            num_keyframes=args.num_keyframes,
            keyframe_strategy=args.keyframe_strategy,
            keyframe_similarity_threshold=args.keyframe_similarity_threshold,
            min_keyframe_gap=args.min_keyframe_gap,
            sss_schedule=args.sss_schedule,
            sss_min_skip=args.sss_min_skip,
            sss_max_skip=args.sss_max_skip,
            sss_transition_points=parse_transition_points(args.sss_transition_points),
            projection_space=args.rhyme_projection_space,
            projection_mode=args.rhyme_projection_mode,
            foca_blend=args.rhyme_foca_blend,
            foca_min_skip=args.rhyme_foca_min_skip,
            context_mode=args.rhyme_context_mode,
            async_context_mode=args.rhyme_async_context_mode,
            no_keyframes=args.rhyme_no_keyframes,
            solver=args.rhyme_solver,
            logging_file=args.logging_file,
        )
        latent_result, rhymeflow_metadata = rhymeflow_wan_generate(
            pipe,
            **pipe_kwargs,
            output_type="latent",
            config=rhyme_config,
        )
        latent_output = latent_result.frames
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        denoise_time = time.perf_counter() - start_time
        log_cuda_memory("After RhymeFlow denoise")
        if args.latent_only:
            status = "latent_only"
        elif args.offload_transformer_before_decode:
            logger.info("Moving transformer to CPU before VAE decode...")
            pipe.transformer.to("cpu")
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                logger.info("Moving text encoder to CPU before VAE decode...")
                pipe.text_encoder.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            log_cuda_memory("After model offload")

        if not args.latent_only:
            decode_start = time.perf_counter()
            try:
                output = decode_wan_latents(pipe, latent_output)[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_time = time.perf_counter() - decode_start
            except torch.cuda.OutOfMemoryError as exc:
                decode_time = time.perf_counter() - decode_start
                status = "decode_oom"
                error_message = str(exc)
                logger.exception("VAE decode failed with CUDA OOM.")
                if not args.allow_decode_failure:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    elif args.decode_after_cache_clear or args.latent_only:
        latent_output = pipe(**pipe_kwargs, output_type="latent").frames
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        denoise_time = time.perf_counter() - start_time
        log_cuda_memory("After denoise")
        if args.pattern == "FOCA":
            foca_metadata = collect_foca_metadata(pipe.transformer)
            clear_foca_cache(pipe.transformer)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            log_cuda_memory("After clearing FoCa feature history")
        clear_sss_caches(pipe)
        log_cuda_memory("After clearing caches")
        if args.latent_only:
            status = "latent_only"
        elif args.offload_transformer_before_decode:
            logger.info("Moving transformer to CPU before VAE decode...")
            pipe.transformer.to("cpu")
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                logger.info("Moving text encoder to CPU before VAE decode...")
                pipe.text_encoder.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            log_cuda_memory("After model offload")

        if not args.latent_only:
            decode_start = time.perf_counter()
            try:
                output = decode_wan_latents(pipe, latent_output)[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_time = time.perf_counter() - decode_start
            except torch.cuda.OutOfMemoryError as exc:
                decode_time = time.perf_counter() - decode_start
                status = "decode_oom"
                error_message = str(exc)
                logger.exception("VAE decode failed with CUDA OOM.")
                if not args.allow_decode_failure:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    else:
        output = pipe(**pipe_kwargs).frames[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    if not foca_metadata:
        foca_metadata = collect_foca_metadata(pipe.transformer)

    #########################################################
    # TODO-3: Clear SSS caches before VAE decode to prevent OOM
    #########################################################
    if args.pattern == "SSS":
        clear_sss_caches(pipe)

    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if output is not None:
        export_to_video(output, args.output_file, fps=16)
    else:
        logger.warning(f"No decoded video exported because run status is {status}.")

    if args.summary_file is not None:
        def json_safe(value):
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.item()
                return value.detach().cpu().tolist()
            return value

        summary_dir = os.path.dirname(args.summary_file)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        summary = {
            "pattern": args.pattern,
            "model_id": args.model_id,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "prompt_source": args.prompt_source,
            "prompt_idx": args.prompt_idx,
            "output_file": args.output_file,
            "status": status,
            "error": error_message,
            "total_time_sec": total_time,
            "denoise_time_sec": denoise_time,
            "decode_time_sec": decode_time,
            "offload_transformer_before_decode": args.offload_transformer_before_decode,
            "vae_tiling": args.vae_tiling,
            "vae_slicing": args.vae_slicing,
            "vae_stream_cpu": args.vae_stream_cpu,
            "latent_only": args.latent_only,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None,
            "first_times_fp": json_safe(args.first_times_fp),
            "first_layers_fp": json_safe(args.first_layers_fp),
            "sparsity": args.sparsity,
            "num_q_centroids": args.num_q_centroids,
            "num_k_centroids": args.num_k_centroids,
            "top_p_kmeans": args.top_p_kmeans,
            "min_kc_ratio": args.min_kc_ratio,
            "kmeans_iter_init": args.kmeans_iter_init,
            "kmeans_iter_step": args.kmeans_iter_step,
            "warmup_steps": args.warmup_steps,
            "num_keyframes": args.num_keyframes,
            "skip_n": args.skip_n,
            "keyframe_strategy": args.keyframe_strategy,
            "keyframe_similarity_threshold": args.keyframe_similarity_threshold,
            "min_keyframe_gap": args.min_keyframe_gap,
            "sss_schedule": args.sss_schedule,
            "sss_min_skip": args.sss_min_skip,
            "sss_max_skip": args.sss_max_skip,
            "sss_transition_points": args.sss_transition_points,
            "sss_use_svg_sparse": os.environ.get("SSS_USE_SVG_SPARSE", "false"),
            "rhyme_projection_space": args.rhyme_projection_space,
            "rhyme_projection_mode": args.rhyme_projection_mode,
            "rhyme_foca_blend": args.rhyme_foca_blend,
            "rhyme_foca_min_skip": args.rhyme_foca_min_skip,
            "rhyme_context_mode": args.rhyme_context_mode,
            "rhyme_async_context_mode": args.rhyme_async_context_mode,
            "rhyme_no_keyframes": args.rhyme_no_keyframes,
            "rhyme_solver": args.rhyme_solver,
            "foca_warmup_steps": args.foca_warmup_steps,
            "foca_cache_interval": args.foca_cache_interval,
            "foca_start_layer": args.foca_start_layer,
            "foca_end_layer": args.foca_end_layer,
            "foca_blend": args.foca_blend,
            "foca_cache_device": args.foca_cache_device,
        }
        summary.update(rhymeflow_metadata)
        summary.update(foca_metadata)
        with open(args.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Run summary written to {args.summary_file}")
