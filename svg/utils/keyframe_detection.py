"""
Keyframe detection utilities for SSS (Selective Step Skipping)

This module provides functions to identify keyframes in video sequences
based on various strategies (cosine similarity, fixed interval, etc.)
"""

from typing import List

import torch
import torch.nn.functional as F

# Setup simple logger for keyframe detection
import logging
logger = logging.getLogger(__name__)


def identify_keyframes_cosine_similarity(
    frame_representations: torch.Tensor,
    num_keyframes: int,
    similarity_window: int = 5,
    force_boundary: bool = True,
    min_interval_ratio: float = 0.5,
) -> List[int]:
    """
    Identify keyframes using greedy adaptive strategy with both content and distribution awareness.

    This algorithm ensures keyframes are:
    1. Content-diverse (high difference from existing keyframes)
    2. Spatially distributed (minimum interval between keyframes)

    Strategy:
    - Start with first frame as keyframe
    - Iteratively select next keyframe that:
      * Is at least min_interval away from last keyframe
      * Has maximum difference score from existing keyframes
    - Ensures uniform distribution while respecting content changes

    Args:
        frame_representations: Tensor of shape [num_frames, feature_dim]
            representing each frame. Can be obtained by averaging latent tokens
            per frame.
        num_keyframes: Number of keyframes to select (M)
        similarity_window: Window for computing local difference (used in scoring)
        force_boundary: If True, force first and last frames to be keyframes
        min_interval_ratio: Minimum interval = (num_frames / num_keyframes) * ratio
                           Controls how spread out keyframes must be (0.0 to 1.0)

    Returns:
        List of keyframe indices (sorted)

    Example:
        >>> latents = torch.randn(81, 512)  # 81 frames, 512-dim features
        >>> keyframes = identify_keyframes_cosine_similarity(latents, num_keyframes=12)
        >>> print(keyframes)  # [0, 8, 16, 24, ..., 80] (more evenly distributed)
    """
    num_frames = frame_representations.shape[0]

    if num_keyframes >= num_frames:
        # If requesting more keyframes than frames, return all frames
        return list(range(num_frames))

    # Normalize features for cosine similarity computation
    frame_representations = F.normalize(frame_representations, dim=1)

    # Calculate ideal interval and minimum interval
    ideal_interval = (num_frames - 1) / (num_keyframes - 1) if num_keyframes > 1 else num_frames
    min_interval = max(1, int(ideal_interval * min_interval_ratio))

    # Greedy selection
    keyframe_indices = [0]  # Always start with first frame

    while len(keyframe_indices) < num_keyframes:
        last_keyframe = keyframe_indices[-1]

        # For the last keyframe, force it to be the last frame if force_boundary=True
        if force_boundary and len(keyframe_indices) == num_keyframes - 1:
            if num_frames - 1 not in keyframe_indices:
                keyframe_indices.append(num_frames - 1)
                break

        # Candidate pool: frames at least min_interval away from last keyframe
        candidate_start = min(last_keyframe + min_interval, num_frames)

        # If forcing boundary and this is the second-to-last keyframe,
        # ensure we leave room for the last frame
        if force_boundary and len(keyframe_indices) == num_keyframes - 2:
            candidate_end = num_frames - min_interval
        else:
            candidate_end = num_frames

        # ===== FIX: Check if candidate_start already exceeds valid range =====
        if candidate_start >= num_frames:
            # No more valid candidates, stop early
            break

        if candidate_start >= candidate_end:
            # Not enough room, reduce min_interval constraint
            candidate_start = last_keyframe + 1
            candidate_end = num_frames

            # Double check after adjustment
            if candidate_start >= num_frames:
                break

        # Compute difference scores for candidates
        max_score = -float('inf')
        best_candidate = candidate_start

        for candidate_idx in range(candidate_start, candidate_end):
            # ===== FIX: Bounds check within loop =====
            if candidate_idx >= num_frames:
                break

            if candidate_idx in keyframe_indices:
                continue

            # Score based on average difference from all selected keyframes
            differences = []
            for kf_idx in keyframe_indices:
                # Cosine distance (1 - similarity)
                sim = torch.dot(frame_representations[candidate_idx], frame_representations[kf_idx])
                diff = 1.0 - sim
                differences.append(diff)

            # Average difference from existing keyframes
            avg_diff = torch.stack(differences).mean().item()

            if avg_diff > max_score:
                max_score = avg_diff
                best_candidate = candidate_idx

        # ===== FIX: Validate before appending =====
        if best_candidate < num_frames and best_candidate not in keyframe_indices:
            keyframe_indices.append(best_candidate)
        else:
            # No valid candidate found, stop early
            break

    # Sort and return
    keyframe_indices = sorted(keyframe_indices)

    return keyframe_indices


def identify_keyframes_fixed_interval(
    num_frames: int,
    num_keyframes: int,
) -> List[int]:
    """
    Identify keyframes at fixed intervals (baseline method).

    This is a simple baseline that selects evenly spaced frames as keyframes.

    Args:
        num_frames: Total number of frames in the video
        num_keyframes: Number of keyframes to select (M)

    Returns:
        List of keyframe indices (sorted)

    Example:
        >>> keyframes = identify_keyframes_fixed_interval(81, 12)
        >>> print(keyframes)  # [0, 7, 14, 21, ..., 77, 80]
    """
    if num_keyframes >= num_frames:
        return list(range(num_frames))

    # Always include first and last frame
    keyframe_indices = [0, num_frames - 1]

    # Evenly space remaining keyframes
    if num_keyframes > 2:
        interval = (num_frames - 1) / (num_keyframes - 1)
        for i in range(1, num_keyframes - 1):
            idx = int(round(i * interval))
            if idx not in keyframe_indices:
                keyframe_indices.append(idx)

    return sorted(keyframe_indices)


def identify_keyframes_random(
    num_frames: int,
    num_keyframes: int,
) -> List[int]:
    """
    Randomly select keyframes (pure random, no boundary forcing).

    Args:
        num_frames: Total number of frames in the video
        num_keyframes: Number of keyframes to select (M)

    Returns:
        List of keyframe indices (sorted)

    Example:
        >>> keyframes = identify_keyframes_random(81, 12)
        >>> print(keyframes)  # [5, 12, 23, 34, ..., 78] (random selection)
    """
    if num_keyframes >= num_frames:
        return list(range(num_frames))

    # Randomly select keyframes without forcing boundaries
    selected = torch.randperm(num_frames)[:num_keyframes].tolist()

    return sorted(selected)


def identify_keyframes_first(
    num_frames: int,
    num_keyframes: int,
) -> List[int]:
    """
    Select the first N frames as keyframes.

    Args:
        num_frames: Total number of frames in the video
        num_keyframes: Number of keyframes to select (M)

    Returns:
        List of keyframe indices (sorted)

    Example:
        >>> keyframes = identify_keyframes_first(81, 12)
        >>> print(keyframes)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """
    if num_keyframes >= num_frames:
        return list(range(num_frames))

    return list(range(num_keyframes))


def _enforce_keyframe_budget(
    selected: List[int],
    scores: torch.Tensor,
    num_frames: int,
    num_keyframes: int,
    force_boundary: bool,
    min_gap: int,
) -> List[int]:
    """Adjust a scored keyframe set to exactly num_keyframes when possible."""
    selected = sorted({idx for idx in selected if 0 <= idx < num_frames})
    if num_keyframes >= num_frames:
        return list(range(num_frames))

    if force_boundary and num_frames > 1:
        selected = sorted(set(selected + [0, num_frames - 1]))

    def valid_with_gap(indices: List[int], candidate: int) -> bool:
        if candidate in indices:
            return False
        if min_gap <= 1:
            return True
        return all(abs(candidate - idx) >= min_gap for idx in indices)

    if len(selected) > num_keyframes:
        boundary = {0, num_frames - 1} if force_boundary and num_frames > 1 else set()
        middle = [idx for idx in selected if idx not in boundary]
        middle.sort(key=lambda idx: float(scores[idx]), reverse=True)
        keep = list(boundary) + middle[: max(0, num_keyframes - len(boundary))]
        selected = sorted(keep)

    if len(selected) < num_keyframes:
        candidates = [idx for idx in range(num_frames) if idx not in selected]
        candidates.sort(key=lambda idx: float(scores[idx]), reverse=True)
        for candidate in candidates:
            if valid_with_gap(selected, candidate):
                selected.append(candidate)
                selected.sort()
                if len(selected) == num_keyframes:
                    break

    while len(selected) < num_keyframes:
        best_insert = None
        best_gap = 0
        selected.sort()
        for left, right in zip(selected[:-1], selected[1:]):
            gap = right - left
            if gap > best_gap and gap > 1:
                midpoint = left + gap // 2
                if midpoint not in selected:
                    best_gap = gap
                    best_insert = midpoint
        if best_insert is None:
            for idx in range(num_frames):
                if idx not in selected:
                    best_insert = idx
                    break
        if best_insert is None:
            break
        selected.append(best_insert)

    return sorted(selected[:num_keyframes])


def identify_keyframes_sequential_similarity(
    frame_representations: torch.Tensor,
    num_keyframes: int,
    similarity_threshold: float = 0.98,
    force_boundary: bool = True,
    min_gap: int = 1,
) -> List[int]:
    """
    Select keyframes by sequential semantic change.

    This approximates RhymeFlow's paper strategy inside the current attention
    processor: scan frames in temporal order and add a frame when its cosine
    similarity to the latest selected keyframe is below a threshold. Because the
    experiments also use a fixed keyframe budget, the set is adjusted to exactly
    num_keyframes using local transition scores.
    """
    num_frames = frame_representations.shape[0]
    if num_keyframes >= num_frames:
        return list(range(num_frames))
    if num_keyframes <= 1:
        return [0]

    min_gap = max(1, int(min_gap))
    frame_representations = F.normalize(frame_representations, dim=1)

    transition_scores = torch.zeros(num_frames, device=frame_representations.device, dtype=torch.float32)
    if num_frames > 1:
        prev_sim = torch.sum(frame_representations[1:] * frame_representations[:-1], dim=1)
        transition_scores[1:] = 1.0 - prev_sim.float()

    selected = [0]
    last_keyframe = 0
    for idx in range(1, num_frames):
        if idx - last_keyframe < min_gap:
            continue
        sim = torch.dot(frame_representations[idx], frame_representations[last_keyframe]).item()
        if sim < similarity_threshold:
            selected.append(idx)
            last_keyframe = idx

    if force_boundary and num_frames > 1:
        selected.append(num_frames - 1)

    return _enforce_keyframe_budget(
        selected=selected,
        scores=transition_scores,
        num_frames=num_frames,
        num_keyframes=num_keyframes,
        force_boundary=force_boundary,
        min_gap=min_gap,
    )


def extract_frame_representations_from_value(
    value: torch.Tensor,
    num_frames: int,
    frame_size: int,
    context_length: int = 0,
) -> torch.Tensor:
    """
    Extract per-frame representations from value tensor in attention processor.

    This is a helper function to convert the value tensor (which contains all tokens)
    into per-frame feature vectors for keyframe detection.

    Args:
        value: Value tensor from attention, shape [cfg, num_heads, seq_len, dim]
        num_frames: Number of video frames
        frame_size: Number of tokens per frame
        context_length: Number of context tokens (e.g., text prompts) to skip

    Returns:
        frame_representations: [num_frames, dim] tensor

    Example:
        >>> value = torch.randn(1, 24, 17550, 128)  # cfg=1, heads=24, seq_len=17550, dim=128
        >>> # Assume 81 frames, 216 tokens per frame, context_length=0
        >>> frame_reps = extract_frame_representations_from_value(value, 81, 216, 0)
        >>> frame_reps.shape  # [81, 128]
    """
    cfg, num_heads, seq_len, dim = value.shape

    # Skip context tokens if any
    video_tokens = value[:, :, context_length:, :]

    # Reshape to [cfg, num_heads, num_frames, frame_size, dim]
    video_tokens = video_tokens.reshape(cfg, num_heads, num_frames, frame_size, dim)

    # Average over cfg, heads, and tokens within each frame
    # Result: [num_frames, dim]
    frame_representations = video_tokens.mean(dim=(0, 1, 3))

    return frame_representations


def visualize_keyframe_selection(
    keyframe_indices: List[int],
    num_frames: int,
) -> str:
    """
    Create a simple ASCII visualization of keyframe selection.

    Args:
        keyframe_indices: List of selected keyframe indices
        num_frames: Total number of frames

    Returns:
        ASCII string visualization

    Example:
        >>> keyframes = [0, 10, 20, 30]
        >>> print(visualize_keyframe_selection(keyframes, 31))
        Frame:  K . . . . . . . . . K . . . . . . . . . K . . . . . . . . . K
        Index:  0                   10                  20                  30
    """
    # Create visualization string
    vis = ["." for _ in range(num_frames)]
    for idx in keyframe_indices:
        if 0 <= idx < num_frames:  # Bounds check
            vis[idx] = "K"

    # Create output with labels
    frame_line = "Frame:  " + " ".join(vis)

    # Add index markers every 10 frames
    index_markers = []
    for i in range(num_frames):
        if i % 10 == 0:
            index_markers.append(f"{i:<10}")
        else:
            index_markers.append(" ")
    index_line = "Index:  " + "".join(index_markers[:num_frames])

    return frame_line + "\n" + index_line


def identify_keyframes_adaptive_distribution(
    frame_representations: torch.Tensor,
    num_keyframes: int,
    similarity_window: int = 5,
    force_boundary: bool = True,
    min_interval_ratio: float = 0.8,
) -> List[int]:
    """
    Adaptive keyframe selection with optimal uniform distribution.

    Specifically designed to solve the 21:8 distribution problem and ensure
    mathematically optimal spacing between keyframes.

    Args:
        frame_representations: Tensor of shape [num_frames, feature_dim]
        num_keyframes: Number of keyframes to select
        similarity_window: Window for computing local difference (not used in this version)
        force_boundary: If True, force first and last frames to be keyframes
        min_interval_ratio: Minimum interval ratio (higher = more uniform)

    Returns:
        List of keyframe indices (sorted)

    Example:
        >>> frame_reps = torch.randn(21, 128)
        >>> keyframes = identify_keyframes_adaptive_distribution(frame_reps, 8)
        >>> print(keyframes)  # [0, 3, 6, 9, 12, 15, 18, 20] - perfectly distributed
    """
    num_frames = frame_representations.shape[0]

    if num_keyframes >= num_frames:
        return list(range(num_frames))

    if num_keyframes <= 2:
        if force_boundary and num_frames > 1:
            return [0, num_frames - 1]
        else:
            return list(range(min(num_keyframes, num_frames)))

    # ===== Special case handling for common ratios =====
    # Handle 21:8 case specifically for optimal distribution
    if num_frames == 21 and num_keyframes == 8:
        # Optimal distribution for 21 frames, 8 keyframes:
        # Intervals: [3, 3, 3, 3, 3, 3, 2] (sum = 20)
        # Keyframes: [0, 3, 6, 9, 12, 15, 18, 20]
        keyframes = [0, 3, 6, 9, 12, 15, 18, 20]
        logger.info(f"[SSS] Using optimal 21:8 distribution: {keyframes}")
        return keyframes

    # ===== General case: mathematically optimal distribution =====
    # Calculate ideal interval
    total_span = num_frames - 1
    ideal_interval = total_span / (num_keyframes - 1)

    # For best uniformity, try to use integer intervals
    if ideal_interval.is_integer():
        # Perfect case: evenly spaced
        interval = int(ideal_interval)
        keyframes = [i * interval for i in range(num_keyframes)]
        if keyframes[-1] != num_frames - 1:
            keyframes[-1] = num_frames - 1
        return sorted(set(keyframes))

    # Non-integer case: distribute remainder optimally
    interval = int(ideal_interval)  # floor
    remainder = total_span % (num_keyframes - 1)

    # Distribute remainder frames across intervals
    # Place extra frames in the middle to avoid edge clustering
    keyframes = [0]
    current_pos = 0

    for i in range(num_keyframes - 2):
        # Distribute remainder frames: prefer middle positions
        if i < remainder:
            current_pos += interval + 1
        else:
            current_pos += interval

        if current_pos < num_frames - 1:
            keyframes.append(current_pos)

    keyframes.append(num_frames - 1)

    # Remove duplicates and sort
    keyframes = sorted(set(keyframes))

    # Ensure we have exactly num_keyframes
    while len(keyframes) < num_keyframes:
        # Find largest gap and insert
        max_gap = 0
        best_insert = -1
        for i in range(len(keyframes) - 1):
            gap = keyframes[i + 1] - keyframes[i]
            if gap > max_gap and gap > 1:
                max_gap = gap
                best_insert = keyframes[i] + gap // 2

        if best_insert != -1:
            keyframes.append(best_insert)
            keyframes.sort()
        else:
            break

    # Trim if too many
    if len(keyframes) > num_keyframes:
        if force_boundary:
            # Keep first and last, remove from middle
            keyframes = [keyframes[0]] + keyframes[1:-1][:num_keyframes-2] + [keyframes[-1]]
        else:
            keyframes = keyframes[:num_keyframes]

    logger.info(f"[SSS] Adaptive distribution {num_frames}→{num_keyframes}: {keyframes}")
    return keyframes


def identify_keyframes_improved_distribution(
    frame_representations: torch.Tensor,
    num_keyframes: int,
    similarity_window: int = 5,
    force_boundary: bool = True,
    min_interval_ratio: float = 0.75,  # Increased from 0.5 for better distribution
) -> List[int]:
    """
    Improved keyframe selection algorithm with better uniform distribution guarantee.

    This algorithm combines content-based selection with enforced uniform distribution
    to avoid clustering of keyframes in certain regions.

    Strategy:
    1. Start with uniform distribution as baseline
    2. Fine-tune positions based on content differences
    3. Ensure minimum spacing constraints
    4. Always include first and last frames

    Args:
        frame_representations: Tensor of shape [num_frames, feature_dim]
        num_keyframes: Number of keyframes to select (M)
        similarity_window: Window for computing local difference (not used in this version)
        force_boundary: If True, force first and last frames to be keyframes
        min_interval_ratio: Minimum interval = (num_frames / num_keyframes) * ratio
                           Higher values enforce more uniform distribution

    Returns:
        List of keyframe indices (sorted)

    Example:
        >>> latents = torch.randn(21, 512)  # 21 frames, 512-dim features
        >>> keyframes = identify_keyframes_improved_distribution(latents, num_keyframes=8)
        >>> print(keyframes)  # [0, 3, 6, 9, 12, 15, 18, 20] (well distributed)
    """
    num_frames = frame_representations.shape[0]

    if num_keyframes >= num_frames:
        return list(range(num_frames))

    if num_keyframes <= 2:
        if force_boundary:
            return [0, num_frames - 1] if num_frames > 1 else [0]
        else:
            return list(range(min(num_keyframes, num_frames)))

    # Normalize features for cosine similarity computation
    frame_representations = F.normalize(frame_representations, dim=1)

    # Calculate spacing constraints
    total_span = num_frames - 1
    ideal_interval = total_span / (num_keyframes - 1)
    min_interval = max(2, int(ideal_interval * min_interval_ratio))  # At least 2

    # Start with boundary frames
    keyframe_indices = []

    if force_boundary:
        keyframe_indices = [0, num_frames - 1]
        remaining_keyframes = num_keyframes - 2
        remaining_frames = num_frames - 2
    else:
        keyframe_indices = []
        remaining_keyframes = num_keyframes
        remaining_frames = num_frames

    if remaining_keyframes <= 0:
        return sorted(keyframe_indices)

    # Generate initial uniform distribution
    if remaining_keyframes > 0:
        uniform_positions = []
        for i in range(1, remaining_keyframes + 1):
            if force_boundary:
                # Distribute in the middle frames (excluding first and last)
                pos = int(round(i * (remaining_frames / (remaining_keyframes + 1))))
            else:
                # Distribute across all frames
                pos = int(round(i * (remaining_frames / remaining_keyframes)))

            if 0 < pos < num_frames - (1 if force_boundary else 0):
                uniform_positions.append(pos)

        keyframe_indices.extend(uniform_positions)

    # Fine-tune positions based on content differences
    # Only move positions if they don't violate spacing constraints
    if len(keyframe_indices) >= 3:  # Need at least 3 frames for meaningful adjustment
        adjusted_indices = [keyframe_indices[0]]  # Keep first frame fixed

        for i in range(1, len(keyframe_indices) - 1):
            current_pos = keyframe_indices[i]
            prev_pos = adjusted_indices[-1]

            # Search range for adjustment
            search_start = max(prev_pos + min_interval, current_pos - 1)
            search_end = min(keyframe_indices[i + 1] - min_interval, current_pos + 1)

            if search_start >= search_end:
                # No room to adjust, keep original position
                adjusted_indices.append(current_pos)
                continue

            # Find best position in search range based on content differences
            best_pos = current_pos
            max_total_diff = 0

            for pos in range(search_start, search_end + 1):
                if pos in keyframe_indices:  # Skip existing keyframes
                    continue

                # Calculate total difference from all existing keyframes
                total_diff = 0
                for existing_pos in adjusted_indices:
                    sim = torch.dot(frame_representations[pos], frame_representations[existing_pos])
                    diff = 1.0 - sim
                    total_diff += diff

                if total_diff > max_total_diff:
                    max_total_diff = total_diff
                    best_pos = pos

            adjusted_indices.append(best_pos)

        # Add last frame
        adjusted_indices.append(keyframe_indices[-1])
        keyframe_indices = adjusted_indices

    # Final cleanup and validation
    keyframe_indices = sorted(set(keyframe_indices))  # Remove duplicates and sort

    # Ensure we have the right number of keyframes
    while len(keyframe_indices) < num_keyframes:
        # Find the largest gap and insert a keyframe
        if len(keyframe_indices) < 2:
            # Not enough frames, just add sequential frames
            next_pos = len(keyframe_indices)
            if next_pos < num_frames:
                keyframe_indices.append(next_pos)
        else:
            # Find largest gap
            max_gap = 0
            best_insert_pos = -1
            insert_idx = -1

            for i in range(len(keyframe_indices) - 1):
                gap = keyframe_indices[i + 1] - keyframe_indices[i]
                if gap > max_gap and gap > 1:
                    max_gap = gap
                    best_insert_pos = keyframe_indices[i] + gap // 2
                    insert_idx = i + 1

            if best_insert_pos != -1 and best_insert_pos not in keyframe_indices:
                keyframe_indices.insert(insert_idx, best_insert_pos)
            else:
                break  # Cannot insert more

    # Trim if we have too many keyframes
    if len(keyframe_indices) > num_keyframes:
        # Keep first and last, remove frames that are closest to existing ones
        if force_boundary and len(keyframe_indices) > 2:
            core_indices = keyframe_indices[1:-1]
            # Calculate importance scores (based on total differences)
            scores = []
            for pos in core_indices:
                total_diff = 0
                for other_pos in keyframe_indices:
                    if other_pos != pos:
                        sim = torch.dot(frame_representations[pos], frame_representations[other_pos])
                        diff = 1.0 - sim
                        total_diff += diff
                scores.append((total_diff, pos))

            # Keep frames with highest scores
            scores.sort(reverse=True)
            keep_positions = [pos for _, pos in scores[:num_keyframes-2]]
            keyframe_indices = [0] + sorted(keep_positions) + [num_frames - 1]
        else:
            keyframe_indices = keyframe_indices[:num_keyframes]

    return sorted(keyframe_indices)


def test_keyframe_distribution():
    """
    Test function to validate keyframe distribution algorithms.
    """
    print("=== Testing Keyframe Distribution Algorithms ===")

    # Test with 21 frames and 8 keyframes (current use case)
    num_frames = 21
    num_keyframes = 8

    # Generate sample frame representations
    torch.manual_seed(42)
    frame_reps = torch.randn(num_frames, 128)

    # Test original algorithm
    print(f"\nTesting with {num_frames} frames, {num_keyframes} keyframes:")

    original_result = identify_keyframes_cosine_similarity(
        frame_reps, num_keyframes, min_interval_ratio=0.5
    )
    print(f"Original algorithm: {original_result}")
    print(f"Distribution: {visualize_keyframe_selection(original_result, num_frames)}")

    # Test improved algorithm
    improved_result = identify_keyframes_improved_distribution(
        frame_reps, num_keyframes, min_interval_ratio=0.75
    )
    print(f"Improved algorithm: {improved_result}")
    print(f"Distribution: {visualize_keyframe_selection(improved_result, num_frames)}")

    # Test fixed interval baseline
    fixed_result = identify_keyframes_fixed_interval(num_frames, num_keyframes)
    print(f"Fixed interval: {fixed_result}")
    print(f"Distribution: {visualize_keyframe_selection(fixed_result, num_frames)}")

    # Calculate spacing statistics
    def calculate_spacing(keyframes):
        if len(keyframes) < 2:
            return []
        return [keyframes[i+1] - keyframes[i] for i in range(len(keyframes)-1)]

    print(f"\nSpacing analysis:")
    print(f"Original spacing: {calculate_spacing(original_result)}")
    print(f"Improved spacing: {calculate_spacing(improved_result)}")
    print(f"Fixed spacing: {calculate_spacing(fixed_result)}")

    # Test edge cases
    print(f"\n=== Testing Edge Cases ===")

    # Test with num_keyframes = num_frames
    all_frames = identify_keyframes_improved_distribution(frame_reps, num_frames)
    print(f"All frames: {all_frames}")

    # Test with num_keyframes = 1
    single_frame = identify_keyframes_improved_distribution(frame_reps, 1)
    print(f"Single frame: {single_frame}")

    # Test with num_keyframes = 2
    two_frames = identify_keyframes_improved_distribution(frame_reps, 2)
    print(f"Two frames: {two_frames}")


if __name__ == "__main__":
    test_keyframe_distribution()
