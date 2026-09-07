"""Fix MiniCPM5 MoE expert_bias mismatch between training and inference.

The Megatron checkpoint conversion does not include expert_bias, so training
uses zeros while SGLang inference loads non-zero values from the HF checkpoint,
causing a large train-inference discrepancy.

Solution: after loading the Megatron model, manually copy e_score_correction_bias
from the HF checkpoint into the router.expert_bias buffers.
"""

import logging
import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


def fix_expert_bias_from_hf(model, hf_checkpoint_path: str):
    """Load e_score_correction_bias from HF checkpoint into Megatron expert_bias buffers.

    Args:
        model: Megatron GPT model (list of model chunks for virtual pipeline).
        hf_checkpoint_path: Path to HuggingFace checkpoint (model.safetensors).
    """
    logger.info(f"[fix_expert_bias] Loading e_score_correction_bias from HF checkpoint: {hf_checkpoint_path}")

    try:
        f = safe_open(hf_checkpoint_path, framework='pt')
    except Exception as e:
        logger.error(f"[fix_expert_bias] Failed to load HF checkpoint: {e}")
        return False
    
    hf_bias_keys = [k for k in f.keys() if 'e_score_correction_bias' in k]
    if not hf_bias_keys:
        logger.warning("[fix_expert_bias] No e_score_correction_bias found in HF checkpoint")
        return False
    
    logger.info(f"[fix_expert_bias] Found {len(hf_bias_keys)} e_score_correction_bias keys in HF checkpoint")
    
    fixed_count = 0

    for model_chunk in model:
        for name, buffer in model_chunk.named_buffers():
            if 'expert_bias' in name:
                # Extract layer index from name, e.g.
                # module.module.decoder.layers.{idx}.mlp.router.expert_bias
                parts = name.split('.')
                try:
                    layer_idx = None
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            break
                    
                    if layer_idx is None:
                        logger.warning(f"[fix_expert_bias] Could not extract layer index from: {name}")
                        continue
                    
                    hf_key = f'model.layers.{layer_idx}.mlp.gate.e_score_correction_bias'

                    if hf_key in hf_bias_keys:
                        hf_bias = f.get_tensor(hf_key)

                        if buffer.shape == hf_bias.shape:
                            old_mean = buffer.mean().item()
                            buffer.copy_(hf_bias.to(buffer.device))
                            new_mean = buffer.mean().item()
                            logger.info(f"[fix_expert_bias] Fixed {name}: mean {old_mean:.4f} -> {new_mean:.4f}")
                            fixed_count += 1
                        else:
                            logger.warning(f"[fix_expert_bias] Shape mismatch for {name}: "
                                         f"buffer {buffer.shape} vs HF {hf_bias.shape}")
                    else:
                        logger.warning(f"[fix_expert_bias] HF key not found: {hf_key}")
                        
                except Exception as e:
                    logger.error(f"[fix_expert_bias] Error processing {name}: {e}")
    
    logger.info(f"[fix_expert_bias] Fixed {fixed_count} expert_bias buffers")
    return fixed_count > 0


def verify_expert_bias(model, expected_mean_range=(10.0, 25.0)):
    """Verify that expert_bias values are within a reasonable range.

    If the mean is close to 0, the bias was not loaded correctly.
    MiniCPM5 MoE e_score_correction_bias values are typically 14-20.
    """
    all_valid = True
    
    for model_chunk in model:
        for name, buffer in model_chunk.named_buffers():
            if 'expert_bias' in name:
                mean_val = buffer.mean().item()
                min_val = buffer.min().item()
                max_val = buffer.max().item()
                
                if mean_val < expected_mean_range[0] or mean_val > expected_mean_range[1]:
                    logger.warning(f"[verify_expert_bias] {name} has unexpected mean: {mean_val:.4f} "
                                  f"(expected {expected_mean_range[0]}-{expected_mean_range[1]})")
                    all_valid = False
                else:
                    logger.info(f"[verify_expert_bias] {name}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f} OK")
    
    return all_valid
