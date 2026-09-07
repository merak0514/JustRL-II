import re
import torch


def convert_minicpm5_moe_to_hf(args, name, param):
    """
    Convert MiniCPM5 MoE weights from Megatron format to HuggingFace format.
    
    Key differences from standard MiniCPM:
    - Gated Attention: q_proj contains both query and gate (interleaved)
    - MoE layers: has experts, shared_experts, and router with expert_bias
    - Mixed dense/MoE layers (first_k_dense_replace=1)
    """
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    
    num_attention_heads = args.num_attention_heads
    num_kv_heads = args.num_query_groups
    value_num_per_group = num_attention_heads // num_kv_heads
    
    # Check if gated attention is enabled
    use_gated_attention = getattr(args, 'attention_output_gate', False)

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        
        # Attention weights
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        
        elif rest == "self_attention.linear_qkv.weight":
            if use_gated_attention:
                # For Gated Attention: QKV layout is [num_kv_heads, (q_per_kv + gate_per_kv + k + v) * head_dim, hidden_size]
                # Total dim1 = num_kv_heads * (value_num_per_group * 2 + 2) * head_dim
                # For MiniCPM5 MoE: 2 * (10 * 2 + 2) * 128 = 2 * 22 * 128 = 5632
                param = param.view(num_kv_heads, -1, head_dim, args.hidden_size)
                # Split: query, gate, k, v
                # dim1 per kv_head = (value_num_per_group + value_num_per_group + 1 + 1) * head_dim / head_dim
                #                  = value_num_per_group * 2 + 2
                q_param, gate_param, k_param, v_param = torch.split(
                    param, 
                    split_size_or_sections=[value_num_per_group, value_num_per_group, 1, 1], 
                    dim=1
                )
                q_param = q_param.reshape(-1, args.hidden_size)  # [num_heads * head_dim, hidden_size]
                gate_param = gate_param.reshape(-1, args.hidden_size)  # [num_heads * head_dim, hidden_size]
                k_param = k_param.reshape(-1, args.hidden_size)  # [num_kv_heads * head_dim, hidden_size]
                v_param = v_param.reshape(-1, args.hidden_size)  # [num_kv_heads * head_dim, hidden_size]
                
                # HF/SGLang: q_proj is packed per-head as [q_head0, gate_head0, q_head1, gate_head1, ...].
                try:
                    qh = q_param.view(num_attention_heads, head_dim, args.hidden_size)
                    gh = gate_param.view(num_attention_heads, head_dim, args.hidden_size)
                    q_proj = torch.stack([qh, gh], dim=1).reshape(-1, args.hidden_size)
                except Exception:
                    q_proj = torch.cat([q_param, gate_param], dim=0)
                
                return [
                    (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_proj),
                    (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                    (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
                ]
            else:
                # Standard GQA without gated attention
                param = param.view(num_kv_heads, -1, head_dim, args.hidden_size)
                q_param, k_param, v_param = torch.split(
                    param, 
                    split_size_or_sections=[value_num_per_group, 1, 1], 
                    dim=1
                )
                q_param = q_param.reshape(-1, args.hidden_size)
                k_param = k_param.reshape(-1, args.hidden_size)
                v_param = v_param.reshape(-1, args.hidden_size)
                return [
                    (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                    (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                    (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
                ]
        
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        
        # Dense MLP (layer 0)
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight" or rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        
        # MoE Router
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]
        
        # MoE Shared Experts - MCore name: mlp.shared_experts.linear_fc1.weight
        elif rest == "mlp.shared_experts.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.shared_experts.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
        
        # MoE Routed Experts - format: mlp.experts.linear_fc1.weight{expert_id}
        elif "mlp.experts.linear_fc1.weight" in rest:
            expert_id = rest.split("weight")[-1]
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight", up_weight),
            ]
        elif "mlp.experts.linear_fc2.weight" in rest:
            expert_id = rest.split("weight")[-1]
            return [(f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
