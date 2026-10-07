use crate::{
    weights::{decode_shard, weight_specs},
    Qwen3Config,
};
use half::bf16;
use rsglang_distributed::{Shard, TensorParallel};
fn config() -> Qwen3Config {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": true,
        "hidden_act": "silu",
        "eos_token_id": 151645,
    }))
    .unwrap()
}
#[test]
fn explicit_head_dimension_and_architecture_checks() {
    let mut c = config();
    assert!(c.validate().is_ok());
    assert_ne!(c.hidden_size / c.num_attention_heads, c.head_dim);
    c.rope_scaling = Some(serde_json::json!({"type":"linear"}));
    assert!(c.validate().is_err());
    c.rope_scaling = None;
    c.attention_bias = true;
    assert!(c.validate().is_err());
}
#[test]
fn checkpoint_eos_list_is_validated() {
    let mut c = config();
    c.eos_token_id = serde_json::json!([151645, 151643]);
    assert_eq!(c.eos_ids().unwrap(), vec![151645, 151643]);
    for value in [
        serde_json::json!([151936]),
        serde_json::json!([]),
        serde_json::json!([-1]),
    ] {
        c.eos_token_id = value;
        assert!(c.eos_ids().is_err());
    }
}
#[test]
fn moe_shards_replicate_router_and_kv_but_partition_every_expert() {
    let mut c = config();
    c.model_type = "qwen3_moe".into();
    c.architectures = vec!["Qwen3MoeForCausalLM".into()];
    c.num_experts = 4;
    c.num_experts_per_tok = 2;
    c.moe_intermediate_size = 64;
    c.num_hidden_layers = 2;
    c.num_key_value_heads = 2;
    c.mlp_only_layers = vec![1];
    c.validate_tp(TensorParallel::new(7, 8).unwrap()).unwrap();
    let specs = weight_specs(&c, TensorParallel::new(7, 8).unwrap()).unwrap();
    assert_eq!(specs["model.layers.0.mlp.gate.weight"].1, Shard::Replicated);
    assert_eq!(
        specs["model.layers.0.self_attn.k_proj.weight"].1,
        Shard::rows(128..256)
    );
    for expert in 0..4 {
        let key = format!("model.layers.0.mlp.experts.{expert}.gate_proj.weight");
        assert_eq!(specs[&key].1, Shard::rows(56..64));
        let key = format!("model.layers.0.mlp.experts.{expert}.down_proj.weight");
        assert_eq!(specs[&key].1, Shard::Columns { start: 56, len: 8 });
    }
    assert!(specs.contains_key("model.layers.1.mlp.gate_proj.weight"));
    assert!(!specs.contains_key("model.layers.1.mlp.gate.weight"));
    let tp = TensorParallel::new(0, 8).unwrap();
    assert_eq!(
        c.kv_bytes_per_page_tp(16, tp).unwrap(),
        c.kv_bytes_per_page(16).unwrap() / 2
    );
    c.num_experts_per_tok = 5;
    assert!(c.validate().is_err());
}
#[test]
fn checkpoint_slices_columns_and_zero_pads_empty_vocab_ranks() {
    let values: Vec<u8> = (0..24)
        .flat_map(|i| bf16::from_f32(i as f32).to_bits().to_le_bytes())
        .collect();
    let shard = Shard::Columns { start: 2, len: 2 };
    let got = decode_shard(&values, safetensors::Dtype::BF16, &[3, 8], &shard, &[3, 2]).unwrap();
    assert_eq!(
        got.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        [2., 3., 10., 11., 18., 19.]
    );
    for rank in [6, 7] {
        let v = TensorParallel::new(rank, 8).unwrap().vocab(17).unwrap();
        assert_eq!((v.start, v.end, v.padded_rows), (17, 17, 3));
        let shard = Shard::Rows {
            start: v.start,
            len: 0,
            padded: v.padded_rows,
        };
        let got = decode_shard(&[], safetensors::Dtype::BF16, &[17, 8], &shard, &[3, 8]).unwrap();
        assert_eq!(got, vec![bf16::ZERO; 24]);
    }
}
