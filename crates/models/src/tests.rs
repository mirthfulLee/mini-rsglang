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

fn gpt_config() -> crate::GptOssConfig {
    serde_json::from_value(serde_json::json!({
        "model_type":"gpt_oss", "architectures":["GptOssForCausalLM"],
        "hidden_size":64, "intermediate_size":64, "num_hidden_layers":2,
        "num_attention_heads":8, "num_key_value_heads":2, "head_dim":8,
        "vocab_size":67, "max_position_embeddings":64, "num_local_experts":4,
        "num_experts_per_tok":2, "rms_norm_eps":1e-5, "rope_theta":150000,
        "layer_types":["sliding_attention","full_attention"], "sliding_window":8,
        "attention_bias":true, "hidden_act":"silu", "tie_word_embeddings":false,
        "eos_token_id":66
    }))
    .unwrap()
}
#[test]
fn gpt_oss_rejects_invalid_architecture_routing_windows_and_scaling() {
    let c = gpt_config();
    c.validate_tp(TensorParallel::new(3, 4).unwrap()).unwrap();
    let mut bad = c.clone();
    bad.num_experts_per_tok = 5;
    assert!(bad.validate_tp(TensorParallel::default()).is_err());
    let mut bad = c.clone();
    bad.layer_types[0] = "unknown".into();
    assert!(bad.validate_tp(TensorParallel::default()).is_err());
    let mut bad = c.clone();
    bad.sliding_window = 0;
    assert!(bad.validate_tp(TensorParallel::default()).is_err());
    let mut bad = c.clone();
    bad.eos_token_id = serde_json::json!([67]);
    assert!(bad.validate_tp(TensorParallel::default()).is_err());
    let mut bad = c.clone();
    bad.quantization_config = Some(serde_json::json!({"quant_method":"fp8"}));
    assert!(bad.validate_tp(TensorParallel::default()).is_err());
    let mut bad = c;
    bad.rope_scaling=Some(serde_json::from_value(serde_json::json!({"rope_type":"yarn","factor":0.0,"original_max_position_embeddings":16,"beta_fast":32.0,"beta_slow":1.0})).unwrap());
    assert!(bad.validate_tp(TensorParallel::default()).is_err());
}
#[test]
fn streamed_checkpoint_checks_byte_ranges_before_reading() {
    use crate::checkpoint::Checkpoint;
    let path = std::env::temp_dir().join(format!(
        "rsglang-checkpoint-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let write = |header: serde_json::Value, data: &[u8]| {
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        std::fs::write(path.join("model.safetensors"), bytes).unwrap();
    };
    write(
        serde_json::json!({"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}),
        &[128, 63],
    );
    let mut cp = Checkpoint::open(&path).unwrap();
    let tensor = cp.take("weight", &[1]).unwrap();
    assert_eq!(tensor.read().unwrap(), [128, 63]);
    assert!(tensor.read_range(1, 2).is_err());
    cp.finish().unwrap();
    write(
        serde_json::json!({"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,3]}}),
        &[128, 63],
    );
    assert!(Checkpoint::open(&path).is_err());
    write(
        serde_json::json!({"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}),
        &[128, 63],
    );
    assert!(Checkpoint::open(&path).is_err());
    std::fs::write(path.join("model.safetensors"), u64::MAX.to_le_bytes()).unwrap();
    assert!(Checkpoint::open(&path).is_err());
    std::fs::remove_dir_all(path).unwrap();
}
