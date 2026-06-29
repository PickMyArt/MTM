from easydict import EasyDict as EDict
from Models.MTM.model import MTM_Net
def get_models(cfg):
    model_config = EDict(
        visual_input_size=cfg["visual_feat_dim"], 
        visual_feat_dim=cfg["visual_feat_dim"], 
        q_feat_size=cfg["q_feat_size"],
        hidden_size=cfg["hidden_size"],  
        max_ctx_l=cfg["max_ctx_l"],
        max_desc_l=cfg["max_desc_l"],
        map_size=cfg["map_size"],
        input_drop=cfg["input_drop"],
        drop=cfg["drop"], 
        n_heads=cfg["n_heads"],  
        initializer_range=cfg["initializer_range"],  
        margin=cfg["margin"],  
        use_hard_negative=False, 
        hard_pool_size=cfg["hard_pool_size"],
        sft_factor=cfg["sft_factor"],
        gmm_widths=cfg.get("gmm_widths", [0.5, 1.0, 5.0, 10.0, 3.0, 0.1, 8.0, 0.05, 2.0, 15.0]),
        num_mamba_layers=cfg["num_mamba_layers"],
        clip_scale_w=cfg.get("clip_scale_w", 1.0),
        frame_scale_w=cfg.get("frame_scale_w", 1.0),
        video_topk=cfg.get("video_topk", 100),
        video_pair_chunk_size=cfg.get("video_pair_chunk_size", 512),
        safe_vgtr_gate_init=cfg.get("safe_vgtr_gate_init", -5.0),
        safe_vgtr_refine_type=cfg.get("safe_vgtr_refine_type", "paper"),
        safe_vgtr_residual_scale=cfg.get("safe_vgtr_residual_scale", 1.0),
        safe_vgtr_late_weight=cfg.get("safe_vgtr_late_weight", 0.0),
    )
    model = MTM_Net(model_config)
    return model
