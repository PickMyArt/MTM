import os
import yaml
import time
cfg = {}
_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg['model_name'] = 'MTM'
cfg['dataset_name'] = 'charades'
cfg['seed'] = 9527
cfg['root'] = _src_root + os.sep
cfg['data_root'] = os.path.join(_src_root, 'data')
cfg['visual_feature'] = 'i3d_rgb_lgi'
cfg['collection'] = 'charades'
cfg['map_size'] = 32
cfg['clip_scale_w'] = 0.74
cfg['frame_scale_w'] = 0.21
cfg['video_scale_w'] = 0.05
cfg['model_root'] = os.path.join(cfg['root'], 'results_cha', cfg['dataset_name'], cfg['model_name'], time.strftime("%Y_%m_%d_%H_%M_%S"))
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')

# extra
cfg['sft_factor'] = 0.6
cfg['gmm_widths'] = [0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.65, 1.0, 1.6, 2.5, 4.0, 6.0, 10.0]

# dataset
cfg['num_workers'] = 8
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 32

# opt
cfg['lr'] = 0.00016
cfg['lr_warmup_proportion'] = 0.02
cfg['wd'] = 0.01
cfg['margin'] = 0.2

# train
cfg['n_epoch'] = 200
cfg['max_es_cnt'] = 12
cfg['hard_negative_start_epoch'] = 14
cfg['min_hard_negative_epochs'] = 20
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = False

# loss_factor: [clip_nce, frame_nce, qdl, ot_loss_factor, video_nce]
cfg['loss_factor'] = [0.02, 0.03, 0.003, 0.11, 0.02]
cfg['triplet_loss_factor'] = [1.0, 1.0, 0.8] 
cfg['neg_factor'] = [0.19, 32, 1]
cfg['lambda_recon'] = 0.6

# eval
cfg['eval_query_bsz'] = 50
cfg['eval_context_bsz'] = 100
cfg['eval_fusion_grid'] = True
cfg['eval_fusion_grid_step'] = 0.1
cfg['eval_hubness_norm'] = True
cfg['eval_hubness_norm_mode'] = 'clip_frame'
cfg['eval_hubness_norm_alpha'] = 0.16
cfg['eval_hubness_norm_tau'] = 0.007
cfg['video_topk'] = 80
cfg['video_pair_chunk_size'] = 512
cfg['safe_vgtr_refine_type'] = 'paper'
cfg['safe_vgtr_gate_init'] = -4.75
cfg['safe_vgtr_residual_scale'] = 0.35
cfg['safe_vgtr_late_weight'] = 0.0

# model
cfg['max_desc_l'] = 30  
cfg['max_ctx_l'] = 128    
cfg['q_feat_size'] = 1024  
cfg['visual_feat_dim'] = 1024 
cfg['hidden_size'] = 384 
cfg['n_heads'] = 4  
cfg['input_drop'] = 0.25
cfg['drop'] = 0.25
cfg['initializer_range'] = 0.02
cfg['num_mamba_layers'] = 2
cfg['use_ema'] = False 
cfg['ema_decay'] = 0.9995
cfg['save_checkpoint'] = True
cfg['num_workers'] = 1 if cfg['no_core_driver'] else cfg['num_workers']
cfg['pin_memory'] = not cfg['no_pin_memory']

def _override_float(cfg_key, env_key):
    value = os.environ.get(env_key)
    if value is not None:
        cfg[cfg_key] = float(value)

def _override_int(cfg_key, env_key):
    value = os.environ.get(env_key)
    if value is not None:
        cfg[cfg_key] = int(value)

def _override_bool(cfg_key, env_key):
    value = os.environ.get(env_key)
    if value is not None:
        cfg[cfg_key] = value.lower() in ['1', 'true', 'yes', 'y']

def _override_str(cfg_key, env_key):
    value = os.environ.get(env_key)
    if value is not None:
        cfg[cfg_key] = value

def _override_float_list(cfg_key, env_key):
    value = os.environ.get(env_key)
    if value is not None:
        items = [item.strip() for item in value.split(',') if item.strip()]
        cfg[cfg_key] = [float(item) for item in items]

_override_float('clip_scale_w', 'MTM_CLIP_SCALE_W')
_override_float('frame_scale_w', 'MTM_FRAME_SCALE_W')
_override_float('video_scale_w', 'MTM_VIDEO_SCALE_W')
_override_float('lr', 'MTM_LR')
_override_float('margin', 'MTM_MARGIN')
_override_float('input_drop', 'MTM_INPUT_DROP')
_override_float('drop', 'MTM_DROP')
_override_float('lambda_recon', 'MTM_LAMBDA_RECON')
_override_float('safe_vgtr_gate_init', 'MTM_SAFE_VGTR_GATE_INIT')
_override_float('safe_vgtr_residual_scale', 'MTM_SAFE_VGTR_RESIDUAL_SCALE')
_override_float('safe_vgtr_late_weight', 'MTM_SAFE_VGTR_LATE_WEIGHT')
_override_float('eval_hubness_norm_alpha', 'MTM_EVAL_HUBNESS_NORM_ALPHA')
_override_float('eval_hubness_norm_tau', 'MTM_EVAL_HUBNESS_NORM_TAU')
_override_int('video_topk', 'MTM_VIDEO_TOPK')
_override_int('video_pair_chunk_size', 'MTM_VIDEO_PAIR_CHUNK_SIZE')
_override_int('hard_negative_start_epoch', 'MTM_HARD_NEGATIVE_START_EPOCH')
_override_int('hard_pool_size', 'MTM_HARD_POOL_SIZE')
_override_int('max_es_cnt', 'MTM_MAX_ES_CNT')
_override_int('batchsize', 'MTM_BATCHSIZE')
_override_int('num_workers', 'MTM_NUM_WORKERS')
_override_bool('eval_fusion_grid', 'MTM_EVAL_FUSION_GRID')
_override_bool('eval_hubness_norm', 'MTM_EVAL_HUBNESS_NORM')
_override_bool('save_checkpoint', 'MTM_SAVE_CHECKPOINT')
_override_str('safe_vgtr_refine_type', 'MTM_SAFE_VGTR_REFINE_TYPE')
_override_str('eval_hubness_norm_mode', 'MTM_EVAL_HUBNESS_NORM_MODE')
_override_float_list('loss_factor', 'MTM_LOSS_FACTOR')
_override_float_list('triplet_loss_factor', 'MTM_TRIPLET_LOSS_FACTOR')
_override_float_list('gmm_widths', 'MTM_GMM_WIDTHS')

run_tag = os.environ.get('MTM_RUN_TAG')
if run_tag:
    safe_tag = ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in run_tag)
    cfg['model_root'] = '{}_{}'.format(cfg['model_root'], safe_tag)
    cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')

if not os.path.exists(cfg['model_root']):
    os.makedirs(cfg['model_root'], exist_ok=True)
if not os.path.exists(cfg['ckpt_path']):
    os.makedirs(cfg['ckpt_path'], exist_ok=True)

def get_cfg_defaults():
    with open(os.path.join(cfg['model_root'], 'hyperparams.yaml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    return cfg
