import os
import yaml
import time
cfg = {}
_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg['model_name'] = 'MTM'
cfg['dataset_name'] = 'tvr'
cfg['seed'] = 9527
cfg['root'] = _src_root + os.sep
cfg['data_root'] = os.path.join(_src_root, 'data')
cfg['visual_feature'] = 'i3d_resnet'
cfg['collection'] = 'tvr'
cfg['map_size'] = 32
cfg['clip_scale_w'] = 0.7
cfg['frame_scale_w'] = 0.35
cfg['video_scale_w'] = 0.03
cfg['model_root'] = os.path.join(cfg['root'], 'results-tvr', cfg['dataset_name'], cfg['model_name'], time.strftime("%Y_%m_%d_%H_%M_%S"))
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')
cfg['sft_factor'] = 0.09
cfg['gmm_widths'] = [0.5, 1.0, 5.0, 10.0, 3.0, 0.1, 8.0, 0.05, 2.0, 15.0]

# dataset
cfg['num_workers'] = 16
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 32

# EMA Configuration
cfg['use_ema'] = True 
cfg['ema_decay'] = 0.9995 

# opt
cfg['lr'] = 0.00025
cfg['lr_warmup_proportion'] = 0.01
cfg['wd'] = 0.01
cfg['margin'] = 0.1


# train
cfg['n_epoch'] = 150
cfg['max_es_cnt'] = 10
cfg['hard_negative_start_epoch'] = 50
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = False

# Loss Weights
# loss_factor: [clip_nce, frame_nce, qdl, ot_loss_factor, video_nce]
cfg['loss_factor'] = [0.05, 0.04, 7e-05, 0.09, 0.015]
cfg['triplet_loss_factor'] = [1.0, 1.0, 1.0] 
cfg['neg_factor'] = [0.15, 32, 1]
cfg['lambda_recon'] = 0.4


# eval
cfg['eval_query_bsz'] = 50
cfg['eval_context_bsz'] = 100
cfg['eval_hubness_norm'] = False
cfg['eval_hubness_norm_mode'] = 'clip_frame'
cfg['eval_hubness_norm_alpha'] = 0.15
cfg['eval_hubness_norm_tau'] = 0.01


# model
cfg['max_desc_l'] = 30
cfg['max_ctx_l'] = 128
cfg['sub_feat_size'] = 768
cfg['q_feat_size'] = 768
cfg['visual_feat_dim'] = 1024
cfg['max_position_embeddings'] = 300
cfg['hidden_size'] = 384
cfg['n_heads'] = 4
cfg['input_drop'] = 0.2
cfg['drop'] = 0.2
cfg['initializer_range'] = 0.02
cfg['num_mamba_layers'] = 2


cfg['num_workers'] = 1 if cfg['no_core_driver'] else cfg['num_workers']
cfg['pin_memory'] = not cfg['no_pin_memory']


if not os.path.exists(cfg['model_root']):
    os.makedirs(cfg['model_root'], exist_ok=True)
if not os.path.exists(cfg['ckpt_path']):
    os.makedirs(cfg['ckpt_path'], exist_ok=True)


def get_cfg_defaults():
    with open(os.path.join(cfg['model_root'], 'hyperparams.yaml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    return cfg
