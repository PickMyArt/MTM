import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from Utils.utils import gpu
import numpy as np


def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


def eval_q2m(scores, q2m_gts):

    n_q, n_m = scores.shape

    gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = torch.argsort()
        rank = n_m + 1
        for k in q2m_gts[i]:
            tmp = torch.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q

    return (r1, r5, r10, r100)


def cal_perf(t2v_all_errors, t2v_gt):

    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def score_to_metrics(score_sum, t2v_gt):
    t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * score_sum, t2v_gt)
    metrics = [to_float(t2v_r1), to_float(t2v_r5), to_float(t2v_r10), to_float(t2v_r100)]
    metrics.append(sum(metrics))
    return metrics


def candidate_recall_at_k(score_sum, t2v_gt, ks):
    n_q, n_m = score_sum.shape
    if n_q == 0 or n_m == 0:
        return {int(k): 0.0 for k in ks}

    recalls = {}
    for k in ks:
        top_k = min(int(k), n_m)
        top_indices = torch.topk(score_sum, k=top_k, dim=1).indices.detach().cpu().numpy()
        hits = 0
        for q_idx in range(n_q):
            gt_indices = t2v_gt.get(q_idx, [])
            if len(gt_indices) == 0:
                continue
            if np.isin(gt_indices, top_indices[q_idx]).any():
                hits += 1
        recalls[int(k)] = 100.0 * hits / n_q
    return recalls


class validations(nn.Module):
    def __init__(self, cfg):
        super(validations, self).__init__()

        self.cfg = cfg


    def forward(self, model, context_dataloader, query_eval_loader):

        model.eval()

        context_info = self.compute_context_info(model, context_dataloader)
        score_sum, clip_score_sum, frame_score_sum, video_score_sum, query_metas = self.compute_query2ctx_info(model,
                                                             query_eval_loader,
                                                             context_info)
        video_metas = context_info['video_metas']

        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)

        raw_clip_score_sum = clip_score_sum
        raw_frame_score_sum = frame_score_sum
        raw_video_score_sum = video_score_sum
        raw_score_sum = score_sum

        clip_score_sum, frame_score_sum, video_score_sum, hubness_norm = \
            self.apply_hubness_normalization(clip_score_sum, frame_score_sum, video_score_sum)

        score_sum = self.fuse_scores(clip_score_sum, frame_score_sum, video_score_sum)
        if hubness_norm['enabled'] and hubness_norm['mode'] == 'fusion':
            score_sum = self.normalize_score_matrix(
                score_sum,
                hubness_norm['alpha'],
                hubness_norm['tau']
            )

        fusion_metrics = score_to_metrics(score_sum, t2v_gt)
        base_no_video_score_sum = self.cfg['clip_scale_w'] * clip_score_sum + self.cfg['frame_scale_w'] * frame_score_sum
        diagnostics = {
            'fusion': fusion_metrics,
            'base_no_video': score_to_metrics(base_no_video_score_sum, t2v_gt),
            'clip': score_to_metrics(clip_score_sum, t2v_gt),
            'frame': score_to_metrics(frame_score_sum, t2v_gt),
            'video': score_to_metrics(video_score_sum, t2v_gt),
            'base_candidate_recall': candidate_recall_at_k(
                base_no_video_score_sum,
                t2v_gt,
                self.cfg.get('base_candidate_recall_ks', [40, 60, 80, 100, 120, 150])
            ),
            'fusion_weights': {
                'clip': float(self.cfg['clip_scale_w']),
                'frame': float(self.cfg['frame_scale_w']),
                'video': float(self.cfg['video_scale_w'])
            },
            'gmm_widths': [float(width) for width in self.cfg.get('gmm_widths', [])],
            'video_reranker': {
                'topk': int(self.cfg.get('video_topk', 0)),
                'pair_chunk_size': int(self.cfg.get('video_pair_chunk_size', 512)),
                'safe_refine_type': self.cfg.get('safe_vgtr_refine_type', 'paper'),
                'safe_gate_init': float(self.cfg.get('safe_vgtr_gate_init', -5.0)),
                'safe_residual_scale': float(self.cfg.get('safe_vgtr_residual_scale', 1.0)),
                'safe_late_weight': float(self.cfg.get('safe_vgtr_late_weight', 0.0)),
                'safe_context': 'query_conditioned_temporal_pool',
                'safe_gate': 'pair_specific_residual_gate'
            },
            'hubness_norm': hubness_norm
        }
        if hubness_norm['enabled']:
            raw_base_no_video_score_sum = (
                self.cfg['clip_scale_w'] * raw_clip_score_sum +
                self.cfg['frame_scale_w'] * raw_frame_score_sum
            )
            diagnostics['raw_fusion'] = score_to_metrics(raw_score_sum, t2v_gt)
            diagnostics['raw_base_no_video'] = score_to_metrics(raw_base_no_video_score_sum, t2v_gt)
            diagnostics['raw_clip'] = score_to_metrics(raw_clip_score_sum, t2v_gt)
            diagnostics['raw_frame'] = score_to_metrics(raw_frame_score_sum, t2v_gt)
            diagnostics['raw_video'] = score_to_metrics(raw_video_score_sum, t2v_gt)
        model_for_diag = model.module if hasattr(model, 'module') else model
        if hasattr(model_for_diag, 'video_guided_text_refinement') and hasattr(
                model_for_diag.video_guided_text_refinement, 'residual_gate'):
            diagnostics['video_reranker']['safe_gate_value'] = to_float(
                torch.sigmoid(model_for_diag.video_guided_text_refinement.residual_gate)
            )
        if self.cfg.get('eval_fusion_grid', False):
            diagnostics['best_fusion_grid'] = self.search_fusion_weights(
                clip_score_sum, frame_score_sum, video_score_sum, t2v_gt
            )

        return fusion_metrics + [diagnostics]

    def fuse_scores(self, clip_score_sum, frame_score_sum, video_score_sum):
        return (
            self.cfg['clip_scale_w'] * clip_score_sum +
            self.cfg['frame_scale_w'] * frame_score_sum +
            self.cfg['video_scale_w'] * video_score_sum
        )

    @staticmethod
    def normalize_score_matrix(score_sum, alpha, tau):
        tau = max(float(tau), 1e-6)
        alpha = float(alpha)
        hub_bias = tau * torch.logsumexp(score_sum.float() / tau, dim=0, keepdim=True)
        return score_sum - alpha * hub_bias.to(score_sum.dtype)

    def apply_hubness_normalization(self, clip_score_sum, frame_score_sum, video_score_sum):
        enabled = bool(self.cfg.get('eval_hubness_norm', False))
        mode = str(self.cfg.get('eval_hubness_norm_mode', 'clip_frame')).lower()
        alpha = float(self.cfg.get('eval_hubness_norm_alpha', 0.15))
        tau = float(self.cfg.get('eval_hubness_norm_tau', 0.01))
        diagnostics = {
            'enabled': enabled,
            'mode': mode,
            'alpha': alpha,
            'tau': tau,
            'applied_to': []
        }
        if (not enabled) or alpha <= 0.0 or mode in ['none', 'off', 'false']:
            diagnostics['enabled'] = False
            return clip_score_sum, frame_score_sum, video_score_sum, diagnostics

        valid_modes = ['clip_frame', 'all', 'video', 'fusion']
        if mode not in valid_modes:
            raise ValueError('Unsupported eval_hubness_norm_mode: {}'.format(mode))

        if mode in ['clip_frame', 'all']:
            clip_score_sum = self.normalize_score_matrix(clip_score_sum, alpha, tau)
            frame_score_sum = self.normalize_score_matrix(frame_score_sum, alpha, tau)
            diagnostics['applied_to'].extend(['clip', 'frame'])
        if mode in ['video', 'all']:
            video_score_sum = self.normalize_score_matrix(video_score_sum, alpha, tau)
            diagnostics['applied_to'].append('video')
        if mode == 'fusion':
            diagnostics['applied_to'].append('fusion')

        return clip_score_sum, frame_score_sum, video_score_sum, diagnostics


    def search_fusion_weights(self, clip_score_sum, frame_score_sum, video_score_sum, t2v_gt):
        step = float(self.cfg.get('eval_fusion_grid_step', 0.1))
        n_steps = max(1, int(round(1.0 / step)))
        best = None
        hubness_norm = {
            'enabled': bool(self.cfg.get('eval_hubness_norm', False)),
            'mode': str(self.cfg.get('eval_hubness_norm_mode', 'clip_frame')).lower(),
            'alpha': float(self.cfg.get('eval_hubness_norm_alpha', 0.15)),
            'tau': float(self.cfg.get('eval_hubness_norm_tau', 0.01))
        }
        for clip_i in range(n_steps + 1):
            for frame_i in range(n_steps + 1 - clip_i):
                video_i = n_steps - clip_i - frame_i
                clip_w = clip_i / n_steps
                frame_w = frame_i / n_steps
                video_w = video_i / n_steps
                candidate_scores = clip_w * clip_score_sum + frame_w * frame_score_sum + video_w * video_score_sum
                if hubness_norm['enabled'] and hubness_norm['mode'] == 'fusion':
                    candidate_scores = self.normalize_score_matrix(
                        candidate_scores,
                        hubness_norm['alpha'],
                        hubness_norm['tau']
                    )
                candidate_metrics = score_to_metrics(candidate_scores, t2v_gt)
                if best is None or candidate_metrics[4] > best['metrics'][4]:
                    best = {
                        'weights': {'clip': clip_w, 'frame': frame_w, 'video': video_w},
                        'metrics': candidate_metrics
                    }
        return best


    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info):

        query_metas = []
        score_sum = []
        clip_score_sum = []
        frame_score_sum = []
        video_score_sum = []

        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]

            _clip_scale_scores, _frame_scale_scores, _video_scale_scores = model.get_pred_from_raw_query(
                query_feat, query_mask, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"],
                ctx_info["video_global_feat"],
                encoded_video_sequence=ctx_info["video_sequence_feat"],
                encoded_video_mask=ctx_info["video_mask"],
                encoded_frame_mask=ctx_info["video_mask"],
                return_query_feats=False)

            _score_sum = self.cfg['clip_scale_w'] * _clip_scale_scores + \
                         self.cfg['frame_scale_w'] * _frame_scale_scores + \
                         self.cfg['video_scale_w'] * _video_scale_scores

            score_sum.append(_score_sum)
            clip_score_sum.append(_clip_scale_scores)
            frame_score_sum.append(_frame_scale_scores)
            video_score_sum.append(_video_scale_scores)


        score_sum = torch.cat(score_sum, dim=0)
        clip_score_sum = torch.cat(clip_score_sum, dim=0)
        frame_score_sum = torch.cat(frame_score_sum, dim=0)
        video_score_sum = torch.cat(video_score_sum, dim=0)


        return score_sum, clip_score_sum, frame_score_sum, video_score_sum, query_metas


    def compute_context_info(self, model, context_dataloader):

        metas = []
        vid_proposal_feat = []
        frame_feat, frame_mask = [], []
        video_global_feat = []
        video_sequence_feat = []

        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing video embedding",
                            total=len(context_dataloader)):

            batch = gpu(batch)
            metas.extend(batch[-1])

            clip_video_feat_ = batch[0]
            frame_video_feat_ = batch[2]
            frame_mask_ = batch[3]

            _frame_feat, _video_proposal_feat, _video_global_feat, _video_sequence_feat = model.encode_context(
                clip_video_feat_, frame_video_feat_, frame_mask_, return_video_sequence=True
            ) 

            frame_feat.append(_frame_feat)
            frame_mask.append(frame_mask_)

            vid_proposal_feat.append(_video_proposal_feat)
            video_global_feat.append(_video_global_feat)
            video_sequence_feat.append(_video_sequence_feat)

        vid_proposal_feat = torch.cat(vid_proposal_feat, dim=0)
        video_global_feat = torch.cat(video_global_feat, dim=0)

        def cat_tensor(tensor_list):
            if len(tensor_list) == 0:
                return None
            else:
                seq_l = [e.shape[1] for e in tensor_list]
                b_sizes = [e.shape[0] for e in tensor_list]
                b_sizes_cumsum = np.cumsum([0] + b_sizes)
                if len(tensor_list[0].shape) == 3:
                    hsz = tensor_list[0].shape[2]
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
                elif len(tensor_list[0].shape) == 2:
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
                else:
                    raise ValueError("Only support 2/3 dimensional tensors")
                for i, e in enumerate(tensor_list):
                    res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
                return res_tensor
                
        return dict(
            video_metas=metas,
            video_proposal_feat=vid_proposal_feat,
            video_feat=cat_tensor(frame_feat),
            video_mask=cat_tensor(frame_mask),
            video_global_feat=video_global_feat,
            video_sequence_feat=cat_tensor(video_sequence_feat)
            )
