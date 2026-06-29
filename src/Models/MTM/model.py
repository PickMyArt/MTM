import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from Models.MTM.model_components import BertAttention, LinearLayer, TrainablePositionalEncoding, DyGMMBlock
from mamba_ssm import Mamba
from scipy.optimize import linear_sum_assignment

class MLPReconstructor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, hidden_dim_factor=2, dropout=0.1):
        super(MLPReconstructor, self).__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            next_dim = int(current_dim * hidden_dim_factor)
            if next_dim > output_dim:
                next_dim = output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TCN(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(TCN, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        out = self.layer_norm(x + alpha_clamped * y)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attn_proj = nn.Linear(hidden_size, 1) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        attn_scores = self.attn_proj(x) 
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(2) == 0, -1e9)

        attn_weights = self.softmax(attn_scores) 
        pooled_output = torch.sum(x * attn_weights, dim=1) 
        return pooled_output

class VideoGuidedTextRefiner(nn.Module):
    def __init__(self, config):
        super(VideoGuidedTextRefiner, self).__init__()
        self.hidden_size = config.hidden_size
        self.safe_residual_scale = float(getattr(config, 'safe_vgtr_residual_scale', 1.0))
        self.safe_late_weight = float(getattr(config, 'safe_vgtr_late_weight', 0.0))
        gate_init = float(getattr(config, 'safe_vgtr_gate_init', -5.0))
        self.residual_gate = nn.Parameter(torch.tensor(gate_init))
        self.context_score_scale = config.hidden_size ** -0.5
        self.refine_context_pool_proj = nn.Linear(config.hidden_size, 1)
        self.context_query_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_video_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pair_gate_proj = nn.Linear(config.hidden_size * 3, 1)
        self.safe_vgtr_refine_type = getattr(config, 'safe_vgtr_refine_type', 'paper')
        if self.safe_vgtr_refine_type == 'paper':
            self.token_text_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.token_context_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.token_score_bias = nn.Parameter(torch.zeros(config.hidden_size))
            self.token_score_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.safe_output_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.safe_output_norm = nn.LayerNorm(config.hidden_size)
            self.token_weight_net = None
        elif self.safe_vgtr_refine_type == 'mlp':
            self.token_text_proj = None
            self.token_context_proj = None
            self.token_score_bias = None
            self.token_score_proj = None
            self.safe_output_proj = None
            self.safe_output_norm = None
            self.token_weight_net = nn.Sequential(
                nn.Linear(config.hidden_size * 3, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(config.drop),
                nn.Linear(config.hidden_size, 1)
            )
        else:
            raise ValueError('Unsupported safe_vgtr_refine_type: {}'.format(self.safe_vgtr_refine_type))

    def reset_safe_parameters(self):
        self.pair_gate_proj.weight.data.zero_()
        if self.pair_gate_proj.bias is not None:
            self.pair_gate_proj.bias.data.zero_()

    def refine_context_pool(self, video_sequence_feat, video_mask=None, query_feat=None):
        """Build candidate context for text-token reweighting.

        When query_feat is provided, the temporal evidence is selected by the
        current query-candidate pair. This keeps VGTR candidate-adaptive without
        injecting visual values into the refined query.
        """
        if query_feat is None:
            context_logits = self.refine_context_pool_proj(video_sequence_feat).squeeze(-1)
        else:
            query_proj = self.context_query_proj(query_feat).unsqueeze(1)
            video_proj = self.context_video_proj(video_sequence_feat)
            context_logits = torch.sum(query_proj * video_proj, dim=-1) * self.context_score_scale
        if video_mask is not None:
            valid_mask = video_mask.to(torch.bool)
            context_logits = context_logits.masked_fill(~valid_mask, -10000.0)
        context_weights = F.softmax(context_logits, dim=-1)
        if video_mask is not None:
            context_weights = context_weights * valid_mask.to(context_weights.dtype)
            context_weights = context_weights / context_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context_feat = torch.sum(video_sequence_feat * context_weights.unsqueeze(-1), dim=1)
        return context_feat, context_weights

    @staticmethod
    def _build_pair_indices(n_query, n_video, candidate_indices, device):
        if candidate_indices is None:
            query_indices = torch.arange(n_query, device=device).repeat_interleave(n_video)
            video_indices = torch.arange(n_video, device=device).repeat(n_query)
            output_shape = (n_query, n_video)
        else:
            if candidate_indices.dim() == 1:
                candidate_indices = candidate_indices.unsqueeze(1)
            n_candidates = candidate_indices.shape[1]
            query_indices = torch.arange(n_query, device=device).unsqueeze(1).expand(
                n_query, n_candidates
            ).reshape(-1)
            video_indices = candidate_indices.reshape(-1).to(device)
            output_shape = (n_query, n_candidates)
        return query_indices, video_indices, output_shape

    def refine_text_tokens_with_context(self, query_token_feat, query_mask, query_base_feat, video_context_feat):
        """Refine query embeddings through candidate-conditioned token reweighting."""
        if self.safe_vgtr_refine_type == 'paper':
            token_hidden = torch.tanh(
                self.token_text_proj(query_token_feat) +
                self.token_context_proj(video_context_feat).unsqueeze(1) +
                self.token_score_bias.view(1, 1, -1)
            )
            token_logits = self.token_score_proj(token_hidden).squeeze(-1)
        else:
            video_context = video_context_feat.unsqueeze(1).expand(-1, query_token_feat.shape[1], -1)
            token_inputs = torch.cat(
                [query_token_feat, video_context, query_token_feat * video_context],
                dim=-1
            )
            token_logits = self.token_weight_net(token_inputs).squeeze(-1)
        if query_mask is not None:
            valid_mask = query_mask.to(torch.bool)
            token_logits = token_logits.masked_fill(~valid_mask, -10000.0)
        token_weights = F.softmax(token_logits, dim=-1)
        if query_mask is not None:
            token_weights = token_weights * valid_mask.to(token_weights.dtype)
            token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        reweighted_query = torch.sum(query_token_feat * token_weights.unsqueeze(-1), dim=1)
        gate_delta = self.pair_gate_proj(
            torch.cat([query_base_feat, reweighted_query, video_context_feat], dim=-1)
        ).squeeze(-1)
        gate = torch.sigmoid(self.residual_gate + gate_delta).unsqueeze(-1)
        if self.safe_vgtr_refine_type == 'paper':
            projected_query = self.safe_output_proj(reweighted_query)
            refined_query = self.safe_output_norm((1.0 - gate) * query_base_feat + gate * projected_query)
        else:
            refined_query = query_base_feat + gate * (reweighted_query - query_base_feat)
        return refined_query, token_weights

    @staticmethod
    def late_interaction_score(query_token_feat, query_mask, video_sequence_feat, video_mask=None):
        query_norm = F.normalize(query_token_feat, dim=-1)
        video_norm = F.normalize(video_sequence_feat, dim=-1)
        sim = torch.bmm(query_norm, video_norm.transpose(1, 2))
        if video_mask is not None:
            video_valid = video_mask.to(torch.bool)
            sim = sim.masked_fill(~video_valid.unsqueeze(1), -10000.0)
        token_max = sim.max(dim=2).values
        if query_mask is not None:
            query_valid = query_mask.to(torch.bool)
            token_max = token_max.masked_fill(~query_valid, 0.0)
            denom = query_valid.to(token_max.dtype).sum(dim=1).clamp_min(1.0)
            return token_max.sum(dim=1) / denom
        return token_max.mean(dim=1)

    def score_safe_pairs(
            self, query_feat_2d, query_token_feat, query_mask, video_global_feat,
            video_sequence_feat=None, video_mask=None, candidate_indices=None,
            chunk_size=512, fallback_scores=None
    ):
        n_query = query_feat_2d.shape[0]
        n_video = video_global_feat.shape[0]
        device = query_feat_2d.device
        query_indices, video_indices, output_shape = self._build_pair_indices(
            n_query, n_video, candidate_indices, device
        )

        pair_count = query_indices.numel()
        pair_scores = query_feat_2d.new_empty(pair_count)
        pair_scores_unnormalized = query_feat_2d.new_empty(pair_count)
        chunk_size = max(1, int(chunk_size))

        if fallback_scores is not None:
            fallback_norm, fallback_unnorm = fallback_scores
        else:
            fallback_norm, fallback_unnorm = None, None

        for start in range(0, pair_count, chunk_size):
            end = min(start + chunk_size, pair_count)
            q_idx = query_indices[start:end]
            v_idx = video_indices[start:end]
            query_chunk = query_feat_2d[q_idx]
            query_token_chunk = query_token_feat[q_idx]
            query_mask_chunk = query_mask[q_idx] if query_mask is not None else None
            video_match_chunk = video_global_feat[v_idx]
            video_chunk = None
            video_mask_chunk = None
            if video_sequence_feat is not None:
                video_chunk = video_sequence_feat[v_idx]
                video_mask_chunk = video_mask[v_idx] if video_mask is not None else None
                video_context_chunk, _ = self.refine_context_pool(
                    video_chunk, video_mask_chunk, query_chunk
                )
            else:
                video_context_chunk = video_match_chunk

            refined_query, _ = self.refine_text_tokens_with_context(
                query_token_chunk,
                query_mask_chunk,
                query_chunk,
                video_context_chunk
            )

            safe_scores = torch.sum(
                F.normalize(refined_query, dim=-1) * F.normalize(video_match_chunk, dim=-1),
                dim=-1
            )
            safe_scores_unnormalized = torch.sum(refined_query * video_match_chunk, dim=-1)

            if self.safe_late_weight > 0.0 and video_chunk is not None:
                late_scores = self.late_interaction_score(
                    query_token_chunk, query_mask_chunk, video_chunk, video_mask_chunk
                )
                safe_scores = safe_scores + self.safe_late_weight * late_scores
                safe_scores_unnormalized = safe_scores_unnormalized + self.safe_late_weight * late_scores

            if fallback_norm is not None:
                base_scores = fallback_norm[q_idx, v_idx]
                base_scores_unnormalized = fallback_unnorm[q_idx, v_idx]
                pair_scores[start:end] = base_scores + self.safe_residual_scale * (safe_scores - base_scores)
                pair_scores_unnormalized[start:end] = base_scores_unnormalized + self.safe_residual_scale * (
                    safe_scores_unnormalized - base_scores_unnormalized
                )
            else:
                pair_scores[start:end] = safe_scores
                pair_scores_unnormalized[start:end] = safe_scores_unnormalized

        if candidate_indices is None:
            return pair_scores.view(*output_shape), pair_scores_unnormalized.view(*output_shape)

        if fallback_scores is None:
            dense_scores = query_feat_2d.new_full((n_query, n_video), -10000.0)
            dense_scores_unnormalized = query_feat_2d.new_full((n_query, n_video), -10000.0)
        else:
            dense_scores, dense_scores_unnormalized = fallback_scores
            dense_scores = dense_scores.clone()
            dense_scores_unnormalized = dense_scores_unnormalized.clone()

        dense_scores[query_indices, video_indices] = pair_scores
        dense_scores_unnormalized[query_indices, video_indices] = pair_scores_unnormalized
        return dense_scores, dense_scores_unnormalized

class MTM_Net(nn.Module):
    def __init__(self, config):
        super(MTM_Net, self).__init__()
        self.config = config
        self.num_mamba_layers = config.num_mamba_layers 
        self.query_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l,
            hidden_size=config.hidden_size,
            dropout=config.input_drop
        )
        self.clip_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.map_size, 
            hidden_size=config.hidden_size,
            dropout=config.input_drop
        )
        self.frame_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l, 
            hidden_size=config.hidden_size,
            dropout=config.input_drop
        )
        self.video_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l, 
            hidden_size=config.hidden_size,
            dropout=config.input_drop
        )
        self.query_input_proj = LinearLayer(
            config.q_feat_size, config.hidden_size,
            layer_norm=True, dropout=config.input_drop, relu=True
        )
        self.query_encoder = BertAttention(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop
        ))

        # Clip-level branch.
        self.clip_input_proj = LinearLayer(
            config.visual_feat_dim, config.hidden_size, 
            layer_norm=True, dropout=config.input_drop, relu=True
        )
        self.clip_encoder = DyGMMBlock(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop,
            sft_factor=config.sft_factor,
            initializer_range=config.initializer_range,
            gmm_widths=getattr(config, 'gmm_widths', None),
            map_size=config.map_size
        ))
        clip_mamba_layers = []
        for _ in range(self.num_mamba_layers):
            clip_mamba_layers.append(Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=2, use_fast_path=False))
            clip_mamba_layers.append(nn.LayerNorm(config.hidden_size, eps=1e-5))
        self.clip_mamba = nn.ModuleList(clip_mamba_layers)
        
        # Frame-level branch.
        self.frame_input_proj = LinearLayer(
            config.visual_feat_dim, config.hidden_size, 
            layer_norm=True, dropout=config.input_drop, relu=True
        )
        self.frame_tcn = TCN(
            hidden_size=config.hidden_size,
            dropout=0.1
        )
        self.frame_encoder_1 = DyGMMBlock(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop,
            sft_factor=config.sft_factor,
            initializer_range=config.initializer_range,
            gmm_widths=getattr(config, 'gmm_widths', None),
            map_size=config.max_ctx_l
        ))
        frame_mamba_layers = []
        for _ in range(self.num_mamba_layers):
            frame_mamba_layers.append(Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=2, use_fast_path=False))
            frame_mamba_layers.append(nn.LayerNorm(config.hidden_size, eps=1e-5))
        self.frame_mamba = nn.ModuleList(frame_mamba_layers)

        # Video-level branch.
        self.video_input_proj = LinearLayer(
            config.visual_feat_dim, config.hidden_size, 
            layer_norm=True, dropout=config.input_drop, relu=True
        )
        self.video_encoder = DyGMMBlock(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop,
            sft_factor=config.sft_factor,
            initializer_range=config.initializer_range,
            gmm_widths=getattr(config, 'gmm_widths', None),
            map_size=config.max_ctx_l
        ))
        self.video_pooling = AttentionPooling(config.hidden_size)
        self.video_guided_text_refinement = VideoGuidedTextRefiner(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size, 
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads, 
            attention_probs_dropout_prob=config.drop,
            drop=config.drop,
            safe_vgtr_refine_type=getattr(config, 'safe_vgtr_refine_type', 'paper'),
            safe_vgtr_gate_init=getattr(config, 'safe_vgtr_gate_init', -5.0),
            safe_vgtr_residual_scale=getattr(config, 'safe_vgtr_residual_scale', 1.0),
            safe_vgtr_late_weight=getattr(config, 'safe_vgtr_late_weight', 0.0)
        ))

        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.weight_token = nn.Parameter(torch.randn(1, 1, config.hidden_size)) 
        self.text_reconstructor = MLPReconstructor(
            input_dim=config.hidden_size,
            output_dim=config.max_desc_l * config.q_feat_size,
            num_layers=2, 
            hidden_dim_factor=4,
            dropout=config.drop
        )
        self.video_reconstructor = MLPReconstructor(
            input_dim=config.hidden_size,
            output_dim=config.max_ctx_l * config.visual_feat_dim,
            num_layers=2, 
            hidden_dim_factor=4,
            dropout=config.drop
        )
        self.reset_parameters()

    def reset_parameters(self):
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.apply(re_init)
        if hasattr(self.video_guided_text_refinement, 'reset_safe_parameters'):
            self.video_guided_text_refinement.reset_safe_parameters()

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """Update hard-negative mining settings."""
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def forward(self, batch):
        clip_video_feat = batch['clip_video_features']      
        query_feat      = batch['text_feat']                 
        query_mask      = batch['text_mask']               
        query_labels    = batch['text_labels']
        frame_video_feat = batch['frame_video_features']     
        frame_video_mask = batch['videos_mask']             
        encoded_frame_feat, vid_proposal_feat, encoded_video_feat_pooled, encoded_video_sequence = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask, return_video_sequence=True
        )
        encoded_query_tokens = self.encode_query_tokens(query_feat, query_mask)
        video_query = self.get_modularized_queries(encoded_query_tokens, query_mask)
        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, video_scale_scores, video_scale_scores_ = \
            self.get_pred_from_raw_query(
                query_feat, query_mask, query_labels,
                vid_proposal_feat, encoded_frame_feat, encoded_video_feat_pooled,
                encoded_video_sequence=encoded_video_sequence,
                encoded_video_mask=frame_video_mask,
                encoded_frame_mask=frame_video_mask,
                precomputed_query=video_query,
                precomputed_query_tokens=encoded_query_tokens,
                return_query_feats=True
            )

        label_dict = {}
        for idx, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(idx)
            else:
                label_dict[label] = [idx]

        total_sim = []
        for vid_idx, idx_list in label_dict.items():
            temp_clip_emb = vid_proposal_feat[vid_idx] 
            temp_text_emb = video_query[idx_list]      
            if temp_text_emb.shape[0] == 1: 
                continue
            sim = -1.0 * torch.matmul(
                F.normalize(temp_text_emb, dim=-1),
                F.normalize(temp_clip_emb, dim=-1).t()
            )
            sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)
            indices = linear_sum_assignment(sim.detach().cpu())
            q_idx, c_idx = indices
            for i in range(q_idx.shape[0]):
                total_sim.append(sim[q_idx[i], c_idx[i]])
        if len(total_sim) > 0:
            total_sim = 1 + torch.stack(total_sim).mean() 
        else:
            total_sim = torch.tensor(1.0, device=clip_video_feat.device) 
        reconstructed_text_flat = self.text_reconstructor(video_query)
        reconstructed_text_feat = reconstructed_text_flat.view(
            video_query.shape[0], self.config.max_desc_l, self.config.q_feat_size)
        reconstructed_video_flat = self.video_reconstructor(encoded_video_feat_pooled)
        reconstructed_video_feat = reconstructed_video_flat.view(
            encoded_video_feat_pooled.shape[0], self.config.max_ctx_l, self.config.visual_feat_dim)
        return [
            clip_scale_scores, clip_scale_scores_,
            label_dict, frame_scale_scores, frame_scale_scores_,
            video_scale_scores, video_scale_scores_, 
            video_query, total_sim,
            reconstructed_text_feat,  
            reconstructed_video_feat 
        ]

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_query_tokens(query_feat, query_mask)
        video_query = self.get_modularized_queries(encoded_query, query_mask)
        return video_query

    def encode_query_tokens(self, query_feat, query_mask):
        return self.encode_input(
            query_feat, query_mask,
            self.query_input_proj, self.query_encoder,
            self.query_pos_embed
        )

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None, return_video_sequence=False):
        clip_feat = self.clip_input_proj(clip_video_feat)
        clip_feat = self.clip_pos_embed(clip_feat)
        encoded_clip_feat = self.clip_encoder(clip_feat, None, self.weight_token)
        encoded_clip_feat_m = encoded_clip_feat
        for i in range(0, len(self.clip_mamba), 2): 
            mamba_block = self.clip_mamba[i]
            layer_norm = self.clip_mamba[i+1]
            encoded_clip_feat_m = mamba_block(encoded_clip_feat_m)
            encoded_clip_feat_m = layer_norm(encoded_clip_feat_m)
        encoded_clip_feat = encoded_clip_feat + encoded_clip_feat_m 

        frame_feat = self.frame_input_proj(frame_video_feat)
        frame_feat = self.frame_pos_embed(frame_feat)
        frame_feat = self.frame_tcn(frame_feat)
        attn_mask = video_mask.unsqueeze(1) if video_mask is not None else None
        encoded_frame_feat = self.frame_encoder_1(frame_feat, attn_mask, self.weight_token)
        encoded_frame_feat = self.apply_sequence_mask(encoded_frame_feat, video_mask)
        encoded_frame_feat_m = encoded_frame_feat
        for i in range(0, len(self.frame_mamba), 2): 
            mamba_block = self.frame_mamba[i]
            layer_norm = self.frame_mamba[i+1]
            encoded_frame_feat_m = mamba_block(encoded_frame_feat_m)
            encoded_frame_feat_m = layer_norm(encoded_frame_feat_m)
        encoded_frame_feat = encoded_frame_feat + encoded_frame_feat_m 
        encoded_frame_feat = self.apply_sequence_mask(encoded_frame_feat, video_mask)

        video_feat_raw = self.video_input_proj(frame_video_feat)
        video_feat_raw = self.video_pos_embed(video_feat_raw)
        encoded_video_sequence = self.video_encoder(video_feat_raw, attn_mask, self.weight_token)
        encoded_video_sequence = self.apply_sequence_mask(encoded_video_sequence, video_mask)
        encoded_video_feat_pooled = self.video_pooling(encoded_video_sequence, video_mask)
        if return_video_sequence:
            return encoded_frame_feat, encoded_clip_feat, encoded_video_feat_pooled, encoded_video_sequence
        return encoded_frame_feat, encoded_clip_feat, encoded_video_feat_pooled

    @staticmethod
    def apply_sequence_mask(sequence_feat, sequence_mask):
        if sequence_mask is None:
            return sequence_feat
        return sequence_feat * sequence_mask.to(sequence_feat.dtype).unsqueeze(-1)

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, weight_token=None):
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token)
        else:
            return encoder_layer(feat, mask)
        
    def get_modularized_queries(self, encoded_query, query_mask):
        modular_attention_scores = self.modular_vector_mapping(encoded_query) 
        modular_attention_scores = F.softmax(
            mask_logits(modular_attention_scores, query_mask.unsqueeze(2)),
            dim=1
        )  
        modular_queries = torch.einsum(
            "blm,bld->bmd",
            modular_attention_scores,
            encoded_query
        )
        return modular_queries.squeeze(1) 

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat, context_mask=None):
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)
        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        if context_mask is not None:
            score_mask = context_mask.to(torch.bool).t().unsqueeze(0)
            clip_level_query_context_scores = clip_level_query_context_scores.masked_fill(
                ~score_mask, -float('inf')
            )
        query_context_scores, _ = torch.max(clip_level_query_context_scores, dim=1)
        return query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat, context_mask=None):
        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        if context_mask is not None:
            score_mask = context_mask.to(torch.bool).t().unsqueeze(0)
            query_context_scores = query_context_scores.masked_fill(~score_mask, -float('inf'))
        output_query_context_scores, _ = torch.max(query_context_scores, dim=1)
        return output_query_context_scores

    def get_pred_from_raw_query(
            self, query_feat, query_mask, query_labels=None,
            video_proposal_feat=None, encoded_frame_feat=None, encoded_video_feat_raw=None,
            encoded_video_sequence=None, encoded_video_mask=None,
            encoded_frame_mask=None, precomputed_query=None, precomputed_query_tokens=None,
            video_candidate_indices=None,
            return_query_feats=False
    ):
        query_token_feat = precomputed_query_tokens
        if precomputed_query is None or query_token_feat is None:
            query_token_feat = self.encode_query_tokens(query_feat, query_mask)
        if precomputed_query is None:
            video_query = self.get_modularized_queries(query_token_feat, query_mask)
        else:
            video_query = precomputed_query
        clip_scale_scores = self.get_clip_scale_scores(video_query, video_proposal_feat)
        frame_scale_scores = self.get_clip_scale_scores(video_query, encoded_frame_feat, encoded_frame_mask)
        video_scale_scores_global = self.get_video_scale_scores_from_guided_text(video_query, encoded_video_feat_raw)
        video_scale_scores_unnormalized_global = self.get_unnormalized_video_scale_scores_from_guided_text(
            video_query, encoded_video_feat_raw)

        if encoded_video_sequence is None:
            video_scale_scores = video_scale_scores_global
            video_scale_scores_unnormalized = video_scale_scores_unnormalized_global
        else:
            candidate_indices = video_candidate_indices
            top_k = int(getattr(self.config, 'video_topk', 0))
            if candidate_indices is None and (not self.training) and top_k > 0 and top_k < video_scale_scores_global.shape[1]:
                base_scores = (
                    float(getattr(self.config, 'clip_scale_w', 1.0)) * clip_scale_scores.detach() +
                    float(getattr(self.config, 'frame_scale_w', 1.0)) * frame_scale_scores.detach()
                )
                candidate_indices = torch.topk(base_scores, k=top_k, dim=1).indices
            video_scale_scores, video_scale_scores_unnormalized = self.video_guided_text_refinement.score_safe_pairs(
                video_query,
                query_token_feat,
                query_mask,
                encoded_video_feat_raw,
                video_sequence_feat=encoded_video_sequence,
                video_mask=encoded_video_mask,
                candidate_indices=candidate_indices,
                chunk_size=int(getattr(self.config, 'video_pair_chunk_size', 512)),
                fallback_scores=(video_scale_scores_global, video_scale_scores_unnormalized_global)
            )

        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = self.get_unnormalized_clip_scale_scores(
                video_query, encoded_frame_feat, encoded_frame_mask
            )
            return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, video_scale_scores, video_scale_scores_unnormalized
        else:
            return clip_scale_scores, frame_scale_scores, video_scale_scores

    def get_video_scale_scores_from_guided_text(self, refined_text_feat, video_feat):
        refined_text_feat_norm = F.normalize(refined_text_feat, dim=-1)
        video_feat_norm = F.normalize(video_feat, dim=-1)
        scores = torch.matmul(refined_text_feat_norm, video_feat_norm.t()) 
        return scores

    def get_unnormalized_video_scale_scores_from_guided_text(self, refined_text_feat, video_feat):
        scores = torch.matmul(refined_text_feat, video_feat.t()) 
        return scores

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
