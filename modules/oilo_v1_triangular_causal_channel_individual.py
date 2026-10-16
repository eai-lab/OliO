import torch
import torch.nn as nn
import torch.nn.functional as F

class TriangularCausalShift(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.pos_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.raw_K = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_model))
        
    def get_k_matrix(self):
        return torch.triu(self.raw_K, diagonal=1)
    
    def forward(self, z_t, z_tau, tau):
        p_t  = z_t + self.pos_embed
        p_tau = z_tau + self.pos_embed
        
        K = self.get_k_matrix()
        weight_fwd = torch.matrix_exp(K * tau)
        p_tau_pred = F.linear(p_t, weight_fwd, self.bias * tau)
        
        weight_bwd = torch.matrix_exp(K * -tau)
        p_t_pred = F.linear(p_tau, weight_bwd, self.bias * (-tau))
        
        return {
            'z_t': p_t,
            'z_tau': p_tau,
            'z_t_pred': p_t_pred,
            'z_tau_pred': p_tau_pred
        }

class ProjectionHead(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)

class OilO_Wrapper(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.n_vars = args.enc_in

        assert self.args.module_version == 'v1-triangular-causal-channel-individual'

        self.projector = nn.ModuleList([
            ProjectionHead(
                d_in=args.d_model, 
                d_hidden=args.d_ff, 
                d_out=args.projection_dim,
            ) for _ in range(self.n_vars)
        ])
        self.version = args.module_version
        
        self.S_module = nn.ModuleList([
            TriangularCausalShift(args.projection_dim) for _ in range(self.n_vars)
        ])

    def forward(self, x_t, x_tau, tau, x_t_mark=None, x_tau_mark=None, *args, **kwargs):
        # 1. Backbone Forward (Representation Learning: f)
        # x_t: (B, L, N)
        x_combined = torch.cat([x_t, x_tau], dim=0)
        x_mark_combined = None
        if x_t_mark is not None and x_tau_mark is not None:
            x_mark_combined = torch.cat([x_t_mark, x_tau_mark], dim=0)
        h_combined = self.backbone(
            batch_x=x_combined,
            batch_x_mark=x_mark_combined,
            dec_inp=None,
            batch_y_mark=None,
            return_latent=True, 
        ) # h = f(x)
        h_t, h_tau = torch.chunk(h_combined, 2, dim=0)
        
        # 2. Flatten & Projection (g)
        # h_t shape: [B, nvars, d_model]
        B, nvars, D = h_t.shape
        
        # Projector 통과: z = g(h)
        z_t = torch.stack([self.projector[i](h_t[:, i, :]) for i in range(nvars)], dim=1)   # [B, nvars, projection_dim]
        z_tau = torch.stack([self.projector[i](h_tau[:, i, :]) for i in range(nvars)], dim=1)   # [B, nvars, projection_dim]
        
        outputs = {'z_t': [], 'z_tau': [], 'z_t_pred': [], 'z_tau_pred': []}
        for i in range(nvars):
            out_i = self.S_module[i](z_t[:, i, :], z_tau[:, i, :], tau)
            outputs['z_t'].append(out_i['z_t'])
            outputs['z_tau'].append(out_i['z_tau'])
            outputs['z_t_pred'].append(out_i['z_t_pred'])
            outputs['z_tau_pred'].append(out_i['z_tau_pred'])
        
        outputs = {k: torch.stack(v, dim=1) for k, v in outputs.items()}
        return outputs
        
    def loss_fn(self, outputs):
        p_tau_pred = outputs['z_tau_pred']
        p_tau_target = outputs['z_tau'].detach()
        p_t_pred = outputs['z_t_pred']
        p_t_target = outputs['z_t'].detach()
        
        loss_fwd = 1 - F.cosine_similarity(p_tau_pred, p_tau_target, dim=-1).mean()
        loss_bwd = 1 - F.cosine_similarity(p_t_pred, p_t_target, dim=-1).mean()
        loss = (loss_fwd + loss_bwd) / 2.0
        return {
            'loss': loss,
            'loss_fwd': loss_fwd,
            'loss_bwd': loss_bwd
        }