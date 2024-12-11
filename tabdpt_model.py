import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import maskmean, maskstd, normalize_data, clip_outliers, seed_everything

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        bias = True  # Set bias=True to match the original model
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x, eval_pos):
        B, L, _ = x.size()
        h = self.attn_norm(x)
        q = self.q_proj(h)
        k, v = self.kv_proj(h[:, :eval_pos]).chunk(2, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        attn = self.out_proj(attn.reshape(B, L, self.embed_dim))
        x = x + attn
        x = x + self.ff(self.ff_norm(x))
        return x

class TabDPTModel(nn.Module):
    def __init__(self, dropout: float, n_out: int, nhead: int, nhid: int, ninp: int, nlayers: int, norm_first: bool, num_features: int):
        super().__init__()
        self.n_out = n_out
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.y_encoder = nn.Linear(1, ninp)
        self.cls_head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.reg_head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, 1))
        self.task2head = {'cls': self.cls_head, 'reg': self.reg_head}
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim=ninp, num_heads=nhead, ff_dim=nhid)
                for _ in range(nlayers)
            ]
        )

    @torch.no_grad()
    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        eval_pos = y_src.shape[0]

        x_src = normalize_data(x_src, -1 if self.training else eval_pos)
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=10)
        x_src = torch.nan_to_num(x_src, nan=0)

        x_src = self.encoder(x_src)

        mean = (x_src**2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean)
        x_src = x_src / rms

        y_src = self.y_encoder(y_src.unsqueeze(-1))
        train_x = x_src[:eval_pos] + y_src
        src = torch.cat([train_x, x_src[eval_pos:]], 0)

        src = src.transpose(0, 1) # (B, L, d) shape
        for layer in self.transformer_encoder:
            src = layer(src, eval_pos)
        pred = self.task2head[task](src)

        return pred[:, eval_pos:].transpose(0, 1) # back to (L, B, d) shape

    @classmethod
    def load(cls, model_state, config):
        model = TabDPTModel(
            dropout=config['training']['dropout'],
            n_out=config['model']['max_num_classes'],
            nhead=config['model']['nhead'],
            nhid=config['model']['emsize'] * config['model']['nhid_factor'],
            ninp=config['model']['emsize'],
            nlayers=config['model']['nlayers'],
            norm_first=config['model']['norm_first'],
            num_features=config['model']['max_num_features'],
        )

        # Remove any module prefixes if necessary
        module_prefix = '_orig_mod.'
        model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}

        # Mapping function to convert state_dict keys
        def map_state_dict(original_state_dict, model):
            new_state_dict = {}
            for key, value in original_state_dict.items():
                if key.startswith('transformer_encoder.'):
                    # Handle transformer encoder layers
                    parts = key.split('.')
                    layer_idx = parts[1]
                    sub_module = parts[2]
                    param_name = '.'.join(parts[3:])
                    if sub_module == 'self_attn':
                        if param_name == 'in_proj_weight':
                            in_proj_weight = value
                            embed_dim = model.transformer_encoder[int(layer_idx)].embed_dim
                            q_proj_weight = in_proj_weight[:embed_dim, :]
                            k_proj_weight = in_proj_weight[embed_dim:2*embed_dim, :]
                            v_proj_weight = in_proj_weight[2*embed_dim:, :]
                            kv_proj_weight = torch.cat([k_proj_weight, v_proj_weight], dim=0)
                            new_state_dict[f'transformer_encoder.{layer_idx}.q_proj.weight'] = q_proj_weight
                            new_state_dict[f'transformer_encoder.{layer_idx}.kv_proj.weight'] = kv_proj_weight
                        elif param_name == 'in_proj_bias':
                            in_proj_bias = value
                            embed_dim = model.transformer_encoder[int(layer_idx)].embed_dim
                            q_proj_bias = in_proj_bias[:embed_dim]
                            k_proj_bias = in_proj_bias[embed_dim:2*embed_dim]
                            v_proj_bias = in_proj_bias[2*embed_dim:]
                            kv_proj_bias = torch.cat([k_proj_bias, v_proj_bias], dim=0)
                            new_state_dict[f'transformer_encoder.{layer_idx}.q_proj.bias'] = q_proj_bias
                            new_state_dict[f'transformer_encoder.{layer_idx}.kv_proj.bias'] = kv_proj_bias
                        elif param_name == 'out_proj.weight':
                            new_state_dict[f'transformer_encoder.{layer_idx}.out_proj.weight'] = value
                        elif param_name == 'out_proj.bias':
                            new_state_dict[f'transformer_encoder.{layer_idx}.out_proj.bias'] = value
                    elif sub_module == 'linear1':
                        if param_name == 'weight':
                            new_state_dict[f'transformer_encoder.{layer_idx}.ff.0.weight'] = value
                        elif param_name == 'bias':
                            new_state_dict[f'transformer_encoder.{layer_idx}.ff.0.bias'] = value
                    elif sub_module == 'linear2':
                        if param_name == 'weight':
                            new_state_dict[f'transformer_encoder.{layer_idx}.ff.2.weight'] = value
                        elif param_name == 'bias':
                            new_state_dict[f'transformer_encoder.{layer_idx}.ff.2.bias'] = value
                    elif sub_module == 'norm1':
                        if param_name == 'weight':
                            new_state_dict[f'transformer_encoder.{layer_idx}.attn_norm.weight'] = value
                        elif param_name == 'bias':
                            new_state_dict[f'transformer_encoder.{layer_idx}.attn_norm.bias'] = value
                    elif sub_module == 'norm2':
                        if param_name == 'weight':
                            new_state_dict[f'transformer_encoder.{layer_idx}.ff_norm.weight'] = value
                        elif param_name == 'bias':
                            new_state_dict[f'transformer_encoder.{layer_idx}.ff_norm.bias'] = value
                else:
                    # Copy other parameters directly
                    new_state_dict[key] = value
            return new_state_dict

        # Map the state_dict to the new model
        new_state_dict = map_state_dict(model_state, model)
        model.load_state_dict(new_state_dict)
        model.to(config['env']['device'])
        model.eval()
        return model
