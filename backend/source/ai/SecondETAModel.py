# backend/source/ai/SecondETAModel.py

import torch
import torch.nn as nn

class SecondETAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== Embedding Tables =====
        self.bus_emb = nn.Embedding(200, 8)         # bus_number
        self.dir_emb = nn.Embedding(3, 12)          # direction
        self.branch_emb = nn.Embedding(10, 16)      # branch

        self.pty_emb = nn.Embedding(10, 3)          # PTY

        self.weekday_emb = nn.Embedding(5, 4)       # weekday
        self.timegroup_emb = nn.Embedding(10, 8)    # timegroup
        self.wd_tg_emb = nn.Embedding(40, 12)       # weekday_timegroup

        # ===== Conditional MLPs =====
        self.dir_cond = nn.Sequential(nn.Linear(8, 12))
        self.branch_cond = nn.Sequential(nn.Linear(12, 16))
        self.node_id_cond = nn.Sequential(nn.Linear(16, 20))
        self.ord_vector_cond = nn.Sequential(nn.Linear(20, 240))

        # ===== ORD ratio =====
        self.node_id_ratio_mlp = nn.Sequential(nn.Linear(1, 20), nn.ReLU())
        self.next_node_id_ratio_mlp = nn.Sequential(nn.Linear(1, 48), nn.ReLU())

        # ===== Mean Interval Mini MLPs =====
        self.mean_interval_total_mlp = nn.Sequential(nn.Linear(1, 2), nn.ReLU())
        self.mean_interval_wd_mlp = nn.Sequential(nn.Linear(1 + 4, 4), nn.ReLU())
        self.mean_interval_tg_mlp = nn.Sequential(nn.Linear(1 + 8, 4), nn.ReLU())
        self.mean_interval_wd_tg_mlp = nn.Sequential(nn.Linear(1 + 12, 6), nn.ReLU())
        self.mean_interval_merge = nn.Sequential(nn.Linear(16, 16), nn.ReLU()) # 32 dim

        # ===== Weather Context =====
        self.weather_mlp = nn.Sequential(nn.Linear(5, 8), nn.ReLU())

        # ===== Average Congestion Mini MLPs =====
        self.avg_cong_mlp = nn.Sequential(nn.Linear(1, 24), nn.ReLU())

        # ===== Time Context =====
        self.time_mlp = nn.Sequential(nn.Linear(26, 16), nn.ReLU())

        # ===== Route Context =====
        self.ord_merge_mlp = nn.Sequential(nn.Linear(48, 48), nn.ReLU())
        self.route_context_mlp = nn.Sequential(nn.Linear(240, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU())

        # ===== Prev Predicted Elapsed =====
        self.prev_pred_mlp = nn.Sequential(nn.Linear(1, 4), nn.ReLU())

        # ===== Final MLP =====
        self.final_mlp = nn.Sequential(nn.Linear(80, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 32), nn.ReLU())
        
        self.head_mean = nn.Sequential(nn.Linear(32, 5), nn.Sigmoid())
        self.head_logvar = nn.Linear(32, 5)

    def forward(self, x):
        # === Route-related ===
        bus = self.bus_emb(x['bus_number'].squeeze(1))     # (B, 8)
        dir_adj = self.dir_cond(bus)            # (B, 12)
        dir_raw = self.dir_emb(x['direction'].squeeze(1))  # (B, 12)
        direction = dir_raw + dir_adj           # dir complete, 12 dim

        branch_adj = self.branch_cond(direction)    # (B, 16)
        branch_raw = self.branch_emb(x['branch'].squeeze(1))   # (B, 16)     
        branch = branch_raw + branch_adj            # branch complete, 16 dim

        node_id_adj = self.node_id_cond(branch)                         # (B, 20)
        node_id_ratio_raw = self.node_id_ratio_mlp(x['node_id_ratio'])          # (B, 20)
        print("node_id_ratio raw shape:", x['node_id_ratio'].shape)
        node_id = node_id_adj + node_id_ratio_raw                       # node_id complete, 20 dim

        print("bus", bus.shape)
        print("direction", direction.shape)
        print("branch", branch.shape)
        print("node_id", node_id.shape)

        # === ORD Context ===
        ord_context_list = []
        print("input keys:", x.keys())
        for i in range(1, 6):
            ord_i_ratio = self.next_node_id_ratio_mlp(x[f'ord_{i}_ratio'])  # 48 dim
            if ord_i_ratio.dim() == 3:
                print("squeezing ord_i_ratio:", ord_i_ratio.shape)
                ord_i_ratio = ord_i_ratio.squeeze(1)
            
            average_congestion = self.avg_cong_mlp(x[f'avg_{i}_congestion'])    # 24 dim
            
            pty = self.pty_emb(x[f'weather_{i}_PTY'].squeeze(1))
            weather_context = self.weather_mlp(torch.cat([pty, x[f'weather_{i}_RN1'], x[f'weather_{i}_T1H']], dim=1))   # 8 dim

            mi_total = self.mean_interval_total_mlp(x[f'mean_interval_{i}_total'])
            mi_wd = self.mean_interval_wd_mlp(torch.cat([x[f'mean_interval_{i}_weekday'], self.weekday_emb(x['weekday'].squeeze(1))], dim=1))
            mi_tg = self.mean_interval_tg_mlp(torch.cat([x[f'mean_interval_{i}_timegroup'], self.timegroup_emb(x['timegroup'].squeeze(1))], dim=1))
            mi_wdtg = self.mean_interval_wd_tg_mlp(torch.cat([x[f'mean_interval_{i}_weekday_timegroup'], self.wd_tg_emb(x['weekday_timegroup'].squeeze(1))], dim=1))
            mean_interval = self.mean_interval_merge(torch.cat([mi_total, mi_wd, mi_tg, mi_wdtg], dim=1))   # 16 dim

            ord_merge = self.ord_merge_mlp(torch.cat([average_congestion, weather_context, mean_interval], dim=1))
            if ord_merge.dim() == 3:
                print("squeezing ord_merge:", ord_merge.shape)
                ord_merge = ord_merge.squeeze(1)

            ord_i_context = ord_i_ratio + ord_merge
            if ord_i_context.dim() == 3:
                print("squeezing ord_i_context:", ord_i_context.shape)
                ord_i_context = ord_i_context.squeeze(1)  # squeeze dim=1
            ord_context_list.append(ord_i_context)
        
        ord_context_adj = self.ord_vector_cond(node_id)
        ord_context_raw = torch.cat(ord_context_list, dim=1)  # => [B, 240] 돼야 함

        print("ord_context_raw", ord_context_raw.shape)
        print("ord_context_adj", ord_context_adj.shape)

        if ord_context_adj.dim() == 3:
            print("squeezing ord_context_adj:", ord_context_adj.shape)
            ord_context_adj = ord_context_adj.squeeze(1)

        if ord_context_raw.dim() == 3:
            print("squeezing ord_context_raw:", ord_context_raw.shape)
            ord_context_raw = ord_context_raw.squeeze(1)

        ord_context = ord_context_adj + ord_context_raw
        route_context = self.route_context_mlp(ord_context)
        if route_context.dim() == 3:
            route_context = route_context.squeeze(1)
        
        print("route_context:", route_context.shape)
        
        # === Time Context ===
        weekday_emb = self.weekday_emb(x['weekday'].squeeze(1))
        timegroup_emb = self.timegroup_emb(x['timegroup'].squeeze(1))
        weekday_timegroup_emb = self.wd_tg_emb(x['weekday_timegroup'].squeeze(1))
        time_context = self.time_mlp(torch.cat([
            weekday_emb, timegroup_emb, weekday_timegroup_emb, 
            x['departure_time_sin'], x['departure_time_cos']
            ], dim=1))  # (B, 16)
    
        print("time_context:", time_context.shape)
        
        # === Self Review 용 Prev ETA ===
        prev_eta_feats = []
        for i in range(1, 6):
            prev_eta_raw = x[f'prev_pred_elapsed_{i}']
            prev_eta_i = self.prev_pred_mlp(prev_eta_raw.view(-1, 1))
            if prev_eta_i.dim() == 3:
                prev_eta_i = prev_eta_i.squeeze(1)
            prev_eta_feats.append(prev_eta_i)

        prev_eta = torch.cat(prev_eta_feats, dim=1)  # → (B, 20)

        print("prev_eta:", prev_eta.shape)

        # === final MLP ===
        full_input = torch.cat([route_context, time_context, prev_eta], dim=1) # dim 80
        h = self.final_mlp(full_input)

        pred_mean = self.head_mean(h) # dim 5
        pred_log_var = self.head_logvar(h) # dim 5
        pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=5.0)
        
        return pred_mean, pred_log_var