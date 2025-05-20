# backend/source/ai/FirstETAModel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstETAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== Embedding Tables =====
        self.bus_emb = nn.Embedding(200, 8)         # bus_number
        self.dir_raw_emb = nn.Embedding(2, 4)        # direction raw
        self.branch_raw_emb = nn.Embedding(6, 4)    # branch_num raw

        self.node_emb = nn.Embedding(3000, 8)         # node_id

        self.pty_emb = nn.Embedding(5, 3)             # PTY (강수 형태)

        self.weekday_index_emb = nn.Embedding(3, 4)
        self.timegroup_index_emb = nn.Embedding(8, 8)
        self.wd_tg_index_emb = nn.Embedding(24, 12)

        self.weekday_emb = self.weekday_index_emb
        self.timegroup_emb = self.timegroup_index_emb
        self.wd_tg_emb = self.wd_tg_index_emb

        # ===== Direction / Branch conditional MLPs =====
        self.dir_cond = nn.Sequential(nn.Linear(8, 4))
        self.branch_cond = nn.Sequential(nn.Linear(12, 4))

        # ===== ORD ratio =====
        self.ord_ratio_mlp = nn.Sequential(nn.Linear(1, 4), nn.ReLU())

        # ===== Mean Elapsed Mini MLPs =====
        self.mean_elapsed_total_mlp = nn.Sequential(nn.Linear(1, 2), nn.ReLU())
        self.mean_elapsed_wd_mlp = nn.Sequential(nn.Linear(1 + 4, 4), nn.ReLU())
        self.mean_elapsed_tg_mlp = nn.Sequential(nn.Linear(1 + 8, 4), nn.ReLU())
        self.mean_elapsed_wdtg_mlp = nn.Sequential(nn.Linear(1 + 12, 6), nn.ReLU())
        self.mean_elapsed_merge = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

        # ===== Route-ORD Context MLP =====
        self.route_ord_mlp = nn.Sequential(nn.Linear(32, 32), nn.ReLU())

        # ===== Mean Interval Mini MLPs =====
        self.mean_interval_total_mlp = nn.Sequential(nn.Linear(1, 2), nn.ReLU())
        self.mean_interval_wd_mlp = nn.Sequential(nn.Linear(1 + 4, 4), nn.ReLU())
        self.mean_interval_tg_mlp = nn.Sequential(nn.Linear(1 + 8, 4), nn.ReLU())
        self.mean_interval_wdtg_mlp = nn.Sequential(nn.Linear(1 + 12, 6), nn.ReLU())
        self.mean_interval_merge = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

        self.node_context_mlp = nn.Sequential(nn.Linear(24, 16), nn.ReLU())

        # ===== Time Context =====
        self.time_mlp = nn.Sequential(nn.Linear(26, 16), nn.ReLU())

        # ===== Weather Context =====
        self.weather_mlp = nn.Sequential(nn.Linear(5, 8), nn.ReLU())

        # ===== Prev Predicted Elapsed =====
        self.prev_pred_mlp = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 16), nn.ReLU())

        # ===== Final MLP =====
        self.final_mlp = nn.Sequential(
            nn.Linear(88, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.head_mean = nn.Linear(32, 1)
        self.head_logvar = nn.Linear(32, 1)

    def forward(self, x, phase="self_review"):
        # === Route-related ===
        bus = self.bus_emb(x['bus_number'])                      # (B, 8)
        dir_raw = self.dir_raw_emb(x['direction'])              # (B, 4)
        dir_adj = self.dir_cond(bus)                            # (B, 4)
        direction = dir_raw + dir_adj

        branch_raw = self.branch_raw_emb(x['branch_num'])       # (B, 4)
        branch_input = torch.cat([bus, direction], dim=1)       # (B, 12)
        branch_adj = self.branch_cond(branch_input)
        branch = branch_raw + branch_adj

        ord_ratio = self.ord_ratio_mlp(x['ord_ratio'])          # (B, 4)

        # === Mean Elapsed ===
        me_total = self.mean_elapsed_total_mlp(x['mean_elapsed_total'])
        me_wd = self.mean_elapsed_wd_mlp(torch.cat([x['mean_elapsed_wd'], self.weekday_index_emb(x['weekday'])], dim=1))
        me_tg = self.mean_elapsed_tg_mlp(torch.cat([x['mean_elapsed_tg'], self.timegroup_index_emb(x['timegroup'])], dim=1))
        me_wdtg = self.mean_elapsed_wdtg_mlp(torch.cat([x['mean_elapsed_wdtg'], self.wd_tg_index_emb(x['wd_tg'])], dim=1))
        mean_elapsed = self.mean_elapsed_merge(torch.cat([me_total, me_wd, me_tg, me_wdtg], dim=1))

        route_input = torch.cat([bus, direction, branch, ord_ratio, mean_elapsed], dim=1)
        route_context = self.route_ord_mlp(route_input)         # (B, 32)

        # === Node Context ===
        node = self.node_emb(x['node_id'])                      # (B, 8)
        mi_total = self.mean_interval_total_mlp(x['mean_interval_total'])
        mi_wd = self.mean_interval_wd_mlp(torch.cat([x['mean_interval_wd'], self.weekday_index_emb(x['weekday'])], dim=1))
        mi_tg = self.mean_interval_tg_mlp(torch.cat([x['mean_interval_tg'], self.timegroup_index_emb(x['timegroup'])], dim=1))
        mi_wdtg = self.mean_interval_wdtg_mlp(torch.cat([x['mean_interval_wdtg'], self.wd_tg_index_emb(x['wd_tg'])], dim=1))
        mean_interval = self.mean_interval_merge(torch.cat([mi_total, mi_wd, mi_tg, mi_wdtg], dim=1))

        node_context = self.node_context_mlp(torch.cat([node, mean_interval], dim=1))  # (B, 16)

        # === Time Context ===
        wd_emb = self.weekday_emb(x['weekday'])
        tg_emb = self.timegroup_emb(x['timegroup'])
        wdtg_emb = self.wd_tg_emb(x['wd_tg'])
        time_context = self.time_mlp(torch.cat([wd_emb, tg_emb, wdtg_emb, x['time_sin_cos']], dim=1))  # (B, 16)

        # === Weather Context ===
        pty = self.pty_emb(x['PTY'])
        weather_context = self.weather_mlp(torch.cat([pty, x['RN1'], x['T1H']], dim=1))

        # === Self Review 용 Prev ETA ===
        if phase == 'replay':
            full_input = torch.cat([route_context, node_context, time_context, weather_context], dim=1)  # (B, 72)
        else:
            prev_eta = self.prev_pred_mlp(x['prev_pred_elapsed'])  # (B, 16)
            full_input = torch.cat([route_context, node_context, time_context, weather_context, prev_eta], dim=1)  # (B, 88)

        h = self.final_mlp(full_input)
        pred_mean = self.head_mean(h)
        pred_log_var = self.head_logvar(h)
        return pred_mean, pred_log_var