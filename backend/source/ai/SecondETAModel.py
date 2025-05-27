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

        self.node_emb = nn.Embedding(3000, 8)       # node

        self.pty_emb = nn.Embedding(10, 3)          # PTY

        self.weekday_emb = nn.Embedding(5, 4)       # weekday
        self.timegroup_emb = nn.Embedding(10, 8)    # timegroupd
        self.wd_tg_emb = nn.Embedding(40, 12)       # weekday_timegroup

        # ===== Conditional MLPs =====
        self.dir_cond = nn.Sequential(nn.Linear(8, 12))
        

        # ===== ORD ratio =====
        self.node_id_ratio_mlp = nn.Sequential(nn.Linear(1, 20), nn.ReLU())
        self.next_node_id_ratio_mlp = nn.Sequential(nn.Linear(1, 24), nn.ReLU())

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
        

        # ===== Prev Predicted Elapsed =====
        

        # ===== Final MLP =====
        

    def forward(self, x):
        # === Route-related ===
        
        # === Mean Elapsed ===
        
        # === Prev Mean Elapsed ===
        
        # === Node Context ===
        
        # === Time Context ===
        
        # === Weather Context ===
        
        # === Self Review 용 Prev ETA ===
        
        return