# backend/source/ai/inferenceSecondModel.py

import torch
from SecondETAModel import SecondETAModel

# how to call inference:
# model = load_second_eta_model("/your/path/model.pth", device)
# results = infer_eta_batch(model, [row1, row2, ...], device)

FLOAT_KEYS = [
    "departure_time_sin", "departure_time_cos",
    *[f"weather_{i}_{k}" for i in range(1, 6) for k in ["RN1", "T1H"]],
    *[f"mean_interval_{i}_{k}" for i in range(1, 6) for k in ["total", "weekday", "timegroup", "weekday_timegroup"]],
    *[f"avg_{i}_congestion" for i in range(1, 6)],
    *[f"ord_{i}_ratio" for i in range(1, 6)],
    *[f"prev_pred_elapsed_{i}" for i in range(1, 6)],
    "node_id_ratio",
]
LONG_KEYS = [
    "bus_number", "direction", "branch", "weekday", "timegroup", "weekday_timegroup",
    *[f"weather_{i}_PTY" for i in range(1, 6)],
]

def load_second_eta_model(model_path: str, device: torch.device) -> SecondETAModel:
    model = SecondETAModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def prepare_input_tensor(x_row: dict, device: torch.device) -> dict:
    tensor = {}
    for k, v in x_row.items():
        if isinstance(v, torch.Tensor):
            tensor[k] = v.clone().detach().unsqueeze(0).to(device)
        elif k in FLOAT_KEYS:
            tensor[k] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        elif k in LONG_KEYS:
            tensor[k] = torch.tensor(v, dtype=torch.long, device=device).unsqueeze(0)
        else:
            raise KeyError(f"[infer] Unknown feature key: {k}")
    return tensor

def infer_eta_batch(model: SecondETAModel, x_rows: list[dict], device: torch.device) -> list[list[float]]:
    results = []
    with torch.no_grad():
        for row in x_rows:
            x_tensor = prepare_input_tensor(row, device)
            pred_mean, _ = model(x_tensor)
            results.append(pred_mean.squeeze(0).tolist())
    return results