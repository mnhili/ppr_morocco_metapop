import torch

def print_pt(path):
    print(f"--- {path} ---")
    try:
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.numel() > 10:
                    print(f"{k}: Tensor of shape {v.shape}")
                else:
                    print(f"{k}: {v}")
        else:
            print(data)
    except Exception as e:
        print(f"Error: {e}")

print_pt('outputs/tarp/tarp_results.pt')
