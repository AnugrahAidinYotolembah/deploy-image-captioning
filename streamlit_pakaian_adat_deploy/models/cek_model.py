import torch

# 1. Muat checkpoint
checkpoint_path = "/mnt/nas-hpg9/didiyotolembah/tesis_aidin/streamlit_pakaian_adat_deploy/models/model_pakaian_adat_bundle.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 2. Lihat kunci (keys) pada level atas
print("Kunci utama checkpoint:", list(checkpoint.keys()))

# 3. Jika ada sub‑dictionary, tampilkan kunci‑kuncinya juga
for key, value in checkpoint.items():
    if isinstance(value, dict):
        print(f"  {key} ->", list(value.keys()))
    else:
        # Tampilkan tipe dan shape (jika itu tensor)
        if isinstance(value, torch.Tensor):
            print(f"  {key} : {type(value).__name__} dengan shape {value.shape}")
        else:
            print(f"  {key} : {type(value).__name__}")