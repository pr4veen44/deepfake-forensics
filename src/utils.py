from pathlib import Path
import random, numpy as np, torch
VIDEO_EXTS={'.mp4','.avi','.mov','.mkv','.webm'}
CLASSES=['REAL','FAKE']
def project_root(): return Path(__file__).resolve().parents[1]
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
