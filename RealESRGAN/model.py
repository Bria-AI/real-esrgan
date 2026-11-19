import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_download

from .rrdbnet_arch import RRDBNet
from .utils import padded_index


HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=scale
        )
        
    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            assert self.scale in [2,4,8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            cache_dir = os.path.dirname(model_path)
            local_filename = os.path.basename(model_path)
            
            # Use hf_hub_download directly instead of hf_hub_url + cached_download
            model_path = hf_hub_download(
                repo_id=config['repo_id'],
                filename=config['filename'],
                cache_dir=cache_dir,
                force_filename=local_filename
            )
            print('Weights downloaded to:', model_path)
        
        loadnet = torch.load(model_path)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()
        self.model.to(self.device)
        
    @torch.amp.autocast('cuda')
    def predict(self, lr_image: np.ndarray, batch_size=4, patches_size=192,
                padding=24, pad_size=15) -> np.ndarray:
        """Pads, tiles, stitches and quantises on self.device; output identical to the previous numpy version."""
        scale, device = self.scale, self.device
        out_dtype = lr_image.dtype
        div_factor = 255 if out_dtype == np.uint8 else 65535
        if lr_image.shape[2] == 4:
            lr_image = lr_image[:, :, :3]

        # int32 holds uint16 values exactly; avoids uint16 torch tensors
        img = torch.from_numpy(
            np.ascontiguousarray(lr_image, dtype=np.uint8 if out_dtype == np.uint8 else np.int32)
        ).to(device)
        h, w = img.shape[:2]
        ri = padded_index(h, pad_size, patches_size, padding, device)
        ci = padded_index(w, pad_size, patches_size, padding, device)
        padded = img[ri][:, ci].permute(2, 0, 1)
        tile = patches_size + 2 * padding
        patches = padded.unfold(1, tile, patches_size).unfold(2, tile, patches_size)
        n_rows, n_cols = patches.shape[1], patches.shape[2]
        patches = patches.permute(1, 2, 0, 3, 4).reshape(n_rows * n_cols, 3, tile, tile)

        # float64 divide then float32 matches numpy's patches/div_factor -> FloatTensor
        q_dtype = torch.uint8 if out_dtype == np.uint8 else torch.int32
        quantised = []
        with torch.no_grad():
            for i in range(0, patches.shape[0], batch_size):
                # clone (not contiguous) gives the old channels-last strides, incl. for a size-1
                # batch, so torch conv backends round identically
                batch = (patches[i : i + batch_size].double() / div_factor).float()
                batch = batch.clone(memory_format=torch.channels_last)
                res = self.model(batch).clamp_(0, 1)
                quantised.append((res.double() * div_factor).to(q_dtype))
        tiles = torch.cat(quantised, 0)

        crop = padding * scale
        tiles = tiles[:, :, crop:-crop, crop:-crop]
        q = tiles.shape[-1]
        out = tiles.reshape(n_rows, n_cols, 3, q, q).permute(0, 3, 1, 4, 2)
        out = out.reshape(n_rows * q, n_cols * q, 3)
        n_h = (h + 2 * pad_size) * scale
        n_w = (w + 2 * pad_size) * scale
        p = pad_size * scale
        out = out[:n_h, :n_w][p:-p, p:-p].contiguous().cpu().numpy()
        return out.astype(out_dtype, copy=False)
