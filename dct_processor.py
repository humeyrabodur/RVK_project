import torch
import math
import torchvision.transforms as T
from PIL import Image

# --- Constants ---
# Standard JPEG Quantization Table (Luminance)
JPEG_LUM_QUANT = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=torch.float32)

def get_dct_matrix():
    N = 8
    mat = torch.zeros((N, N))
    for u in range(N):
        for x in range(N):
            norm = math.sqrt(1/N) if u == 0 else math.sqrt(2/N)
            mat[u, x] = norm * math.cos(((2*x + 1) * u * math.pi) / (2*N))
    return mat

# Pre-compute DCT matrices
DCT_MAT = get_dct_matrix()
DCT_MAT_T = DCT_MAT.t()

class DCTProcessor:
    def __init__(self, image_size=256, device='cuda'):
        """
        image_size: The spatial resolution (e.g., 256 for 256x256 input).
                    This results in 32x32 DCT blocks.
        """
        self.image_size = image_size
        self.device = device
        
        # Move constants to the correct device
        self.dct_mat = DCT_MAT.to(device)
        self.dct_mat_t = DCT_MAT_T.to(device)
        self.quant_base = JPEG_LUM_QUANT.to(device)

        # Standard Transform for input images
        self.preprocess = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def _get_q_table(self, q_scale):
        # Multiplies base table by q_scale
        q_table = self.quant_base * q_scale
        return torch.clamp(q_table, min=1.0)

    def rgb_to_y_centered(self, img_tensor):
        """Extracts just the Y channel and centers it (-128 to 127)."""
        r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y * 255.0 - 128.0

    # --- 1. FORWARD: Image -> Network Input ---
    def forward_transform(self, image_path, q_quality=10):
        # Load and Preprocess
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).to(self.device)

        # Get Y Channel
        y_channel = self.rgb_to_y_centered(img_tensor) # (H, W)

        # Patchify (Unfold)
        # (H, W) -> (N_blocks, 8, 8)
        patches = y_channel.unfold(0, 8, 8).unfold(1, 8, 8)
        patches = patches.contiguous().view(-1, 8, 8)

        # DCT
        dct_coeffs = torch.matmul(self.dct_mat, torch.matmul(patches, self.dct_mat_t))

        # Quantization
        q_scale = float(q_quality)
        q_table = self._get_q_table(q_scale)
        dct_quantized = torch.round(dct_coeffs / q_table)

        # Reshape for Network (Batch, 64, H/8, W/8)
        blocks_h = self.image_size // 8
        blocks_w = self.image_size // 8

        # (N, 64) -> (H_blocks, W_blocks, 64) -> (64, H_blocks, W_blocks)
        network_input = dct_quantized.view(blocks_h, blocks_w, 64).permute(2, 0, 1)

        # Add Batch Dimension
        network_input = network_input.unsqueeze(0)

        return network_input, img_tensor, q_scale

    # --- 2. INVERSE: Network Input -> Image ---
    def inverse_transform(self, quantized_dct_input, predicted_error=None, q_quality=10):
        """
        quantized_dct_input: (Batch, 64, 32, 32)
        predicted_error: (Batch, 64, 32, 32) - Optional output from your model
        """
        B, C, H_blocks, W_blocks = quantized_dct_input.shape
        q_scale = float(q_quality)
        q_table = self._get_q_table(q_scale)

        # Remove batch dim for single image processing
        dct_stack = quantized_dct_input[0] # (64, 32, 32)

        # A. De-Quantize
        # Reshape to (N_blocks, 8, 8) to match Q-Table shape
        blocks_flat = dct_stack.permute(1, 2, 0).contiguous().view(-1, 8, 8)

        # Apply Quantization Table
        dct_restored = blocks_flat * q_table

        # B. Add Predicted Error (If model output is provided)
        if predicted_error is not None:
            err_stack = predicted_error[0]
            err_flat = err_stack.permute(1, 2, 0).contiguous().view(-1, 8, 8)
            
            # The model predicts the Scaled Error, so we multiply by Q-Table to get real magnitude
            dct_restored = dct_restored + (err_flat * q_table)

        # C. Inverse DCT
        patches_restored = torch.matmul(self.dct_mat_t, torch.matmul(dct_restored, self.dct_mat))

        # D. Stitch Patches Back (Fold)
        patches_grid = patches_restored.view(H_blocks, W_blocks, 8, 8)
        patches_grid = patches_grid.permute(0, 2, 1, 3) # (H_blocks, 8, W_blocks, 8)
        img_y_centered = patches_grid.contiguous().view(H_blocks*8, W_blocks*8)

        # E. Post-process (Un-center Y to 0-255)
        img_y = img_y_centered + 128.0
        img_y = torch.clamp(img_y, 0, 255)

        return img_y

    def calculate_psnr(self, img1, img2):
        # Ensure both are on same device and format
        if len(img1.shape) == 3: img1 = img1[0] # Take Y channel if RGB tensor

        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0: return 100
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
