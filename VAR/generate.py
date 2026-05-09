import os
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

# 导入模型构建函数
from models import build_vae_var

# ==========================================
# 1. 路径配置 (使用绝对路径确保万无一失)
# ==========================================
CKPT_PATH = "/root/autodl-tmp/Lace_Studio_Clean/VAR/local_output/ar-ckpt-last.pth"
VAE_PATH = "/root/autodl-tmp/Lace_Studio_Clean/VAR/vae_ch160v4096z32.pth"
# ⚠️ 确保这个掩膜文件在你的路径中真实存在
COND_IMAGE_PATH = "/root/autodl-tmp/Lace_Studio_Clean/data/processed/lace_0020_cond.png" 
OUTPUT_PATH = "generated_lace_final.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 初始化模型并精确加载权重
# ==========================================
print("🚀 正在初始化 Lace Studio 工业模型...")
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    num_classes=1, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
)

print("📦 加载 VQVAE 编解码器...")
vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu', weights_only=True), strict=True)

print("📦 加载 VAR 自回归模型权重...")
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)

if 'trainer' in ckpt and 'var_wo_ddp' in ckpt['trainer']:
    raw_state = ckpt['trainer']['var_wo_ddp']
elif 'trainer' in ckpt and 'var' in ckpt['trainer']:
    raw_state = ckpt['trainer']['var']
else:
    raw_state = ckpt.get('trainer', ckpt)

clean_state = {}
for k, v in raw_state.items():
    new_k = k
    for prefix in ['module.', 'var.', 'var_wo_ddp.']:
        if new_k.startswith(prefix):
            new_k = new_k[len(prefix):]
    if not new_k.startswith('optimizer'):
        clean_state[new_k] = v

var.load_state_dict(clean_state, strict=True)
vae.eval()
var.eval()
print("✅ 模型加载与权重对齐成功！")

# ==========================================
# 3. 掩膜图处理 (对齐 256x256 物理尺寸)
# ==========================================
def process_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到图片: {path}")
    img = Image.open(path).convert('RGB')
    # 🌟 修改为 256，以匹配模型内在输出尺寸
    img = TF.resize(img, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
    img_tensor = TF.to_tensor(img)
    img_tensor = img_tensor.mul(2).sub(1) 
    return img_tensor.unsqueeze(0).to(device)

print(f"🎨 读取掩膜骨架: {COND_IMAGE_PATH}")
cond_tensor = process_mask(COND_IMAGE_PATH)

# ==========================================
# 4. 工业级生成参数 (核心魔法)
# ==========================================
CFG_SCALE = 3.0  # 越高越贴合掩膜轮廓，越低细节越丰富
TOP_K = 900      # 保持高值以获得蕾丝的丝线肌理
TOP_P = 0.95

print(f"✨ 开始编织蕾丝细节... (CFG={CFG_SCALE})")
with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=True):
        
        # 🌟 autoregressive_infer_cfg 内部其实已经调用了 VAE 解码！
        # 它的返回值直接就是 [B, 3, H, W] 的重建图像，我们直接拿来用！
        generated_image = var.autoregressive_infer_cfg(
            1, cond_tensor, 
            cfg=CFG_SCALE, top_k=TOP_K, top_p=TOP_P
        )

# ==========================================
# 5. 输出与保存
# ==========================================
# 确保是浮点数，转换回 [0, 1] 范围并保存
final_res = generated_image.float().add(1).mul(0.5).clamp(0, 1)
save_image(final_res, OUTPUT_PATH)
print(f"🎉 成功！蕾丝仿真图已保存至: {os.path.abspath(OUTPUT_PATH)}")