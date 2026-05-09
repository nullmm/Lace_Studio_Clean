import os
import torch
from torchvision.utils import save_image

# 导入模型构建函数
from models import build_vae_var

# ==========================================
# 1. 路径配置
# ==========================================
CKPT_PATH = "/root/autodl-tmp/Lace_Studio_Clean/VAR/local_output/ar-ckpt-last.pth"
VAE_PATH = "/root/autodl-tmp/Lace_Studio_Clean/VAR/vae_ch160v4096z32.pth"
# 输出文件名改了一下，防止覆盖你之前的图
OUTPUT_PATH = "generated_lace_uncond_0.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 初始化模型并精确加载权重
# ==========================================
print("🚀 正在初始化 Lace Studio 无条件生成模型...")
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
# 3. 制造无条件输入 (全零张量对齐训练状态)
# ==========================================
print("🎨 正在生成无条件的全零骨架...")
# 直接凭空捏造一个 1批次、3通道、256x256 的全零张量
cond_tensor = torch.zeros((1, 3, 256, 256), device=device)

# ==========================================
# 4. 工业级生成参数 (核心魔法)
# ==========================================
# 🌟 核心修改：无条件生成不需要强引导，CFG调低让模型自由发挥
CFG_SCALE = 1
TOP_K = 250    # 保持高值以获得蕾丝的丝线肌理
TOP_P = 0.90
TEMPERATURE = 0.85

print(f"✨ 開始無條件盲盒抽卡... (CFG={CFG_SCALE}, Temp={TEMPERATURE}, TopK={TOP_K})")
with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=True):
        generated_image = var.autoregressive_infer_cfg(
            1, cond_tensor, 
            cfg=CFG_SCALE, top_k=TOP_K, top_p=TOP_P,
            temperature=TEMPERATURE  # 🌟 把溫度參數傳進去！
        )

# ==========================================
# 5. 输出与保存
# ==========================================
# 确保是浮点数，转换回 [0, 1] 范围并保存
final_res = generated_image.float().add(1).mul(0.5).clamp(0, 1)
save_image(final_res, OUTPUT_PATH)
print(f"🎉 盲盒抽卡成功！全新蕾丝仿真图已保存至: {os.path.abspath(OUTPUT_PATH)}")