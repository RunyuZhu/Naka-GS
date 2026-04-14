import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
checkpoint_path = "./checkpoint/model.pt"
state_dict = torch.load(checkpoint_path)
model = VGGT().to(device)
model.load_state_dict(state_dict)

# 指定图片文件夹路径
folder_path = './images'
# 图片格式
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

# 获取文件夹下所有图片的路径
image_names = [
    os.path.join(folder_path, file)
    for file in os.listdir(folder_path)
    if file.lower().endswith(image_extensions)
]

# 打印结果（可选）
print(image_names)
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
