import torch
import argparse
from model import DGCNN

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=20, help='Number of nearest neighbors')
parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of embeddings')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
args = parser.parse_args()

# 加载模型
device = torch.device('cpu')
model = DGCNN(args).to(device)
from collections import OrderedDict
checkpoint = torch.load('pretrained/model.1024.t7', map_location=device)
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    if k.startswith('module.'):
        name = k[7:] # remove `module.` prefix
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

# 示例输入
dummy_input = torch.randn(1, 3, 1024, device=device)

# 导出ONNX
torch.onnx.export(model, 
                  dummy_input, 
                  "dgcnn.onnx", 
                  export_params=True,
                  opset_version=16,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("ONNX模型导出成功：dgcnn.onnx")
