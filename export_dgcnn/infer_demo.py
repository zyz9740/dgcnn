import openvino as ov
import numpy as np

# 加载模型
core = ov.Core()
model = core.read_model("dgcnn_simplified.xml")
compiled_model = core.compile_model(model, "CPU")
infer_request = compiled_model.create_infer_request()

# 生成随机测试点云：1个样本，3个坐标，1024个点
input_data = np.random.randn(1, 3, 1024).astype(np.float32)

# 推理
result = infer_request.infer(inputs={0: input_data})
output = result[compiled_model.output(0)]

# 输出分类结果
pred_class = np.argmax(output, axis=1)[0]
print(f"推理完成，预测类别ID：{pred_class}")
print(f"输出形状：{output.shape}")
print("Demo运行成功！")
