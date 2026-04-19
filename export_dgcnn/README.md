# DGCNN OpenVINO 转换结果
## 模型说明
DGCNN是点云分类模型，基于ModelNet40数据集训练，支持40类物体分类，输入为(1, 3, 1024)的点云数据，输出为40类的分类概率。
## 文件说明
| 文件名 | 说明 |
|--------|------|
| dgcnn_simplified.xml | OpenVINO模型结构文件 |
| dgcnn_simplified.bin | OpenVINO FP16精度权重文件 |
| infer_demo.py | 开箱即用推理Demo |
| README.md | 本说明文件 |
| benchmark_cpu_result.txt | 官方benchmark_app CPU性能测试原始日志 |
| benchmark_app_usage.md | benchmark_app使用说明 |
## 环境依赖
```bash
pip install openvino numpy
```
## 快速运行
```bash
python infer_demo.py
```
## 从零导出流程
1. 克隆DGCNN仓库：`git clone https://github.com/WangYueFt/dgcnn.git`
2. 进入pytorch目录，修改model.py中的device为`device = x.device`
3. 运行转换脚本：`python convert_to_openvino.py`
