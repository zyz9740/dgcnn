# benchmark_app 使用说明
## CPU性能测试命令
```bash
benchmark_app -m dgcnn_simplified.xml -d CPU -niter 100 -shape [1,3,1024]
```
## GPU性能测试命令
```bash
benchmark_app -m dgcnn_simplified.xml -d GPU -niter 100 -shape [1,3,1024]
```
## 结果查看
- CPU测试结果：benchmark_cpu_result.txt
- GPU测试结果：benchmark_gpu_result.txt
- 关键指标：吞吐量（Throughput）、平均延迟（Latency）
