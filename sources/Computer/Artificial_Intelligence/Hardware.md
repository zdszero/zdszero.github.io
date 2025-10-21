% Hardware

### MFU

**当你计算一个芯片的 MFU（Model FLOPs Utilization）时，必须考虑所使用的数据类型（precision format），例如 FP8、BF16、FP16 等**。

| 名称                                | 含义                                      |
| --------------------------------- | --------------------------------------- |
| **Theoretical peak FLOPs**        | 芯片在给定精度下的最大理论算力（例如 FP8 算力 vs BF16 算力不同） |
| **Measured FLOPs**                | 实际模型执行过程中统计到的有效浮点运算数量                   |
| **MFU (Model FLOPs Utilization)** | 模型在实际运行中利用了理论算力的多少比例                    |

计算公式通常是：

$$\text{MFU} = \frac{\text{Measured FLOPs}}{\text{Theoretical peak FLOPs for the same precision}}$$

H200 上的 Tensor Core 是“同一批硬件单元”，**支持多种数据精度模式**；
不同精度下的峰值算力差异源于每个 Tensor Core 在该模式下的有效吞吐率不同，而不是“专门为某个精度配置了不同数量的 Tensor Core”。

### HGX

| Technical Specifications¹ | GB200 NVL72 | HGX B200 |
|---------------------------|-------------|----------|
| **Blackwell GPUs \| Grace CPUs** | 72 \| 36 | 8 \| 0 |
| **CPU Cores** | 2,592 Arm Neoverse V2 Cores | - |
| **Total FP4 Tensor Core** | 1,440 PFLOPS | 144 PFLOPS |
| **Total FP8/FP6 Tensor Core** | 720 PFLOPS | 72 PFLOPS |
| **Total Fast Memory** | Up to 30TB | Up to 1.4TB |
| **Total Memory Bandwidth** | Up to 576TB/s | Up to 62TB/s |
| **Total NVLink Bandwidth** | 130TB/s | 14.4TB/s |
| **Individual Blackwell GPU Specifications** | | |
| **FP4 Tensor Core** | 20 PFLOPS | 18 PFLOPS |
| **FP8/FP6 Tensor Core** | 10 PFLOPS | 9 PFLOPS |
| **INT8 Tensor Core** | 10 POPS | 9 POPS |
| **FP16/BF16 Tensor Core** | 5 PFLOPS | 4.5 PFLOPS |
| **TF32 Tensor Core** | 2.5 PFLOPS | 2.2 PFLOPS |
| **FP32** | 80 TFLOPS | 75 TFLOPS |
| **FP64/FP64 Tensor Core** | 40 TFLOPS | 37 TFLOPS |
| **GPU Memory \| Bandwidth** | 186GB HBM3E \| 8TB/s | 180GB HBM3E \| 7.7TB/s |
| **Multi-Instance GPU (MIG)** | 7 | |
| **Decompression Engine** | Yes | |
| **Decoders** | 7 NVDEC² | |
|  | 7 nvJPEG | |
| **Max Thermal Design Power (TDP)** | Configurable up to 1,200W | Configurable up to 1,000W |
| **Interconnect** | 5th Generation NVLink: 1.8TB/s | |
|  | PCIe Gen5: 128GB/s | |
| **Server Options** | NVIDIA GB200 NVL72 partner and NVIDIA-Certified Systems™ with 72 GPUs | NVIDIA HGX B200 partner and NVIDIA-Certified Systems with 8 GPUs |

### Hopper

| Specification             | H200 SXM | H200 NVL |
|----------------------------|------------|------------|
| **FP64**                   | 34 TFLOPS  | 30 TFLOPS  |
| **FP64 Tensor Core**       | 67 TFLOPS  | 60 TFLOPS  |
| **FP32**                   | 67 TFLOPS  | 60 TFLOPS  |
| **TF32 Tensor Core**      | 989 TFLOPS | 835 TFLOPS |
| **BFLOAT16 Tensor Core**  | 1,979 TFLOPS | 1,671 TFLOPS |
| **FP16 Tensor Core**      | 1,979 TFLOPS | 1,671 TFLOPS |
| **FP8 Tensor Core**       | 3,958 TFLOPS | 3,341 TFLOPS |
| **INT8 Tensor Core**      | 3,958 TFLOPS | 3,341 TFLOPS |
| **GPU Memory**             | 141 GB     | 141 GB     |
| **GPU Memory Bandwidth**   | 4.8 TB/s   | 4.8 TB/s   |

