% aiak_sglang

编译算子时的几个依赖，需要和运行时保持一致：

- torch 版本：2.8.0a0+5228986c39.nv25.6
- cuda 版本：12.9

### 备份 torch

训练提供的镜像的 torch 版本为特殊版本，备份以免装某些依赖时被重新装了。

```sh
cd $(python -c "import site; print(site.getsitepackages()[0])")
zip -r torch-2.8.0a0+5228986c39.nv25.06-py3-none-any.whl torch torch-*.dist-info
```

### 编译 speedgate

```sh
# 1. 安装 speedgagte
# 2. 子模块 cutlass
git submodule update --recursive --init
# 3. 打包 wheel
python setup.py install
# 4. 单测
python script/gemm/benchmark.py
```

### 编译 deep ep

```sh
# 指定 NVSHMEM_DIR 所在目录
# 指定 CUDA_ARCH
TORCH_CUDA_ARCH_LIST="8.0;9.0" CUDA_ARCH=sm_90 NVSHMEM_DIR=/usr/local/nvshmem/ python3 setup.py build
```

### 编译 flashinfer 0.3.1

```sh
# 指定 --no-deps 使用当前 torch 环境
pip install flashinfer --no-deps
```

### 编译 vllm 0.6.4.post1

```sh
# 下载 vllm 0.6.4.post1 源码
# 替代 torch 版本后编译
python3 use_existing_torch.py
python3 setup.py bdist_wheel
```

### 安装 sgl-kernel

```sh
export JOBS=64
export CMAKE_BUILD_PARALLEL_LEVEL=${JOBS}
export MAX_JOBS=${JOBS}
export TORCH_CUDA_ARCH_LIST=9.0
set_proxy
pip3 install uv
cd sgl-kernel
make rebuild
```

### 安装中的网络问题

有时候 cmake 需要从外网 fetch 依赖，FetchContent 时卡住需要设置代理，使用 `set_proxy`。

```sh
set_proxy() {
    export http_proxy=http://agent.baidu.com:8891
    export https_proxy=http://agent.baidu.com:8891
    export no_proxy=registry.baidubce.com,pip.baidu.com,mirrors.baidubce.com,icode.baidu.com,baidu-ide.bj.bcebos.com,bj.bcebos.com,localhost,127.0.0.1,100.66.166.17,10.11.155.188,10.11.232.15
    export GIT_SSL_NO_VERIFY=1
}

export TMOUT=0

unset_proxy() {
    export http_proxy=""
    export https_proxy=""
}
```

### 额外依赖

```
pip3 install setproctitle
```

### 测试

wget http://10.215.192.25:8291/install.sh

PREFILL 启动

```
MODEL_ID=aiak_deepseek_h200_r1_rd_test1 HTTP_PORT=8000 PYTHONPATH=/workspace/aiak_sglang/python:$PYTHONPATH LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/x86_64-linux-gnu:/usr/local/nvshmem/lib:{LD_LIBRARY_PATH} NCCL_DEBUG=INFO NCCL_IB_ADAPTIVE_ROUTING=1 NCCL_IB_QPS_PER_CONNECTION=2 NCCL_IB_TIMEOUT=22 NCCL_IB_GID_INDEX=3 NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 NVSHMEM_IB_GID_INDEX=3 NVSHMEM_IB_TRAFFIC_CLASS=130 NVSHMEM_BOOTSTRAP=UID NCCL_SOCKET_IFNAME=bond0 PD_SEPARATE_STAGE=prefill ENABLE_SELECT_EXPERTS=1 MOONCAKE_ASYNC_SEND=TRUE MOONCAKE_META_SERVER_URL=redis://10.220.18.51:6379 MOONCAKE_META_SERVER_PASSWORD=zyz-test MOONCAKE_SEND_RECV_TIMEOUT=60 SET_DEEP_DP_NUM_EXPERTS=256 USE_SPEEDGATE_COMPUTE_POSITION=1 USE_SPEEDGATE_REQ_TO_TOKEN_POOL=1 NCCL_NVLS_ENABLE=0 EP_STATS_DUMP_THRESH=0.95 REDUNDANT_EXPERT_META_DIR=/workspace/ep_stats NVSHMEM_DISABLE_P2P=true ENABLE_ONLY_TP0_SEND_KVCACH=1 ENABLE_MICRO_BATCH_OVERLAP=0 ENABLE_PREFILL_OVERLAP=0               python3 -m sglang.launch_server --model-path /dzf-offline-tp1-local --tp 4 --dist-init-addr $(hostname -i):6676 --chunked-prefill-size 16384 --max-prefill-tokens 32768 --dp 2 --nnodes 1 --node-rank 0 --trust-remote-code --enable-ep-moe --disable-overlap-schedule --disable-cuda-graph --port 8000 --host 0.0.0.0 --return-token-ids --load-balance-method dp_affinity --kv-role kv_producer --decode-log-interval 2 --max-total-tokens 150000 --enable-metrics --mem-fraction-static 0.95 --attention-backend fa3 --speculative-algo NEXTN --speculative-draft /dzf-offline-tp1-local-nextn --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --max-running-requests 256 --disable-radix-cache
```

DECODE 启动：

```
MODEL_ID=aiak_deepseek_h200_r1_rd_test1 SELECT_EXPERTS_FREP=1 HTTP_PORT=8000 AIAK_DEBUG_SUBMODULE=0 AICK_BCCL_ALLTOALLV_ALGO=NVSHMEM AICK_NVSHMEM_BLOCKS=1 AICK_NVSHMEM_BUFFSIZE_MB=8192 AICK_NVSHMEM_MAX_TOKEN_NUM=16384 BCCL_ERROR_FILE=/workspace/err.nccl.%h.%p.log DISABLE_TP_WITH_SP=0 ENABLE_COPY_AND_SET_NATIVE=0 ENABLE_DEEP_DP=1 ENABLE_FP8_TRANS=0 ENABLE_FUSED_MOE_CPP_RUNTIME=1 ENABLE_MICRO_BATCH_OVERLAP=1 ENABLE_POST_PROC_CUDA=1 ENABLE_POST_REORDER_FUSION=1 ENABLE_SELECT_EXPERTS=0 ENABLE_SILU_AND_MUL_ONE_EXPERT=1 MC_MAX_EP_PER_CTX=3200 ENABLE_ONLY_TP0_SEND_KVCACH=1 ENABLE_SORT_CHUNKS_BY_IDXS_CUDA=1 GROUPED_GEMM_EXPECTED_M=4096 LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/x86_64-linux-gnu:/usr/local/nvshmem/lib:{LD_LIBRARY_PATH} MOONCAKE_ASYNC_SEND=FALSE MOONCAKE_META_SERVER_PASSWORD=zyz-test MOONCAKE_META_SERVER_URL=redis://10.220.18.51:6379 MOONCAKE_SEND_RECV_TIMEOUT=10 MOONCAKE_REMOTE_TP_SIZE=4 MOONCAKE_USE_RDMA=TRUE NCCL_DEBUG=INFO NCCL_IB_ADAPTIVE_ROUTING=1 NCCL_IB_QPS_PER_CONNECTION=2 NCCL_IB_TIMEOUT=22 NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 NVSHMEM_DISABLE_P2P=true NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 NVSHMEM_IBGDA_NUM_RC_PER_PE=1 NVSHMEM_IB_ENABLE_IBGDA=true NVSHMEM_IB_GID_INDEX=3 NVSHMEM_IB_TRAFFIC_CLASS=130 NVSHMEM_SYMMETRIC_HEAP_SIZE=32G ORIGIN_MODEL_PATH=/qianfan_preset_models_for_yiyan_pipeline/DeepSeek-V3-Hzz1/ PD_SEPARATE_STAGE=decode PYTHONPATH=/workspace/aiak_sglang/python:$PYTHONPATH SET_DEEP_DP_MAX_TOKENS=256 SET_DEEP_DP_NUM_EXPERTS=320 SYNC_COMM=0 USE_CUDA_ROPE=1 USE_CUTLASS_FP8_BLOCK_GEMM=1 USE_FAST_A2A=1 USE_FUSED_INPUT_TO_FP8=1 GC_COLLECT_GEN_STEP=10000 NCCL_NVLS_ENABLE=0 ASYNC_SEND_KV_CACHE=1 AIAK_ENABLE_EXPERT_TRACE=0 EP_STATS_DUMP_THRESH=0.95 REDUNDANT_EXPERT_META_DIR=/workspace/ep_stats NCCL_BUFFSIZE=524288 NCCL_LL128_BUFFSIZE=524288 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 ENABLE_MTP_DECODER_OPT=1               python3 -m sglang.launch_server --model-path /dzf-offline-tp1-local --tp 1 --dist-init-addr $(hostname -i):6676 --chunked-prefill-size -1 --max-prefill-tokens 8192 --dp 8 --nnodes 1 --node-rank 0 --trust-remote-code --enable-ep-moe --disable-radix-cache --disable-overlap-schedule --cuda-graph-max-bs 80 --port 8000 --host 0.0.0.0  --load-balance-method dp_affinity --kv-role kv_consumer --decode-log-interval 2 --max-total-tokens 50000 --enable-metrics --max-running-requests 128 --enable-flashinfer-mla --mem-fraction-static 0.95 --speculative-algo NEXTN --flashinfer-mla-disable-ragged --speculative-draft /dzf-offline-tp1-local-nextn --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
```

```
curl -s http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "aiak_deepseek_h200_r1_rd_test1",
    "messages": [{"role": "user", "content": "what is the capital of france?"}],
    "stream": false,
    "max_tokens": 1
  }'
```
