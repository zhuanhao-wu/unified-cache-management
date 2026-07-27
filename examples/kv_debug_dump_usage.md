# UCM KV Cache 快照校验

该调试功能用于核对 vLLM 交给 UCM 保存的 KV cache，和 UCM 加载回 vLLM buffer 后的数据是否一致：

- `store_before`：调用 `dump_data` 前，从 vLLM paged KV cache 读取的快照。
- `load_after`：`load_data` 的 task 完成后，从同一个 vLLM block 位置读取的快照。

文件保存的是 CPU 上的 PyTorch tensor，不是 ASU 服务端的原始 MR 数据。功能默认关闭，不影响正常路径。

## 开启快照

启动推理服务前设置：

```bash
export UCM_KV_DEBUG_DUMP_DIR=/tmp/ucm_kv_dump
export UCM_KV_DEBUG_MAX_BLOCKS=2
export UCM_KV_DEBUG_MAX_LAYERS=2
rm -rf "${UCM_KV_DEBUG_DUMP_DIR}"
mkdir -p "${UCM_KV_DEBUG_DUMP_DIR}"
```

变量含义：

| 变量 | 含义 |
| --- | --- |
| `UCM_KV_DEBUG_DUMP_DIR` | 输出目录；不设置时关闭快照。 |
| `UCM_KV_DEBUG_MAX_BLOCKS` | 每次最多保存的 block 数，默认 `2`；`0` 表示不限制。 |
| `UCM_KV_DEBUG_MAX_LAYERS` | 每次最多保存的 layer 数，默认 `0`，表示不限制。 |

设备到 CPU 的拷贝和磁盘写入会同步执行，明显影响性能。第一次定位建议使用 `2 blocks + 2 layers`；确认存在错误后再扩大范围。

## 输出内容

普通 connector 每个 request/phase 生成一个文件，例如：

```text
000001_store_before_rank0_reqrequest-1.pt
000002_load_after_rank0_reqrequest-2.pt
```

layer-wise connector 会在每层 LOAD 完成后以及每层 STORE 提交前分别生成文件，每个文件只包含对应层。

每个文件记录：

- `request_id`、`engine_id`、TP/local rank 和进程 PID；
- UCM block hash 到 vLLM paged-cache block id 的映射；
- 选定 block 在各层中的 tensor、shape 和 dtype。

对比时必须选择相同 TP rank 和相同 UCM block hash 的 `store_before`/`load_after` 文件。request id 可以不同，因为 LOAD 通常来自后续请求。

## 比较 STORE 与 LOAD

```bash
python examples/compare_kv_debug_dump.py \
  /tmp/ucm_kv_dump/000001_store_before_rank0_reqrequest-1.pt \
  /tmp/ucm_kv_dump/000004_load_after_rank0_reqrequest-2.pt
```

默认执行精确数值比较。成功时输出：

```text
checked=4, mismatches=0
KV_CHECK_PASSED
```

失败时会逐 tensor 打印 shape、dtype、最大/平均绝对误差，最终输出 `KV_CHECK_FAILED` 并返回退出码 `1`。如需允许浮点误差：

```bash
python examples/compare_kv_debug_dump.py STORE.pt LOAD.pt \
  --atol 1e-5 --rtol 1e-5
```

## 查看具体数值

```bash
python - <<'PY'
import torch

x = torch.load("/tmp/ucm_kv_dump/STORE.pt", map_location="cpu")
layer = next(iter(x["layers"]))
block = next(iter(x["layers"][layer]))
value = x["layers"][layer][block]
print("layer:", layer)
print("block:", block)
if isinstance(value, torch.Tensor):
    print(value.shape, value.dtype, value.flatten()[:100])
else:
    for index, tensor in enumerate(value):
        print(index, tensor.shape, tensor.dtype, tensor.flatten()[:100])
PY
```

如果 KV 数值一致但推理输出仍异常，应继续核对 block 映射、命中 token 数、position/rope、TP rank 对应关系和调度复用边界。
