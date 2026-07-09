# exercise 2

# A 是输入张量，W 是权重张量，b 是批次索引，k 是输出通道，i 和 j 是图像高度和宽度的索引，di 和 dj 是权重的索引，q 是输入通道，strides 是过滤器窗口的步幅
# stride=1, padding=0
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

# torch version
import torch
data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
conv_torch

@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(A: T.Buffer((N, CI, H, W), "int64"),
           B: T.Buffer((CO, CI, K, K), "int64"),
           C: T.Buffer((N, CO, OUT_H, OUT_W), "int64")):
        T.func_attr({"global_symbol": "conv", "tirx.noalias": True})
    
        for n, ci, co, h, w, kh, kw in T.grid(N, CI, CO, OUT_H, OUT_W, K, K):
            with T.sblock("C"):
                vn = T.axis.spatial(1, n)
                vci = T.axis.spatial(1, ci)
                vco = T.axis.spatial(2, co)
                vh = T.axis.spatial(6, h)
                vw = T.axis.spatial(6, w)

                vkh = T.axis.reduce(3, kh)
                vkw = T.axis.reduce(3, kw)
                with T.init():
                    C[vn, vco, vh, vw] = T.int64(0)

                C[vn, vco, vh, vw] = C[vn, vco, vh, vw] + A[vn, vci, vh+vkh, vw+vkw]*B[vco, vci, vkh, vkw]

rt_lib = tvm.compile(MyConv, target="llvm")
data_tvm = tvm.runtime.tensor(data)
weight_tvm = tvm.runtime.tensor(weight)
conv_tvm = tvm.runtime.tensor(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)
