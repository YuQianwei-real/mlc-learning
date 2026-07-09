# exercise 3
#如何变换 TensorIR

# 我们提供 lnumpy 函数作为提示：
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)

# 先补写MyBmmRelu
@tvm.script.ir_module
class MyBmmRelu:
    @T.prim_func
    def bmm_relu( A: T.Buffer((16, 128, 128), "float32"),
                  B: T.Buffer((16, 128, 128), "float32"),
                  C: T.Buffer((16, 128, 128), "float32")):
        T.func_attr({"global_symbol": "bmm_relu", "tirx.noalias": True})
        Y = T.alloc_buffer((16, 128, 128), dtype="float32")
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.sblock("Y"):
                vn = T.axis.spatial(16, n)
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vn, vi, vj] = T.float32(0)
                Y[vn, vi, vj] = Y[vn, vi, vj] + A[vn, vi, vk] * B[vn, vk, vj]
        
        for n, i, j in T.grid(16, 128, 128):
            with T.sblock("C"):
                vn = T.axis.spatial(16, n)
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vn, vi, vj] = T.max(Y[vn, vi, vj], T.float32(0))

sch = tvm.s_tir.Schedule(MyBmmRelu)
IPython.display.Code(sch.mod.script(), language="python")
# Also please validate your result

# 这是目标程序：
@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tirx.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.sblock("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.sblock("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.sblock("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))


# transformations
sch = tvm.s_tir.Schedule(MyBmmRelu)
# Step 1. Get blocks
Y = sch.get_sblock("Y", func_name="bmm_relu")
C_blk = sch.get_sblock("C", func_name="bmm_relu")
# Step 2. Get loops
b, i, j, k = sch.get_loops(Y)
b_c, i_c, j_c = sch.get_loops(C_blk)
# Step 3. Organize the loops
j0, j1 = sch.split(j, factors=[None, 8])   # j0 : 16, j1 : 8
k0, k1 = sch.split(k, factors=[None, 4])   # k0 : 32, k1 : 4

sch.reorder(b, i, j0, k0, k1, j1)
sch.reverse_compute_at(C_blk, j0)

sch.parallel(b)
# Step 4. decompose reduction
Y_init= sch.decompose_reduction(Y, k0)

_, _, _, j_init = sch.get_loops(Y_init)
_, _, _, j1_c = sch.get_loops(C_blk)  # Get C's j1 loop
# Step 5. vectorize / parallel / unroll
sch.vectorize(j_init)    
sch.vectorize(j1_c)      # Vectorize C's j1 loop
sch.unroll(k1)
# 验证
tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")

# 构建和评估
before_rt_lib = tvm.compile(MyBmmRelu, target="llvm")
after_rt_lib = tvm.compile(sch.mod, target="llvm")
a_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.mod.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.mod.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))
