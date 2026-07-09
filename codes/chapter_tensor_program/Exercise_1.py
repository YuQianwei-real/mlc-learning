# exercise 1
#请编写一个 TensorIR 函数，将两个数组以广播的方式相加

# init data
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)

# numpy version
c_np = a + b
c_np

@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4, 4), "int64"),
            B: T.Buffer((4,), "int64"),
            C: T.Buffer((4, 4), "int64")):
        T.func_attr({"global_symbol": "add", "tirx.noalias": True})
        for i, j in T.grid(4, 4):
            with T.sblock("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vj]
    
rt_lib = tvm.compile(MyAdd, target="llvm")
a_tvm = tvm.runtime.tensor(a)
b_tvm = tvm.runtime.tensor(b)
c_tvm = tvm.runtime.tensor(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)

