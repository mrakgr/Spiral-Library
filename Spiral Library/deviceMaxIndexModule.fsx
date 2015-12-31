/// o <- max_ind_col(x)
/// Finds the index of the max element of the columns.
type DeviceMaxIndexColumnModule() = 
    let block_size = 128

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            // The max reduce version.
            __device__ inline "+FloatTypeCpp+" warpReduce("+FloatTypeCpp+" value){
	            for (int i=1; i<32; i*=2) {
                    "+FloatTypeCpp+" tmp = __shfl_xor(value, i);
                    value = (tmp > value) ? tmp : value;
                    }
	            return value;
            }

            __device__ inline "+FloatTypeCpp+" blockReduce("+FloatTypeCpp+" value){
	            __shared__ "+FloatTypeCpp+" temp[32];
                if (threadIdx.x < 32) temp[threadIdx.x] = __int_as_float(0xff800000); "+FloatTypeCpp+" out_partial = warpReduce(value);
                __syncthreads();
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
                __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpReduce(temp[threadIdx.x]);
                return out_partial;
            }

            // Device code
            __global__ void Kernel(const "+FloatTypeCpp+"* A, int* O, const int num_rows, const int num_cols)
            {
                int row = threadIdx.x;
                //const int col = blockIdx.x;
                int col_idx = blockIdx.x*num_rows; "+FloatTypeCpp+" max = __int_as_float(0xff800000); // This is the negative infinity for floats.
                int max_index = -1; 
                while (row < num_rows)
                {
                    if (A[row+col_idx] > max) {
                        max = A[row+col_idx];
                        max_index = row;
                    }
                    row += blockDim.x;
                }
                    "+FloatTypeCpp+" tmp = blockReduce(max);
                    if (max_index != -1 && tmp == max) O[blockIdx.x] = max_index; // In case of multiple equal values there will be conflicts, but it does not matter.
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Kernel")
    do  
        try k.Compile([|"-arch=compute_30"|])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Kernel")

    member t.A(x: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<int>, m:int , n: int) =
        kernel.GridDimensions <- dim3(n)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream,x.DevicePointer,o.DevicePointer,m,n) |> ignore

    member t.A(x: dMatrix) =
        use o = new_dev<int> x.num_cols
        t.A(x.dArray,o,x.num_rows,x.num_cols)
        o.Gather()

