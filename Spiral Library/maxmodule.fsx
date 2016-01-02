/// o <- max_k(x)
/// Sets all except the k number of max of a column to zero.
/// Unlike for the other modules, to ensure it works swiftly, the column size and the number of iterations is fixed so the compiler can unroll the loops.
type DeviceMaxSelectColumnActivationModule(column_size: int) = 
    let block_size = 32 // This should not be changed for this module. Small block sizes such as these are much more efficient on Maxwell.

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            #define INIT_MIN __int_as_float(0xff800000) // The constant init for the reduce operations. This is the float negative infinity.
            #define INIT_MAX __int_as_float(0x7f800000) // The constant init for the reduce operations. This is the float positive infinity.
            #define NUM_VARS "+string (divup column_size 32)+" // This is to ensure that the number of variables is encoded as a constant.
            typedef "+FloatTypeCpp+" floatType;
            // The max reduce version.
            __device__ inline floatType warpReduce(floatType value){
                #pragma unroll
	            for (int i=1; i<32; i*=2) {
                    floatType tmp = __shfl_xor(value, i);
                    value = (tmp > value) ? tmp : value;
                    }
	            return value;
            }
              
            // Device code
            __global__ void Kernel(const floatType* A, floatType* O, const int num_rows, const int k)
            {
                //int row = threadIdx.x;
                //const int col = blockIdx.x;
                const int col_idx = blockIdx.x*num_rows; 
                floatType upper_bound = INIT_MAX; // This is the positive infinity for floats.
                floatType lower_bound = INIT_MIN; // This is the negative infinity for floats.
                
                floatType vars[NUM_VARS]; // The local array size needs to be made constant so the variables there get stored into registers instead of being spilled into global memory.

                #pragma unroll // Loop unrolling for improved performance. For this to work the number of unrolls has to be defined as a constant.
                for (int i=0; i < NUM_VARS;i++) { 
                    const int row = threadIdx.x + i*32;
                    const int idx = row+col_idx;
                    vars[i] = (row < num_rows) ? A[idx] : INIT_MIN;
                }
                for (int iters=1; iters <= k; iters++){
                    #pragma unroll
                    for (int i=0; i < NUM_VARS;i++) { 
                        const int row = threadIdx.x + i*32;
                        if (vars[i] < upper_bound && lower_bound < vars[i]) lower_bound = vars[i];
                    }
                    upper_bound = warpReduce(lower_bound); // Lowers the upper bound.
                    lower_bound = INIT_MIN;
                }
                #pragma unroll
                for (int i=0; i < NUM_VARS;i++) { 
                    const int row = threadIdx.x + i*32;
                    const int idx = row+col_idx;
                    if (row < num_rows){
                        O[idx] = (vars[i] < upper_bound) ? 0.0f : vars[i];
                    }
                }
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

    member t.A(x: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>, m:int, n: int, k: int) =
        kernel.GridDimensions <- dim3(n)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream,x.DevicePointer,o.DevicePointer,m,k) |> ignore

    member t.A(x: dMatrix, k, o: dMatrix) =
        if x.rc <> o.rc then failwith "x.rc <> o.rc"
        if divup x.num_rows 32 <> divup column_size 32 then failwith "Wrong num_rows."
        t.A(x.dArray,o.dArray,x.num_rows,x.num_cols,k)

    member t.A(x: dMatrix, k) =
        let o = dMatrix.create(x.num_rows,x.num_cols)
        if divup x.num_rows 32 <> divup column_size 32 then failwith "Wrong num_rows."
        t.A(x.dArray,o.dArray,x.num_rows,x.num_cols,k)
        o

let DeviceMaxSelectDict = lazy Dictionary<int,DeviceMaxSelectColumnActivationModule>()
/// o <- max_k(x)
/// Sets all except the k number of max of a column to zero.
/// Unlike for the other modules, to ensure it works swiftly, the column size and the number of iterations is fixed so the compiler can unroll the loops.
/// This name is for a function wrapper for the Dict that holds the DeviceMaxSelectColumnActivation modules.
let DeviceMaxSelectColumnActivationModule column_size =
    let d = DeviceMaxSelectDict.Value
    if d.ContainsKey(divup column_size 32) then
        d.[divup column_size 32]
    else
        let t = DeviceMaxSelectColumnActivationModule column_size
        d.Add(divup column_size 32,t)
        t

/// The winner take all activation. Zeroes out the non top-k values along the row.
let sparseActivationErrorModule = lazy new DeviceTrinaryTransformModule "y*((x == 0.0f) ? 0.0f : 1.0f)+z;"
let WTA k (a:DM) =
    let va = a.r.P
    let el = a.r.A

    let node = tape.GetDMIf
    let c = node.P
    let error = node.A

    let aux = tape.GetDMIf
    let aux1 = aux.P

    let ff () = 
        let nr,nc = (va).rc
        node.Resize nr nc
        aux1.ReplaceIf nc nr
        geam2 T T 1.0f va 0.0f va aux1 // Transpose
        DeviceMaxSelectColumnActivationModule(nc).A(aux1,k,aux1)
        geam2 T T 1.0f aux1 0.0f aux1 c // Transpose
    let fb () = sparseActivationErrorModule.Value.A(c,error,el,el)
    let t = DM.create(node,ff,fb)
    tape.Add(t)
    t
