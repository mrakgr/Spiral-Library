// The Spiral library v0. Basic reverse mode AD on the GPU.

#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/ManagedCuda.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/NVRTC.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaBlas.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaRand.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/NPP.dll"

#r "../packages/ManagedCuda-CudaDNN.3.0/lib/net45/CudaDNN.dll"


open ManagedCuda
open ManagedCuda.BasicTypes
open ManagedCuda.VectorTypes
open ManagedCuda.CudaBlas
open ManagedCuda.CudaRand
open ManagedCuda.NVRTC
open ManagedCuda.CudaDNN

open System
open System.IO
open System.Collections

let ctx = new CudaContext()
let numSm = ctx.GetDeviceInfo().MultiProcessorCount // The number of streaming multiprocessors on the device.

let str = new CudaStream()
let cublas = CudaBlas(str.Stream)
let cudnn = new CudaDNN.CudaDNNContext()
cudnn.SetStream(str)
let cudaRandom = new CudaRand.CudaRandDevice(GeneratorType.PseudoDefault)
cudaRandom.SetStream(str.Stream)

let rng = System.Random()

// I've done this type aliasing trick to make the DiffSharp backend generic. It might be worth bringing into Spiral.
type floatType = float32
let inline floatType x = float32 x
let FloatTypeCpp = "float"

let Noop() = ()

let inline to_dev (host_ar: 't []) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

let inline to_dev' (host_ar: 't [,]) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

let inline to_host (dev_ar: CudaDeviceVariable<'t>) =
    let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
    dev_ar.CopyToHost(h_a)
    h_a

type CudaDeviceVariable<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> with
    member inline this.Gather() =
        to_host this

let inline new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
    new CudaDeviceVariable<'t>(SizeT n)

type dMatrix(num_rows:int,num_cols:int,dArray: CudaDeviceVariable<floatType>) = 
        new(num_rows: int,num_cols) =
            let q = (num_rows*num_cols) |> SizeT
            let t = new CudaDeviceVariable<floatType>(q)
            new dMatrix(num_rows,num_cols,t)

        new(num_rows: int,num_cols,dArray: floatType[]) =
            let q = num_rows*num_cols
            if dArray.Length <> q then failwith "Invalid size in dMatrix construction."
            let t = to_dev dArray
            new dMatrix(num_rows,num_cols,t)

        member inline t.num_rows = num_rows
        member inline t.num_cols = num_cols
        member inline t.dArray = dArray

        member inline t.rc = t.num_rows, t.num_cols
        /// Sets the matrix to zero.
        member inline t.setZero() = t.dArray.MemsetAsync(0u,str.Stream)
        /// Set the matrix to a value.
        member inline t.set (x: floatType) = 
            let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
            t.dArray.MemsetAsync(v,str.Stream)
        /// Creates a copy of this matrix with all the values set to zero.
        member inline t.zeroLike() =
            let c = new dMatrix(t.num_rows,t.num_cols)
            c.setZero()
            c
        member inline t.copy() =
            let c = new dMatrix(t.num_rows,t.num_cols)
            c.dArray.AsyncCopyToDevice(t.dArray,str)
            c

        /// Returns a new dMatrix if the current one is less than nr*nc. Otherwise it returns self.
        member inline t.ReplaceIf nr nc =
            if num_rows*num_cols < nr*nc then
                (t :> IDisposable).Dispose()
                new dMatrix(nr,nc)
            else
                t

        /// Outright disposes the matrix and returns a new one.
        member inline t.Replace nr nc =
            (t :> IDisposable).Dispose()
            new dMatrix(nr,nc)

        member inline t.Gather() =
            let h_a = Array.zeroCreate<floatType> (int t.dArray.Size)
            t.dArray.CopyToHost(h_a)
            h_a

        member inline t.isEmpty = if t.num_rows = 0 then true else false
        member inline t.Size = num_rows*num_cols

        override t.ToString() =
            sprintf "dM(%i,%i)" t.num_rows t.num_cols

        interface IDisposable with
            member t.Dispose() = 
                if t.dArray.Equals(CudaDeviceVariable.Null) then
                    t.dArray.Dispose()

        override t.Finalize() = (t :> IDisposable).Dispose()

/// Empty dMatrix.
let empty = new dMatrix(0,0,CudaDeviceVariable.Null)

let T = Operation.Transpose
let nT = Operation.NonTranspose

/// General matrix-matrix multiply from cuBLAS.
let sgemm transa transb (alpha: floatType) (A:dMatrix) (B:dMatrix) =
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm! %i <> %i" a_col b_row)
    let m = if transa = nT then A.num_rows else A.num_cols
    let n = if transb = nT then B.num_cols else B.num_rows
    let k = a_col

    let lda = if transa = nT then m else k
    let ldb = if transb = nT then k else n
    let ldc = m

    let C_dArray = new CudaDeviceVariable<floatType>(m*n |> SizeT)
    cublas.Gemm(transa, transb, m, n, k, alpha, A.dArray, lda, B.dArray, ldb, 0.0f, C_dArray, ldc)
    new dMatrix(m,n,C_dArray)

/// General matrix-matrix multiply from cuBLAS. Inplace version
let sgemm2 transa transb (alpha: floatType) (A:dMatrix) (B:dMatrix) beta (C:dMatrix) =
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm! %i <> %i" a_col b_row)
    let m = if transa = nT then A.num_rows else A.num_cols
    let n = if transb = nT then B.num_cols else B.num_rows
    let k = a_col

    let lda = if transa = nT then m else k
    let ldb = if transb = nT then k else n
    let ldc = m

    let C_dArray = C.dArray
    if int C.dArray.Size <> m*n then failwith "C.dArray.Length <> m*n in sgemm2"
    if m <> C.num_rows || n <> C.num_cols then failwith "m <> C.num_rows || n <> C.num_cols in sgemm2"
    cublas.Gemm(transa, transb, m, n, k, alpha, A.dArray, lda, B.dArray, ldb, beta, C_dArray, ldc)

/// General matrix-matrix addition.
let sgeam transa transb (alpha: floatType) (A:dMatrix) beta (B:dMatrix) =
    let a_row = if transa = nT then A.num_rows else A.num_cols
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    let b_col = if transb = nT then B.num_cols else B.num_rows
        
    if a_row <> b_row then failwith (sprintf "a_row <> b_row in sgeam! %i <> %i" a_row b_row)
    if a_col <> b_col then failwith (sprintf "a_col <> b_col in sgeam! %i <> %i" a_col b_col)

    let lda = if transa = nT then a_row else a_col
    let ldb = if transa = nT then b_row else b_col
    let ldc = a_row

    let C_dArray = new CudaDeviceVariable<floatType>(a_row*a_col |> SizeT)
    if A.dArray.Size <> B.dArray.Size then failwith "A.dArray.Length <> B.dArray.Length in sgeam"
    cublas.Geam(transa, transb, a_row, a_col, alpha, A.dArray, lda, B.dArray, ldb, beta, C_dArray, ldc)
    new dMatrix(a_row,a_col,C_dArray)
    

/// General matrix-matrix addition.
let sgeam2 transa transb (alpha: floatType) (A:dMatrix) beta (B:dMatrix) (C:dMatrix) =
    let a_row = if transa = nT then A.num_rows else A.num_cols
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    let b_col = if transb = nT then B.num_cols else B.num_rows
        
    if a_row <> b_row then failwith (sprintf "a_row <> b_row in sgeam2! %i <> %i" a_row b_row)
    if a_col <> b_col then failwith (sprintf "a_col <> b_col in sgeam2! %i <> %i" a_col b_col)

    let lda = if transa = nT then a_row else a_col
    let ldb = if transa = nT then b_row else b_col
    let ldc = a_row

    if A.dArray.Size <> B.dArray.Size then failwith "A.dArray.Length <> B.dArray.Length in sgeam2"
    cublas.Geam(transa, transb, a_row, a_col, alpha, A.dArray, lda, B.dArray, ldb, beta, C.dArray, ldc)

let divup a b = (a+b-1)/b

/// o <- f(x)
type DeviceUnaryTransformModule(op: string) = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __device__ inline float op(float x)
            {
                return "+op+"
            }
        
            // Device code
            __global__ void Map1Kernel(const float* A, float* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i]);
                    i += stride;
                }
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map1Kernel")
    do  
        try k.Compile([||])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map1Kernel")

    member t.A(x: CudaDeviceVariable<floatType>) =
        let n = int x.Size
        let o = new_dev<floatType> n
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore
        o

    member t.A(x: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
        let n = int o.Size
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore

    member t.A(x: dMatrix) =
        let o = new dMatrix(x.num_rows,x.num_cols)
        t.A(x,o)
        o

    member t.A(x: dMatrix, o: dMatrix) =
        t.A(x.dArray,o.dArray)

/// o <- f(x,y)
type DeviceBinaryTransformModule(op: string) = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __device__ inline float op(float x, float y)
            {
                return "+op+"
            }
        
            // Device code
            __global__ void Map2Kernel(const float* A, const float* B, float* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i],B[i]);
                    i += stride;
                }
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map2Kernel")
    do  
        try k.Compile([||])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map2Kernel")

    member t.A(x: CudaDeviceVariable<floatType>, y: CudaDeviceVariable<floatType>) =
        let n = int x.Size
        let o = new_dev<floatType> n
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore
        o

    member t.A(x: CudaDeviceVariable<floatType>, y: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
        let n = int o.Size
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore

    member t.A(x: dMatrix, y: dMatrix) =
        let o = new dMatrix(x.num_rows,x.num_cols)
        t.A(x,y,o)
        o

    member t.A(x: dMatrix, y: dMatrix, o: dMatrix) =
        if x.rc <> y.rc then failwith "x.rc <> y.rc in DeviceBinaryTransformModule"
        t.A(x.dArray,y.dArray,o.dArray)

/// o <- f(x,y,z)
type DeviceTrinaryTransformModule(op: string) = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __device__ inline float op(float x, float y, float z)
            {
                return "+op+"
            }
        
            // Device code
            __global__ void Map3Kernel(const float* A, const float* B, const float* C, float* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i],B[i],C[i]);
                    i += stride;
                }
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map3Kernel")
    do  
        try k.Compile([||])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map3Kernel")

    member t.A(x: CudaDeviceVariable<floatType>, y: CudaDeviceVariable<floatType>, z: CudaDeviceVariable<floatType>) =
        let n = int x.Size
        let o = new_dev<floatType> n
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,z.DevicePointer,o.DevicePointer,n) |> ignore
        o

    member t.A(x: CudaDeviceVariable<floatType>, y: CudaDeviceVariable<floatType>, z: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
        let n = int o.Size
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,z.DevicePointer,o.DevicePointer,n) |> ignore

    member t.A(x: dMatrix, y: dMatrix, z: dMatrix) =
        let o = new dMatrix(x.num_rows,x.num_cols)
        t.A(x,y,z,o)
        o

    member t.A(x: dMatrix, y: dMatrix, z: dMatrix, o: dMatrix) =
        if x.rc <> y.rc then failwith "x.rc <> y.rc in DeviceTrinaryTransformModule"
        if y.rc <> z.rc then failwith "y.rc <> z.rc in DeviceTrinaryTransformModule"
        t.A(x.dArray,y.dArray,z.dArray,o.dArray)

/// o <- sum(f(x))
type DeviceUnaryMapSumModule(op: string) = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __device__ inline float op(float x)
            {
                return "+op+"
            }
        
            __device__ inline float warpDownReduce(float value){
	            for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	            return value;
            }

            // Device code
            __global__ void MapSumKernel(const float* A, float* O, const int N)
            {
	            int i = blockDim.x * blockIdx.x + threadIdx.x;
	            const int stride = blockDim.x * gridDim.x;
	            __shared__ float temp[32];
                if (threadIdx.x < 32) temp[threadIdx.x] = 0.0f;
	
	            float acc = 0.0f;
	            while (i < N)
	            {
		            acc += op(A[i]);
		            i += stride;
	            }
	            __syncthreads();
	            float out_partial = warpDownReduce(acc);
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	            __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	            if (threadIdx.x == 0) atomicAdd(O, out_partial);
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"MapSumKernel")
    do  
        try k.Compile([|"-arch=compute_30"|])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"MapSumKernel")

    member t.A(x: CudaDeviceVariable<floatType>) =
        let n = int x.Size
        use o = new_dev<floatType> 1
        o.Memset(0u)
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore
        o.[SizeT 0]

    member t.A(x: dMatrix) =
        t.A(x.dArray)

/// o <- sum(f(x,y))
type DeviceBinaryMapSumModule(op: string) = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __device__ inline float op(float x, float y)
            {
                return "+op+"
            }
        
            __device__ inline float warpDownReduce(float value){
	            for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	            return value;
            }

            // Device code
            __global__ void Map2SumKernel(const float* A, const float* B, float* O, const int N)
            {
	            int i = blockDim.x * blockIdx.x + threadIdx.x;
	            const int stride = blockDim.x * gridDim.x;
	            __shared__ float temp[32];
                if (threadIdx.x < 32) temp[threadIdx.x] = 0.0f;
	
	            float acc = 0.0f;
	            while (i < N)
	            {
		            acc += op(A[i],B[i]);
		            i += stride;
	            }
	            __syncthreads();
	            float out_partial = warpDownReduce(acc);
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	            __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	            if (threadIdx.x == 0) atomicAdd(O, out_partial);
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map2SumKernel")
    do  
        try k.Compile([|"-arch=compute_30"|])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map2SumKernel")

    member t.A(x: CudaDeviceVariable<floatType>,y: CudaDeviceVariable<floatType>) =
        let n = int x.Size
        use o = new_dev<floatType> 1
        o.Memset(0u)
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore
        o.[SizeT 0]

    member t.A(x: dMatrix,y: dMatrix) =
        if x.rc <> y.rc then failwith "x.rc <> y.rc in DeviceBinaryMapSumModule"
        t.A(x.dArray,y.dArray)

/// o <- f(coef_x,x)
type DeviceUnaryCoefTransformModule(op: string) = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __device__ inline float op(float coef_x, float x)
            {
                return "+op+"
            }
        
            // Device code
            __global__ void MapCoefKernel(const float coef_A, const float* A, float* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(coef_A,A[i]);
                    i += stride;
                }
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"MapCoefKernel")
    do  
        try k.Compile([||])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"MapCoefKernel")

    member t.A(coef_x: floatType, x: CudaDeviceVariable<floatType>) =
        let n = int x.Size
        let o = new_dev<floatType> n
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,o.DevicePointer,n) |> ignore
        o

    member t.A(coef_x: floatType, x: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
        let n = int o.Size
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,o.DevicePointer,n) |> ignore

    member t.A(coef_x, x: dMatrix) =
        let o = new dMatrix(x.num_rows,x.num_cols)
        t.A(coef_x,x,o)
        o

    member t.A(coef_x, x: dMatrix, o: dMatrix) =
        t.A(coef_x,x.dArray,o.dArray)

/// o <- f(coef_x,x,coef_y,y)
type DeviceBinaryCoefTransformModule(op: string) = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __device__ inline float op(float coef_x, float x, float coef_y, float y)
            {
                return "+op+"
            }
        
            // Device code
            __global__ void MapCoef2Kernel(const float coef_A, const float* A, const float coef_B, const float* B, float* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(coef_A,A[i],coef_B,B[i]);
                    i += stride;
                }
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"MapCoef2Kernel")
    do  
        try k.Compile([||])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"MapCoef2Kernel")

    member t.A(coef_x: floatType, x: CudaDeviceVariable<floatType>,coef_y: floatType, y: CudaDeviceVariable<floatType>) =
        let n = int x.Size
        let o = new_dev<floatType> n
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,coef_y, y.DevicePointer,o.DevicePointer,n) |> ignore
        o

    member t.A(coef_x: floatType, x: CudaDeviceVariable<floatType>, coef_y: floatType, y: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
        let n = int o.Size
        let gridSize = min (divup n block_size) 16*numSm
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,coef_y,y.DevicePointer,o.DevicePointer,n) |> ignore

    member t.A(coef_x, x: dMatrix, coef_y, y: dMatrix) =
        let o = new dMatrix(x.num_rows,x.num_cols)
        t.A(coef_x,x,coef_y,y,o)
        o

    member t.A(coef_x, x: dMatrix, coef_y, y: dMatrix, o: dMatrix) =
        if x.rc <> y.rc then failwith "y.rc <> y.rc in DeviceBinaryCoefTransformModule"
        t.A(coef_x,x.dArray,coef_y,y.dArray,o.dArray)

let gradclipModule = DeviceUnaryCoefTransformModule "(x < -coef_x) ? -coef_x : (x > coef_x ? coef_x : x);"

let biasTensorDesc = new TensorDescriptor()
let dstTensorDesc = new TensorDescriptor()

///o <- preactivations + bias (broadcast addition)
let addBias (preactivations: dMatrix) (bias: dMatrix) =
    let DataType = cudnnDataType.Float
    let TensorFormat = cudnnTensorFormat.NCHW;
    biasTensorDesc.SetTensor4dDescriptor(TensorFormat, DataType, 1, 1, bias.num_rows, bias.num_cols)
    dstTensorDesc.SetTensor4dDescriptor(TensorFormat, DataType, 1, preactivations.num_cols, preactivations.num_rows, 1)
    let alpha, beta = 1.f, 1.f
    let copy_preact = preactivations.copy()
    cudnn.AddTensor(alpha,biasTensorDesc,bias.dArray,beta,dstTensorDesc,copy_preact.dArray)
    new dMatrix(copy_preact.num_rows,copy_preact.num_cols,copy_preact.dArray)
    
///preactivations <- preactivations + bias (broadcast addition)
let addBias2 beta (preactivations: dMatrix) alpha (bias: dMatrix) =
    let DataType = cudnnDataType.Float
    let TensorFormat = cudnnTensorFormat.NCHW;
    biasTensorDesc.SetTensor4dDescriptor(TensorFormat, DataType, 1, 1, bias.num_rows, bias.num_cols)
    dstTensorDesc.SetTensor4dDescriptor(TensorFormat, DataType, 1, preactivations.num_cols, preactivations.num_rows, 1)
    cudnn.AddTensor(alpha,biasTensorDesc,bias.dArray,beta,dstTensorDesc,preactivations.dArray)

/// o <- sum_across_channels(alpha*error)
/// For 2D matrices, channels are the columns.
let calculateBias alpha (error: dMatrix) =
    let DataType = cudnnDataType.Float
    let TensorFormat = cudnnTensorFormat.NHWC;
    dstTensorDesc.SetTensor4dDescriptor(TensorFormat, DataType, 1, error.num_rows, 1, error.num_cols)
    
    let bias = new dMatrix(error.num_rows,1)
    biasTensorDesc.SetTensor4dDescriptor(TensorFormat, DataType, 1, bias.num_rows, 1, bias.num_cols)
    
    cudnn.ConvolutionBackwardBias(alpha/floatType error.num_cols,dstTensorDesc,error.dArray,0.0f,biasTensorDesc,bias.dArray)
    bias

/// bias <- sum_across_channels(alpha*error)
/// For 2D matrices, channels are the columns.
let calculateBias2 alpha (error: dMatrix) (bias: dMatrix) =
    let DataType = cudnnDataType.Float
    let TensorFormat = cudnnTensorFormat.NHWC;
    dstTensorDesc.SetTensor4dDescriptor(TensorFormat, DataType, 1, error.num_rows, 1, error.num_cols)
    biasTensorDesc.SetTensor4dDescriptor(TensorFormat, DataType, 1, bias.num_rows, 1, bias.num_cols)
    
    cudnn.ConvolutionBackwardBias(alpha/floatType bias.num_cols,dstTensorDesc,error.dArray,0.0f,biasTensorDesc,bias.dArray)
    
// coef_x = scale
// coef_y = location
// y does not get used.
let randMapModule = DeviceBinaryCoefTransformModule "coef_x*(x-0.5f)+coef_y;"

type dMatrix with
    static member createRandomUniformMatrix weights_num_rows weights_num_cols (scaling_factor : floatType) location =
        let weights_total_size = weights_num_rows*weights_num_cols
        
        let cudaBuffer = new_dev<floatType> weights_total_size
        cudaRandom.GenerateUniform(cudaBuffer)
        //str.Synchronize()

        // 2.0f*scaling_factor ensures that it is rescaled around zero if the scaling_factor is 1.0f.
        randMapModule.A(2.0f*scaling_factor,cudaBuffer,location,cudaBuffer,cudaBuffer)

        new dMatrix(weights_num_rows,weights_num_cols,cudaBuffer)

    member t.fillRandomUniformMatrix (scaling_factor : floatType) location =
        let weights_total_size = t.num_rows*t.num_cols

        cudaRandom.GenerateUniform(t.dArray)
        // 2.0f*scaling_factor ensures that it is rescaled around zero if the scaling_factor is 1.0f.
        randMapModule.A(2.0f*scaling_factor,t.dArray,location,t.dArray,t.dArray)

type Df_rec = {
    P: floatType ref
    A : floatType ref
    is_constant : bool
    } with

    static member create P =
        {P=P;A=ref 0.0f;is_constant=false}
    static member createConstant P =
        {P=P;A=ref 0.0f;is_constant=true}

type DM_rec = {
    P: dMatrix ref
    A : dMatrix ref
    is_constant : bool
    } with

    static member create (P: dMatrix) =
        {P=ref P;A=ref <| P.zeroLike();is_constant=false}
        
    static member createConstant (P: dMatrix) =
        {P=ref P;A=ref empty;is_constant=true}

    static member createEmpty () =
        {P=ref empty;A=ref empty;is_constant=false}
        
    static member createEmptyConstant () =
        {P=ref empty;A=ref empty;is_constant=true}

    /// Resizes the primal and the adjoint if they are below nr*nc in size.
    member t.Resize nr nc =
        let p = t.P
        let a = t.A
        if (!p).Size < nr*nc then
            p := (!p).Replace nr nc
            a := (!a).Replace nr nc
        

type Rf =
    | DfR_Df_DM of Df_rec * (unit -> unit) * (unit -> unit) * RDM
    | DfR_Df_Dfseq of Df_rec * (unit -> unit) * (unit -> unit) * Rf []

    member t.r =
        match t with
        | DfR_Df_DM(x,_,_,_) -> x
        | DfR_Df_Dfseq(x,_,_,_) -> x

    member t.triggerForward() =
        match t with
        | DfR_Df_DM(x,ff,_,_) -> ff()
        | DfR_Df_Dfseq(x,ff,_,_) -> ff()

    member t.triggerBackward() =
        match t with
        | DfR_Df_DM(x,_,fb,_) -> fb()
        | DfR_Df_Dfseq(x,_,fb,_) -> fb()

and RDM = 
    | DM of DM_rec
    | DMR of DM_rec * ff: (unit -> unit) * db: (unit -> unit) * nodes: RDM[]
    // Looking at it now, I just realized that all these types (except DM) are redundant and could be replaced with DMRlin.

    member t.r =
        match t with
        | DM x -> x
        | DMR(x,_,_,_) -> x

    member t.triggerForward() =
        match t with
        | DM _ -> ()
        | DMR(x,ff,_,_) -> ff()

    member t.triggerBackward() =
        match t with
        | DM _ -> ()
        | DMR(_,_,fb,_) -> fb()

    static member makeNode(hidden_size, input_size) =
        let p = new dMatrix(hidden_size,input_size)
        DM (DM_rec.create p)

    static member makeNode(hidden_size, input_size, input: floatType[]) =
        if hidden_size*input_size <> input.Length then failwith "hidden_size*input_size <> input.Length in makeNode."
        let p = new dMatrix(hidden_size,input_size, input)
        DM (DM_rec.create p)

    static member makeConstantNode(hidden_size, input_size, input: floatType[]) =
        if hidden_size*input_size <> input.Length then failwith "hidden_size*input_size <> input.Length in makeConstantNode."
        let p = new dMatrix(hidden_size,input_size, input)
        DM (DM_rec.createConstant p)

    static member makeUniformRandomNode(hidden_size,input_size) =
        let scale = (1.0f / sqrt(hidden_size+input_size |> floatType))
        let p = dMatrix.createRandomUniformMatrix hidden_size input_size scale 0.0f
        DM (DM_rec.create p)

// The type for the tape.
type R = 
    | Rf of Rf 
    | RDM of RDM
    
    member t.resetAdjoint() =
        match t with
        | Rf x -> x.r.A := 0.0f
        | RDM x -> (!x.r.A).setZero()

    member t.resetPrimal() =
        match t with
        | Rf x -> x.r.P := 0.0f
        | RDM x -> (!x.r.P).setZero()

    member t.triggerForward() =
        match t with
        | Rf x -> x.triggerForward()
        | RDM x -> x.triggerForward()

    member t.triggerBackward() =
        match t with
        | Rf x -> x.triggerBackward()
        | RDM x -> x.triggerBackward()

type tapeType = System.Collections.Generic.List<R>
let mutable tape = tapeType(1000)

let hadamaradMultiplicationModule = new DeviceBinaryTransformModule "x*y;"
let hadamaradMultiplicationErrorModule = new DeviceTrinaryTransformModule "x*y+z;"
let hadmult (a: RDM) (b: RDM) =
    let va = a.r.P
    let vb = b.r.P
    let el = a.r.A
    let er = b.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr, nc = (!va).rc
        node.Resize nr nc
        hadamaradMultiplicationModule.A(!va, !vb, !c)
    let fb () = 
        hadamaradMultiplicationErrorModule.A(!vb,!error,!el,!el)
        hadamaradMultiplicationErrorModule.A(!va,!error,!er,!er)
    let t = DMR(node,ff,fb,[|a;b|])
    tape.Add(RDM t)
    t


/// This is an optimization of the linear layer because the best optimization is to remove operations entirely.
/// Doing it standardly involves too many unnecessary allocations.
/// Can be used for both matrix-matrix standards and Hadamarad multiplications, but it is not intended that they be used at the same time.
/// It might be good to split this function for that reason.
let linear_layer (mm: (RDM*RDM) []) (hads: (RDM*RDM) []) (bias: RDM option) =
    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff() =
        if mm.Length > 0 then 
            let l,r = mm.[0]
            let nr = (!l.r.P).num_rows
            let nc = (!r.r.P).num_cols
            node.Resize nr nc
        else if hads.Length > 0 then
            let l,r = hads.[0]
            let nr,nc = (!l.r.P).rc
            node.Resize nr nc
        else failwith "Invalid input into linear_layer."

        match bias with
        | Some bias -> addBias2 0.0f !c 1.0f !bias.r.P
        | None -> (!c).setZero()
                
        for l,r in mm do sgemm2 nT nT 1.0f !l.r.P !r.r.P 1.0f !c
        for l,r in hads do hadamaradMultiplicationErrorModule.A(!l.r.P, !r.r.P, !c, !c)

    let fb() =
        for l,r in mm do
            if l.r.is_constant = false then sgemm2 nT T 1.0f !error !r.r.P 1.0f !l.r.A
            if r.r.is_constant = false then sgemm2 T nT 1.0f !l.r.P !error 1.0f !r.r.A
        for l,r in hads do 
            hadamaradMultiplicationErrorModule.A(!error, !r.r.P, !l.r.A, !l.r.A)
            hadamaradMultiplicationErrorModule.A(!l.r.P, !error, !r.r.A, !r.r.A)
        match bias with
        | Some bias -> calculateBias2 1.0f !error !bias.r.A
        | None -> ()
    let ar =
        [|
        for l,r in mm do yield l; yield r
        for l,r in hads do yield l; yield r
        match bias with
        | Some bias -> yield bias 
        | None -> () |]
    let t = DMR(node,ff,fb,ar)
    tape.Add(RDM t)
    t

let matmult (a: RDM) (b:RDM) =
    let va = a.r.P
    let vb = b.r.P
    let el = a.r.A
    let er = b.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A
        
    let ff () = 
        let nr = (!va).num_rows
        let nc = (!vb).num_cols
        node.Resize nr nc
        sgemm2 nT nT 1.0f !va !vb 0.0f !c
    let fb () = 
        if a.r.is_constant = false then sgemm2 nT T 1.0f !error !vb 1.0f !el// The derivative with respect to the left. So the above argument gets inserted from the right left. Usually error * input.
        if b.r.is_constant = false then sgemm2 T nT 1.0f !va !error 1.0f !er// The derivative with respect to the right. So the above argument gets inserted from the right side. Usually weights * error.
    let t = DMR(node,ff,fb,[|a;b|])
    tape.Add(RDM t)
    t

/// Addition with broadcasting.
let addb (a: RDM) (b: RDM) = // b is for bias and a is for preactivations.
    let va = a.r.P
    let vb = b.r.P
    let el = a.r.A
    let er = b.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        sgeam2 nT nT 1.0f !va 0.0f !c !c
        addBias2 1.0f !c 1.0f !vb
    let fb () = 
        sgeam2 nT nT 1.0f !el 1.0f !error !el
        calculateBias2 1.0f !error !er
    let t = DMR(node,ff,fb,[|a;b|])
    tape.Add(RDM t)
    t

let sigmoidModule = new DeviceUnaryTransformModule "1.0f/(1.0f+expf(-x));"
//y = error
//z = previous adjoint value
let sigmoidErrorModule = new DeviceTrinaryTransformModule "x*(1.0f-x)*y + z;"
let sigmoid (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        sigmoidModule.A(!va,!c)
    let fb () = sigmoidErrorModule.A(!c,!error,!el,!el)
    let t = DMR(node,ff,fb,[|a|])
    tape.Add(RDM t)
    t

let tanhModule = new DeviceUnaryTransformModule "tanhf(x);"
//y = error
//z = previous adjoint value
let tanhErrorModule = new DeviceTrinaryTransformModule "(1.0f-x*x)*y + z;"
let tanh_ (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        tanhModule.A(!va,!c)
    let fb () = tanhErrorModule.A(!c,!error,!el,!el)
    let t = DMR(node,ff,fb,[|a|])
    tape.Add(RDM t)
    t

let add alpha (a: RDM) beta (b: RDM) =
    let va = a.r.P
    let vb = b.r.P
    let el = a.r.A
    let er = b.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        sgeam2 nT nT alpha !va beta !vb !c
    let fb () = 
        let nr,nc = (!va).rc
        if (a.r.is_constant) = false then sgeam2 nT nT alpha !error 1.0f !el !el
        if (b.r.is_constant) = false then sgeam2 nT nT 1.0f !er beta !error !er
    let t = DMR(node,ff,fb,[|a;b|])
    tape.Add(RDM t)
    t

(*
/// The old inneficient linear layer that just does everything as a sequence of matrix multiplication operation. For debugging purposes.
let linear_layer_ (mm: (RDM*RDM) []) (hh: (RDM*RDM) []) (bias: RDM option) =
    let mats = [|for l,r in mm do yield matmult l r|]
    let hads = [|for l,r in hh do yield hadmult l r|]
    let t = [|mats;hads|] |> Array.concat
    let sum = Array.fold (fun state x -> add 1.0f state 1.0f x) t.[0] t.[1..]
    match bias with
    | Some bias -> addb sum bias
    | None -> sum
*)

let squareModule = new DeviceUnaryTransformModule "x*x;"
//y = error
//z = previous adjoint value
let squareErrorModule = new DeviceTrinaryTransformModule "2.0f*x*y + z;"
let square (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        squareModule.A(!va,!c)
    let fb () = squareErrorModule.A(!va,!error,!el,!el)
    let t = DMR(node,ff,fb,[|a|])
    tape.Add(RDM t)
    t

let sumModule = new DeviceUnaryMapSumModule "x;"
let sumErrorModule = new DeviceUnaryCoefTransformModule "coef_x + x;" 
let sum (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let node = Df_rec.create (ref 0.0f)
    let error = node.A
    
    let ff () = node.P := sumModule.A(!va)
    let fb () = sumErrorModule.A(!error,!el,!el)
    let t = DfR_Df_DM(node,ff,fb,a)
    tape.Add(Rf t)
    t

let scale (alpha: floatType) (a:Rf) =
    let node = Df_rec.create (ref 0.0f)

    let ff () = node.P := alpha * !a.r.P
    let fb () = a.r.A := alpha * !node.A + !a.r.A
    let t = DfR_Df_Dfseq(node,ff,fb,[|a|])
    tape.Add(Rf t)
    t

let sum_scalars (a:Rf[]) =

    let node = Df_rec.create (ref 0.0f)

    let ff () = 
        let c = ref 0.0f
        for l in a do c := !c + !l.r.P
        node.P := !c
    let fb () = 
        for l in a do l.r.A := !node.A + !l.r.A
    let t = DfR_Df_Dfseq(node,ff,fb,a)
    tape.Add(Rf t)
    t

let logModule = new DeviceUnaryTransformModule "logf(x);"
//y=error
//z=previous adjoint
let logErrorModule = new DeviceTrinaryTransformModule "y / x + z;"
let log_ (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        logModule.A(!va,!c)
    let fb () = logErrorModule.A(!va,!error, !el, !el)
    let t = DMR(node,ff,fb,[|a|])
    tape.Add(RDM t)
    t

//coef_x = scalar
//coef_y = coef
let scalarMatrixAddModule = new DeviceBinaryCoefTransformModule "coef_x + coef_y*x;"
let scalar_matrix_add scalar coef (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        scalarMatrixAddModule.A(scalar,!va,coef,!va,!c)
    let fb () = sgeam2 nT nT coef !error 1.0f !el !el
    let t = DMR(node,ff,fb,[|a|])
    tape.Add(RDM t)
    t

//coef_x=scalar
let scalarAddModule = new DeviceUnaryCoefTransformModule "coef_x + x;"
let scalar_add (a:RDM) b =
    let va = a.r.P
    let el = a.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        scalarAddModule.A(b,!va,!c)
    let fb () = sgeam2 nT nT 1.0f !error 1.0f !el !el
    let t = DMR(node,ff,fb,[|a|])
    tape.Add(RDM t)
    t

let neg (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let node = DM_rec.createEmpty()
    let c = node.P
    let error = node.A

    let ff () = 
        let nr,nc = (!va).rc
        node.Resize nr nc
        sgeam2 nT nT -1.0f !va 0.0f !va !c
    let fb () = sgeam2 nT nT -1.0f !error 1.0f !el !el
    let t = DMR(node,ff,fb,[|a|])
    tape.Add(RDM t)
    t

let cross_entropy_cost target activations =
    let cross_ent = linear_layer [||] [|target,log_ activations;scalar_matrix_add 1.0f -1.0f target, log_ (scalar_matrix_add 1.0f -1.0f activations)|] None
    let s = sum cross_ent
    scale (-1.0f/floatType (!target.r.P).num_cols) s

let squared_error_cost target activations =
    let r1 = add 1.0f target -1.0f activations
    let r2 = square r1
    let r3 = sum r2
    scale (0.5f/floatType (!target.r.P).num_cols) r3

// A recurrent layer of neurons
type Layer =
    {
    W:RDM  // Input weight matrix
    U:RDM  // Recurrent weight matrix
    b:RDM  // Bias vector
    a:RDM->RDM
    } with     // Activation function
     
    member l.ToArray = 
        [|l.W;l.U;l.b|]

    static member fromArray (a : RDM[]) act =
        {
         W = a.[0]
         U = a.[1]
         b = a.[2]
         a = act
        }

    static member createRandomLayer hidden_size input_size act =
        {
         W = RDM.makeUniformRandomNode(hidden_size, input_size)
         U = RDM.makeUniformRandomNode(hidden_size, hidden_size)
         b = RDM.makeUniformRandomNode(hidden_size, 1)
         a = act
        } 

    // For the section with no previous hidden state.
    member l.runLayerNoH (x:RDM) =
        linear_layer [|l.W,x|] [||] (Some l.b) |> l.a
    
    // For the section with no input
    member l.runLayerNoI (y:RDM) =
        linear_layer [|l.U,y|] [||] (Some l.b) |> l.a

    // For the section with previous hidden state
    member l.runLayer (x:RDM) (y:RDM) =
        linear_layer [|l.W,x;l.U,y|] [||] (Some l.b) |> l.a


type GRULayer =
    {W_u:RDM  // Input weight matrix for the update gate
     U_u:RDM  // Recurrent weight matrix for the update gate
     b_u:RDM  // Bias vector for the update gate

     W_r:RDM  // Input weight matrix for the reset gate
     U_r:RDM  // Recurrent weight matrix for the reset gate
     b_r:RDM  // Bias vector for the reset gate

     W_n:RDM  // Input weight matrix for the potential new state
     U_n:RDM  // Recurrent weight matrix for the potential new state
     b_n:RDM  // Bias vector for the potential new state

     a : RDM -> RDM
     } with
    
    /// Returns all the weights in an array.
    member l.ToArray =
        [|l.W_u;l.U_u;l.b_u;l.W_r;l.U_r;l.b_r;l.W_n;l.U_n;l.b_n|]

    static member createRandomGRULayer hidden_size input_size act =
        {
        W_u = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_u = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_u = RDM.makeUniformRandomNode(hidden_size, 1)

        W_r = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_r = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_r = RDM.makeUniformRandomNode(hidden_size, 1)

        W_n = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_n = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_n = RDM.makeUniformRandomNode(hidden_size, 1)

        a = act
        }

    // For the section with no previous hidden state.
    member l.runLayerNoH (x:RDM) =
        let update_gate = linear_layer [|l.W_u,x|] [||] (Some l.b_u) |> sigmoid
        let potential_new_state = linear_layer [|l.W_n,x|] [||] (Some l.b_n) |> l.a
        let output_b = hadmult (scalar_matrix_add 1.0f -1.0f update_gate) potential_new_state
        output_b
    
    // For the section with no input
    member l.runLayerNoI (y:RDM) =
        let update_gate = linear_layer [|l.U_u,y|] [||] (Some l.b_u) |> sigmoid
        let reset_gate = linear_layer [|l.U_r,y|] [||] (Some l.b_r) |> sigmoid
        let potential_new_state = linear_layer [|l.U_n, (hadmult reset_gate y)|] [||] (Some l.b_n) |> l.a
        linear_layer [||] [|update_gate,y;(scalar_matrix_add 1.0f -1.0f update_gate),potential_new_state|] None

    // For the section with previous hidden state
    member l.runLayer (x:RDM) (y:RDM) =
        let update_gate = linear_layer [|l.W_u,x;l.U_u,y|] [||] (Some l.b_u) |> sigmoid
        let reset_gate = linear_layer [|l.W_r,x;l.U_r,y|] [||] (Some l.b_r) |> sigmoid
        let potential_new_state = linear_layer [|l.W_n,x;l.U_n, (hadmult reset_gate y)|] [||] (Some l.b_n) |> l.a
        linear_layer [||] [|update_gate,y;(scalar_matrix_add 1.0f -1.0f update_gate),potential_new_state|] None

let forwardpropTape (tape: Generic.List<R>) = for i=0 to tape.Count-1 do tape.[i].triggerForward()
let reversepropTape (tape: Generic.List<R>) = for i=tape.Count-1 downto 0 do tape.[i].triggerBackward()
let resetTapeAdjoint (tape: Generic.List<R>) = for x in tape do x.resetAdjoint()
let resetTapePrimal (tape: Generic.List<R>) = for x in tape do x.resetPrimal()

let add_gradients_to_weights (base_nodes: RDM[]) learning_rate clip_coef = 
    for x in base_nodes do 
        gradclipModule.A(clip_coef,!x.r.A,!x.r.A)
        sgeam2 nT nT 1.0f !x.r.P -learning_rate !x.r.A !x.r.P

let nesterov_add_gradients (base_nodes: RDM[]) (momentum_matrices: dMatrix[]) (copy_weights: dMatrix[]) learning_rate momentum_rate clip_coef = 
    for i=0 to base_nodes.Length-1 do
        let x = base_nodes.[i] 
        let m = momentum_matrices.[i]
        let c = copy_weights.[i]
        gradclipModule.A(clip_coef,!x.r.A,!x.r.A)
        sgeam2 nT nT -learning_rate !x.r.A momentum_rate m m // Add the gradients to the momentum matrices
        sgeam2 nT nT 1.0f m 1.0f c c // Add momentum to the copy matrix
        sgeam2 nT nT 1.0f c momentum_rate m !x.r.P // Apply Nesterov's momentum to the weights. It is really the copy weights that serve as the basis.

let save_data filename (ar: RDM []) =
    let stream_data = File.OpenWrite(filename)
    let writer_data = new BinaryWriter(stream_data)

    // Magic number
    writer_data.Write(929856)

    writer_data.Write(ar.Length)
    for x in ar do
        writer_data.Write((!x.r.P).num_rows)
        writer_data.Write((!x.r.P).num_cols)
        let t = (!x.r.P).dArray.Gather()
        for f in t do writer_data.Write(f)

    writer_data.Close()
    stream_data.Close()

let load_data file_name is_constant =
    let stream_data = File.OpenRead(file_name)
    let reader_data = new BinaryReader(stream_data)

    let m = reader_data.ReadInt32()
    if m <> 929856 then failwith "Wrong file type in load_weights"

    let l = reader_data.ReadInt32()
    let weights = [|
        for i=1 to l do
            let num_rows = reader_data.ReadInt32()
            let num_cols = reader_data.ReadInt32()
            let ar = [|for x=1 to num_rows*num_cols do yield reader_data.ReadSingle()|]
            match is_constant with
            | true -> yield RDM.makeConstantNode(num_rows,num_cols,ar)
            | false -> yield RDM.makeNode(num_rows,num_cols,ar)
        |]

    reader_data.Close()
    stream_data.Close()
    weights

type LSTMLayer =
    {W_z:RDM  // Input weight matrix for the block input
     U_z:RDM  // Recurrent weight matrix for the block input
     b_z:RDM  // Bias vector for the block input

     W_i:RDM  // Input weight matrix for the input gate
     U_i:RDM  // Recurrent weight matrix for the input gate
     b_i:RDM  // Bias vector for the input gate
     P_i:RDM  // Peephole weight matrix for the input gate

     W_f:RDM  // Input weight matrix for the forget gate
     U_f:RDM  // Recurrent weight matrix for the forget gate
     b_f:RDM  // Bias vector for the forget gate
     P_f:RDM  // Peephole weight matrix for the forget gate

     W_o:RDM  // Input weight matrix for the output gate
     U_o:RDM  // Recurrent weight matrix for the output gate
     b_o:RDM  // Bias vector for the output gate
     P_o:RDM  // Peephole weight matrix for the output gate

     block_input_a : RDM -> RDM
     block_output_a : RDM -> RDM
     } with
    
    /// Returns all the weights in an array.
    member l.ToArray = [|l.W_z;l.U_z;l.b_z;l.W_i;l.U_i;l.b_i;l.P_i;l.W_f;l.U_f;l.b_f;l.P_f;l.W_o;l.U_o;l.b_o;l.P_o|]
    static member fromArray (a: RDM[]) block_input_a block_output_a =
        {
         W_z = a.[0]
         U_z = a.[1]
         b_z = a.[2]

         W_i = a.[3]
         U_i = a.[4]
         b_i = a.[5]
         P_i = a.[6]

         W_f = a.[7]
         U_f = a.[8]
         b_f = a.[9]
         P_f = a.[10]

         W_o = a.[11]
         U_o = a.[12]
         b_o = a.[13]
         P_o = a.[14]

         block_input_a = block_input_a
         block_output_a = block_output_a
        }

    static member createRandomLSTMLayer hidden_size input_size block_input_a block_output_a =
        {
        W_z = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_z = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_z = RDM.makeUniformRandomNode(hidden_size, 1)

        W_i = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_i = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_i = RDM.makeUniformRandomNode(hidden_size, 1)
        P_i = RDM.makeUniformRandomNode(hidden_size, hidden_size)

        W_f = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_f = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_f = RDM.makeUniformRandomNode(hidden_size, 1)
        P_f = RDM.makeUniformRandomNode(hidden_size, hidden_size)

        W_o = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_o = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_o = RDM.makeUniformRandomNode(hidden_size, 1)
        P_o = RDM.makeUniformRandomNode(hidden_size, hidden_size)

        block_input_a = block_input_a
        block_output_a = block_output_a
        }

    member l.runLayer (x:RDM) (y:RDM) (c:RDM) =
        let block_input = linear_layer [|l.W_z,x;l.U_z,y|] [||] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer [|l.W_i,x;l.U_i,y;l.P_i,c|] [||] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer [|l.W_f,x;l.U_f,y;l.P_f,c|] [||] (Some l.b_f) |> sigmoid
        let c' = linear_layer [||] [|block_input,input_gate;c,forget_gate|] None
        let output_gate = linear_layer [|l.W_o,x;l.U_o,y;l.P_o,c'|] [||] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'

    member l.runLayerNoH (x:RDM) =
        let block_input = linear_layer [|l.W_z,x|] [||] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer [|l.W_i,x|] [||] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer [|l.W_f,x|] [||] (Some l.b_f) |> sigmoid
        let c' = hadmult block_input input_gate
        let output_gate = linear_layer [|l.W_o,x;l.P_o,c'|] [||] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'

    member l.runLayerNoI (y:RDM) (c:RDM) =
        let block_input = linear_layer [|l.U_z,y|] [||] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer [|l.U_i,y;l.P_i,c|] [||] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer [|l.U_f,y;l.P_f,c|] [||] (Some l.b_f) |> sigmoid
        let c' = linear_layer [||] [|block_input,input_gate;c,forget_gate|] None
        let output_gate = linear_layer [|l.U_o,y;l.P_o,c'|] [||] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'

type DeviceGetSliceModule() = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __global__ void getSliceKernel(const "+FloatTypeCpp+"* matrix, "+FloatTypeCpp+"* out_matrix, const int start_row, const int end_row, const int num_rows, const int start_col, const int end_col, const int num_cols, const unsigned col_major){
                const int stride = blockDim.x * gridDim.x;
                if (col_major){
                    int i = threadIdx.x+blockIdx.x*blockDim.x;
                    const int row_stride = end_row-start_row+1;
                    const int col_stride = end_col-start_col+1;
                    while (1) {
                        const int row_i = i % row_stride;
                        const int col_i = i / row_stride;
                        const int row = start_row+row_i;
                        const int col = start_col+col_i;
                        const int idx = row+col*num_rows;
                        if (row_i < row_stride && col_i < col_stride) {
                            out_matrix[i] = matrix[idx];
                            i += stride;
                        } else return;
                    }
                }
                else{
                    int i = threadIdx.x+blockIdx.x*blockDim.x;
                    const int row_stride = end_row-start_row+1;
                    const int col_stride = end_col-start_col+1;
                    while (1) {
                        const int row_i = i / col_stride;
                        const int col_i = i % col_stride;
                        const int row = start_row+row_i;
                        const int col = start_col+col_i;
                        const int idx = col+row*num_cols;
                        if (row_i < row_stride && col_i < col_stride) {
                            out_matrix[i] = matrix[idx];
                            i += stride;
                        } else return;
                    }
                }
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"getSliceKernel")
    do  
        try k.Compile([|"-arch=compute_30"|])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"getSliceKernel")

    /// For matrices stored in row major order.
    /// Zero based indexing.
    member t.AR(x: dMatrix, start_row, end_row, start_col, end_col) =
        if (start_row < 0 || start_col < 0) then failwith "start_row < 0 || start_col < 0"
        if (end_row >= x.num_rows || start_col >= x.num_cols) then failwith "end_row >= x.num_rows || start_col >= x.num_cols"
        let order = 0u
        let row_stride = end_row-start_row+1
        let col_stride = end_col-start_col+1
        let y = new dMatrix(row_stride, col_stride)
        let n = row_stride*col_stride
        let gridSize = divup n block_size
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.dArray.DevicePointer,y.dArray.DevicePointer,start_row, end_row, x.num_rows, start_col, end_col, x.num_cols, order) |> ignore
        y

    /// For matrices stored in column major order.
    /// Zero based indexing.
    member t.AC(x: dMatrix, start_row, end_row, start_col, end_col) =
        if (start_row < 0 || start_col < 0) then failwith "start_row < 0 || start_col < 0"
        if (end_row >= x.num_rows || start_col >= x.num_cols) then failwith "end_row >= x.num_rows || start_col >= x.num_cols"
        let order = 1u
        let row_stride = end_row-start_row+1
        let col_stride = end_col-start_col+1
        let y = new dMatrix(row_stride, col_stride)
        let n = row_stride*col_stride
        let gridSize = divup n block_size
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.dArray.DevicePointer,y.dArray.DevicePointer,start_row, end_row, x.num_rows, start_col, end_col, x.num_cols, order) |> ignore
        y

type DeviceSetSliceModule() = 
    let block_size = 256

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __global__ void setSliceKernel("+FloatTypeCpp+"* matrix, const "+FloatTypeCpp+"* out_matrix, const int start_row, const int end_row, const int num_rows, const int start_col, const int end_col, const int num_cols, const unsigned col_major){
                const int stride = blockDim.x * gridDim.x;
                if (col_major){
                    int i = threadIdx.x+blockIdx.x*blockDim.x;
                    const int row_stride = end_row-start_row+1;
                    const int col_stride = end_col-start_col+1;
                    while (1) {
                        const int row_i = i % row_stride;
                        const int col_i = i / row_stride;
                        const int row = start_row+row_i;
                        const int col = start_col+col_i;
                        const int idx = row+col*num_rows;
                        if (row_i < row_stride && col_i < col_stride) {
                            matrix[idx] = out_matrix[i];
                            i += stride;
                        } else return;
                    }
                }
                else{
                    int i = threadIdx.x+blockIdx.x*blockDim.x;
                    const int row_stride = end_row-start_row+1;
                    const int col_stride = end_col-start_col+1;
                    while (1) {
                        const int row_i = i / col_stride;
                        const int col_i = i % col_stride;
                        const int row = start_row+row_i;
                        const int col = start_col+col_i;
                        const int idx = col+row*num_cols;
                        if (row_i < row_stride && col_i < col_stride) {
                            matrix[idx] = out_matrix[i];
                            i += stride;
                        } else return;
                    }
                }
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"setSliceKernel")
    do  
        try k.Compile([|"-arch=compute_30"|])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"setSliceKernel")

    /// For matrices stored in row major order.
    /// Zero based indexing.
    member t.AR(x: dMatrix, y: dMatrix, start_row, end_row, start_col, end_col) =
        if (start_row < 0 || start_col < 0) then failwith "start_row < 0 || start_col < 0"
        if (end_row >= x.num_rows || start_col >= x.num_cols) then failwith "end_row >= x.num_rows || start_col >= x.num_cols"
        let order = 0u
        let row_stride = end_row-start_row+1
        let col_stride = end_col-start_col+1
        let n = row_stride*col_stride
        let gridSize = divup n block_size
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.dArray.DevicePointer,y.dArray.DevicePointer,start_row, end_row, x.num_rows, start_col, end_col, x.num_cols, order) |> ignore

    /// For matrices stored in column major order.
    /// Zero based indexing.
    member t.AC(x: dMatrix, y: dMatrix, start_row, end_row, start_col, end_col) =
        if (start_row < 0 || start_col < 0) then failwith "start_row < 0 || start_col < 0"
        if (end_row >= x.num_rows || start_col >= x.num_cols) then failwith "end_row >= x.num_rows || start_col >= x.num_cols"
        let order = 1u
        let row_stride = end_row-start_row+1
        let col_stride = end_col-start_col+1
        let n = row_stride*col_stride
        let gridSize = divup n block_size
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, x.dArray.DevicePointer,y.dArray.DevicePointer,start_row, end_row, x.num_rows, start_col, end_col, x.num_cols, order) |> ignore

// The Item and GetSlice operators. Row major.

let getsliceModule = DeviceGetSliceModule()

type dMatrix with
    member t.Item
        with get(a: int, b: int) = t.dArray.[b+a*t.num_cols |> SizeT]
        and set(a: int, b: int) (value: floatType) = t.dArray.[b+a*t.num_cols |> SizeT] <- value
    member t.setItem(a: int, b: int) (value: floatType) = t.dArray.[b+a*t.num_cols |> SizeT] <- value

    member t.GetSlice(rowStart: int option, rowFinish : int option,
                         colStart: int option, colFinish : int option) =
        let rowStart = defaultArg rowStart 0
        let rowFinish = defaultArg rowFinish (t.num_rows-1)
        let colStart = defaultArg colStart 0
        let colFinish = defaultArg colFinish (t.num_cols-1)
        getsliceModule.AR(t,rowStart,rowFinish,colStart,colFinish)

    member t.GetSlice(row: int, colStart: int option, colFinish: int option) =
            let colStart = defaultArg colStart 0
            let colFinish = defaultArg colFinish t.num_cols-1
            getsliceModule.AR(t,row,row,colStart,colFinish)

    member t.GetSlice(rowStart: int option, rowFinish: int option, col: int) =
            let rowStart = defaultArg rowStart 0
            let rowFinish = defaultArg rowFinish t.num_rows-1
            getsliceModule.AR(t,rowStart,rowFinish,col,col)

    /// For displaying column majors matrices inside Array2D (which is row major.)
    member inline t.Gather'() =
            let h_a = Array2D.zeroCreate<floatType> t.num_rows t.num_cols
            use t' = sgeam T T 1.0f t 0.0f t // Transpose
            t'.dArray.CopyToHost(h_a)
            h_a

