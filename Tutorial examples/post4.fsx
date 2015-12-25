// The Spiral library v1. Basic reverse mode AD on the GPU.

#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/ManagedCuda.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/NVRTC.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaBlas.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaRand.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/NPP.dll"
#r "../packages/ManagedCuda-CudaDNN.3.0/lib/net45/CudaDNN.dll"

// Open up the namespaces.
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

// Initialize the context. Analogous to a CPU process. Cuda tries to offload as much as possible during context creation so there aren't
// any unexpected delays later.
let ctx = new CudaContext()
let numSm = ctx.GetDeviceInfo().MultiProcessorCount // The number of streaming multiprocessors on the device.

// Make a stream class.
let str = new CudaStream()
// Set the Cuda libraries handles to the above stream.
let cublas = CudaBlas(str.Stream)
let cudnn = new CudaDNN.CudaDNNContext()
cudnn.SetStream(str)
let cudaRandom = new CudaRand.CudaRandDevice(GeneratorType.PseudoDefault)
cudaRandom.SetStream(str.Stream)

// Type aliasing trick to make Spiral more generic. It is incomplete at the moment though due to Cuda math function being non-overloadable.
type floatType = float32
let inline floatType x = float32 x
let FloatTypeCpp = "float"

/// Copies a host array to device.
let inline to_dev (host_ar: 't []) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

/// Copies a device array to host.
let inline to_host (dev_ar: CudaDeviceVariable<'t>) =
    let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
    dev_ar.CopyToHost(h_a)
    h_a

/// Copies the device array to host. Extends the CudaDeviceVariable class.
type CudaDeviceVariable<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> with
    member inline this.Gather() =
        to_host this

/// Allocates a new device array without initializing it.
let inline new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
    new CudaDeviceVariable<'t>(SizeT n)

/// The main matrix type.
type dMatrix =
    {
    mutable num_rows:int
    mutable num_cols:int
    dArray: CudaDeviceVariable<floatType>
    }  

    /// The main create function. A substitute for the constructor.
    static member create(num_rows: int,num_cols,dArray: CudaDeviceVariable<floatType>) =
        {num_rows=num_rows;num_cols=num_cols;dArray=dArray}

    /// Throws an exception if it tries to allocate an array of size 0.
    static member create(num_rows: int,num_cols) =
        let q = (num_rows*num_cols) |> SizeT
        let t = new CudaDeviceVariable<floatType>(q)
        {num_rows=num_rows;num_cols=num_cols;dArray=t}

    /// Copies a host to a device array.
    /// Throws an exception if it tries to allocate an array of size 0.
    static member create(num_rows: int,num_cols,dArray: floatType[]) =
        let q = num_rows*num_cols
        if dArray.Length <> q then failwith "Invalid size in dMatrix construction."
        let t = to_dev dArray
        {num_rows=num_rows;num_cols=num_cols;dArray=t}

    /// Returns num_rows, num_cols as a tuple
    member inline t.rc = t.num_rows, t.num_cols
    /// Sets the matrix to zero.
    member inline t.setZero() = t.dArray.MemsetAsync(0u,str.Stream)
    /// Set the matrix to a value.
    member inline t.set (x: floatType) = 
        let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
        t.dArray.MemsetAsync(v,str.Stream)
    /// Creates a copy of this matrix with all the values set to zero.
    member inline t.zeroLike() =
        let c = dMatrix.create(t.num_rows,t.num_cols)
        c.setZero()
        c
    /// Copies a matrix.
    member inline t.copy() =
        let c = dMatrix.create(t.num_rows,t.num_cols)
        c.dArray.AsyncCopyToDevice(t.dArray,str)
        c

    // Note: The ReplaceIf function could be written in a more functional manner without the use of option types, but it would clog up the GC.
    /// Returns a new dMatrix if the current one is less than nr*nc. Otherwise it returns self with the num_rows and num_cols adjusted.
    member inline t.ReplaceIf nr nc =
        if int t.dArray.Size < nr*nc then
            (t :> IDisposable).Dispose()
            Some (dMatrix.create(nr,nc,new_dev<floatType> (nr*nc)))
        else
            t.num_rows <- nr
            t.num_cols <- nc
            None

    /// Copies a matrix to a host array.
    member inline t.Gather() =
        let h_a = Array.zeroCreate<floatType> (int t.dArray.Size)
        t.dArray.CopyToHost(h_a)
        h_a

    member inline t.isEmpty = t.dArray.Equals(CudaDeviceVariable.Null)

    override t.ToString() =
        sprintf "dM(%i,%i)" t.num_rows t.num_cols

    interface IDisposable with
        member t.Dispose() = 
            if t.isEmpty = false then
                t.dArray.Dispose()

    // Note: The native GC collector is not as sensitive to the device memory as it is to host memory.
    // It will not get triggered if the GPU is running low, so it necessary to be more careful than usual
    // when handling device memory.
    override t.Finalize() = (t :> IDisposable).Dispose()

/// An instance of an empty dMatrix.
let empty = dMatrix.create(0,0,CudaDeviceVariable.Null)



