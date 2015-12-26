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

let a = [|1.0f;2.0f;3.0f|]
let a' = to_dev a
a'.[SizeT 0] <- 5.0f // Annoyingly, it is necessary to add explicitly convert ints to SizeT when accessing individual items in the CudaDeviceVariable class.
let b = to_host a'