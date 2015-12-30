// Batched gemm test. Is significantly faster than gemm, but unfortunately cannot be used to when the targets are the same
// which would be good for LSTMs.

#load "ad_utils_spiral_v1.fsx"
open Ad_utils_spiral_v1

#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"

open FSharp.Charting

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

/// General matrix-matrix multiply from cuBLAS. Inplace version
let gemm2_batched transa transb (alpha: floatType) (A:dMatrix []) (B:dMatrix []) beta (C:dMatrix []) =
    if A.Length <> B.Length || B.Length <> C.Length then failwith "Array sizes must match"
    for i=1 to C.Length-1 do
        if A.[0].rc <> A.[i].rc then failwith "All the rows and cols of A must match"
        if B.[0].rc <> B.[i].rc then failwith "All the rows and cols of B must match"
        if C.[0].rc <> C.[i].rc then failwith "All the rows and cols of C must match"
    let a_col = if transa = nT then A.[0].num_cols else A.[0].num_rows
    let b_row = if transb = nT then B.[0].num_rows else B.[0].num_cols
    if a_col <> b_row then failwith (sprintf "a_col <> b_row in gemm! %i <> %i" a_col b_row)
    let m = if transa = nT then A.[0].num_rows else A.[0].num_cols
    let n = if transb = nT then B.[0].num_cols else B.[0].num_rows
    let k = a_col

    let lda = if transa = nT then m else k
    let ldb = if transb = nT then k else n
    let ldc = m

    if m <> C.[0].num_rows || n <> C.[0].num_cols then failwith "m <> C.num_rows || n <> C.num_cols in gemm2"

    use a_ptrs = new_dev A.Length
    use b_ptrs = new_dev B.Length
    use c_ptrs = new_dev C.Length
    for i=0 to C.Length-1 do
        let i' = SizeT i
        a_ptrs.[i'] <- A.[i].dArray.DevicePointer
        b_ptrs.[i'] <- B.[i].dArray.DevicePointer
        c_ptrs.[i'] <- C.[i].dArray.DevicePointer

    cublas.GemmBatched(transa,transb,m,n,k,alpha,a_ptrs,lda,b_ptrs,ldb,beta,c_ptrs,ldc,C.Length)

let a = Array.init 3 (fun _ -> dMatrix.createRandomUniformMatrix 128 128 1.0f 0.0f)
let a' = a |> Array.map(fun x -> x.dArray.DevicePointer)
let b = Array.init 3 (fun _ -> dMatrix.createRandomUniformMatrix 128 128 1.0f 0.0f)
let t = dMatrix.create (128, 128)
let c = [|t;t;t|]

let w = c |> Array.map(fun x -> x.Gather())
let w' = c |> Array.map(fun x -> x.Gather())

let diffs = Array.map2 (fun a b -> abs(a-b)) w.[1] w'.[1] |> Array.sum

#time
for i=1 to 1 do
    gemm2_batched nT nT 1.0f a b 0.0f c
    str.Synchronize()
#time
#time
for i=1 to 1 do
    for j=0 to 2 do
        gemm2 nT nT 1.0f a.[j] b.[j] 0.0f c.[j]
    str.Synchronize()
#time
