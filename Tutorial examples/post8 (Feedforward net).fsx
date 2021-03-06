﻿// Spiral reverse AD example. Used for testing.
#load "../Spiral Library/ad_utils_spiral_v1.fsx"
open Ad_utils_spiral_v1

#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"

open FSharp.Charting

#load "load_mnist.fsx"
open Load_mnist
open System

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

// "__SOURCE_DIRECTORY__ + testSetData" gives parsing errors if it is written like "__SOURCE_DIRECTORY__+testSetData"
// __SOURCE_DIRECTORY__ is just a string literal refering to the directory where the script resides.
let train_data = make_imageset (__SOURCE_DIRECTORY__ + trainSetData) (__SOURCE_DIRECTORY__ + trainSetLabels) 
let test_data = make_imageset (__SOURCE_DIRECTORY__ + testSetData) (__SOURCE_DIRECTORY__ + testSetLabels)

/// Returns a tuple of training set and label set split into minibatches.
/// Uses the GetSlice module.
let make_set (s : MnistImageset) batch_size =
    /// Function that splits the dataset along the columns.
    let split_cols (x:dMatrix) batch_size =
        [|
        for i=0 to (x.num_cols-1)/batch_size do
            let start_pos = i*batch_size
            let end_pos = min ((i+1)*batch_size-1) (x.num_cols-1)
            yield x.[*,start_pos..end_pos]
            |]
    use d_data = dMatrix.create(s.num_rows*s.num_cols,s.num_images,s.float_data) // Loads the data
    use d_label = dMatrix.create(10,s.num_images,s.float_labels) // Loads the labels
    let ar_data = split_cols d_data batch_size |> Array.map (fun x -> DM.makeConstantNode x)
    let ar_label = split_cols d_label batch_size |> Array.map (fun x -> DM.makeConstantNode x)
    Array.zip ar_data ar_label

// The type of each of these two variable is dMatrix [], dMatrix [] - a tuple.
let dtrain = make_set train_data 128
let dtest = make_set test_data 128 

let l1 = FeedforwardLayer.createRandomLayer 1024 784 relu
let l2 = FeedforwardLayer.createRandomLayer 2048 1024 relu
let l3 = FeedforwardLayer.createRandomLayer 1024 2048 relu
let l4 = FeedforwardLayer.createRandomLayer 10 1024 (clipped_steep_sigmoid 3.0f)
let layers = [|l1;l2;l3;l4|]

// This does not actually train it, it just initiates the tree for later training.
let training_loop' (data: DM) (targets: DM) (layers: FeedforwardLayer[]) =
    let outputs = layers |> Array.fold (fun state layer -> layer.runLayer state) data
    // I make the accuracy calculation lazy. This is similar to returning a lambda function that calculates the accuracy
    // although in this case it will be calculated at most once.
    lazy get_accuracy targets.r.P outputs.r.P, cross_entropy_cost targets outputs 

// This does not actually train it, it just initiates the tree for later training.
// Imperative form.
let training_loop (data: DM) (targets: DM) (layers: FeedforwardLayer[]) =
    let mutable outputs = data
    for x in layers do
        outputs <- x.runLayer outputs
    // I make the accuracy calculation lazy. This is similar to returning a lambda function that calculates the accuracy
    // although in this case it will be calculated at most once.
    lazy get_accuracy targets.r.P outputs.r.P, cross_entropy_cost targets outputs 

let train_mnist_sgd num_iters learning_rate (layers: FeedforwardLayer[]) =
    [|
    let mutable r' = 0.0f
    let base_nodes = layers |> Array.map (fun x -> x.ToArray) |> Array.concat // Stores all the base nodes of the layer so they can later be reset.
    for i=1 to num_iters do
        for x in dtrain do
            let data, target = x
            let _,r = training_loop data target layers // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 dtrain.Length) // Adds the cost to the accumulator.
            if System.Single.IsNaN r' then failwith "Nan error"

            for x in base_nodes do x.r.A.setZero() // Resets the base adjoints
            tape.resetTapeAdjoint 0 // Resets the adjoints for the training select
            r.r.A := 1.0f // Pushes 1.0f from the top node
            tape.reversepropTape 0 // Resets the adjoints for the test select
            add_gradients_to_weights' base_nodes learning_rate // The optimization step
            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed for the simple recurrent and feedforward case.

        printfn "The training cost at iteration %i is %f" i r'
        let r1 = r'
        r' <- 0.0f
        let mutable acc = 0.0f

        for x in dtest do
            let data, target = x
            let lazy_acc,r = training_loop data target layers // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 dtest.Length) // Adds the cost to the accumulator.
            acc <- acc+lazy_acc.Value // Here the accuracy calculation is triggered by accessing it through the Lazy property.

            if System.Single.IsNaN r' then failwith "Nan error"

            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed.

        printfn "The validation cost at iteration %i is %f" i r'
        printfn "The accuracy is %i/10000" (int acc)
        let r2 = r'
        r' <- 0.0f
        yield r1,r2
    |]
let num_iters = 5
let learning_rate = 0.1f
#time
let s = train_mnist_sgd num_iters learning_rate layers
#time

let l = [|for l,_ in s do yield l|]
let r = [|for _,r in s do yield r|]

(Chart.Combine [|Chart.Line l;Chart.Line r|]).ShowChart()



