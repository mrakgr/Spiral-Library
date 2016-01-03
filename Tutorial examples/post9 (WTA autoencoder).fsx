// Spiral reverse AD example. Used for testing.
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
open System.Collections.Generic

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

//let l1 = FeedforwardLayer.fromArray (load_data (__SOURCE_DIRECTORY__ + @"\l1_weights.dat") false) (WTAT 6) \\ For loading the weights from file.
let l1 = FeedforwardLayer.createRandomLayer 1024 784 (WTA 6)
let l2 = InverseFeedforwardLayer.createRandomLayer l1 (fun x -> x) // No nonlinearity at the end. With a steep sigmoid the cost is much better, but the visualizations are less crisp.
let layers = [|l1 :> IFeedforwardLayer;l2 :> IFeedforwardLayer|] // Upcasting to the base type. The correct functions will get called with dynamic dispatch.

//save_data (__SOURCE_DIRECTORY__ + @"\l1_weights.dat") l1.ToArray // For saving the weights
// This does not actually train it, it just initiates the tree for later training.
// The autoencoder version.
let training_loop (data: DM) (layers: IFeedforwardLayer[]) =
    let outputs = Array.fold (fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data layers
    squared_error_cost data outputs 

let train_mnist_sgd num_iters learning_rate (layers: IFeedforwardLayer[]) =
    [|
    let mutable r' = 0.0f
    let base_nodes = layers |> Array.map (fun x -> x.ToArray) |> Array.concat // Stores all the base nodes of the layer so they can later be reset.
    for i=1 to num_iters do
        for x in dtrain do
            let data, target = x
            let r = training_loop data layers // Builds the tape.

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

        for x in dtest do
            let data, target = x
            let r = training_loop data layers // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 dtest.Length) // Adds the cost to the accumulator.

            if System.Single.IsNaN r' then failwith "Nan error"

            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed.

        printfn "The validation cost at iteration %i is %f" i r'
        let r2 = r'
        r' <- 0.0f
        yield r1,r2
    |]
let num_iters = 20
let learning_rate = 0.1f
#time
let s = train_mnist_sgd num_iters learning_rate layers
#time

let l = [|for l,_ in s do yield l|]
let r = [|for _,r in s do yield r|]

(Chart.Combine [|Chart.Line l;Chart.Line r|]).ShowChart()

let bitmap = make_bitmap_from_dmatrix l1.W.r.P 28 28 25 40
bitmap.Save(__SOURCE_DIRECTORY__ + @"\weights.png")

