// Spiral reverse AD example. Used for testing.

#load "../Spiral Library/ad_utils_spiral_v1.fsx"
open Ad_utils_spiral_v1

#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"

open FSharp.Charting

#load "load_mnist.fsx"
open Load_mnist

// "__SOURCE_DIRECTORY__ + testSetData" gives parsing errors if it is written like "__SOURCE_DIRECTORY__+testSetData"
// __SOURCE_DIRECTORY__ is just a string literal refering to the directory where the script resides.
let train_data = make_imageset (__SOURCE_DIRECTORY__ + trainSetData) (__SOURCE_DIRECTORY__ + trainSetLabels) 
let test_data = make_imageset (__SOURCE_DIRECTORY__ + testSetData) (__SOURCE_DIRECTORY__ + testSetLabels)

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
    let ar_data = split_cols d_data batch_size
    let ar_label = split_cols d_label batch_size
    ar_data |> Array.map (fun x -> DM.makeConstantNode x), ar_label |> Array.map (fun x -> DM.makeConstantNode x)

let dtrain_data, dtrain_label = make_set train_data 128
let dtest_data, dtest_label = make_set test_data 128

let l1 = FeedforwardLayer.createRandomLayer 10 784 sigmoid
//let l2 = FeedforwardLayer.createRandomLayer 10 1024 sigmoid

let base_nodes = [|l1.ToArray|] |> Array.concat

// This does not actually train it, it just initiates the tree for later training.
let training_loop (data: DM) (targets: DM) (layers: FeedforwardLayer[]) =
    let outputs = layers |> Array.fold (fun state layer -> layer.runLayer state) data
    cross_entropy_cost targets outputs


let train_mnist_sgd num_iters learning_rate layers =
    [|
    let mutable r' = 0.0f
    for i=1 to num_iters do
        for i=0 to dtrain_data.Length-1 do
            let data, target = dtrain_data.[i], dtrain_label.[i]
            let r = training_loop data target layers

            tape.forwardpropTape 0
            r' <- r' + (!r.r.P/ float32 dtrain_data.Length)
            if System.Single.IsNaN r' then failwith "Nan error"

            for x in base_nodes do x.r.A.contents.setZero() // Resets the base adjoints
            tape.resetTapeAdjoint 0 // Resets the adjoints for the training select
            r.r.A := 1.0f // Pushes 1.0f from the top node
            tape.reversepropTape 0 // Resets the adjoints for the test select
            add_gradients_to_weights' base_nodes learning_rate // The optimization step
            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed.

        printfn "The training cost at iteration %i is %f" i r'
        let r1 = r'
        r' <- 0.0f

        for i=0 to dtest_data.Length-1 do
            let data, target = dtest_data.[i], dtest_label.[i]
            let r = training_loop data target layers

            tape.forwardpropTape 0
            r' <- r' + (!r.r.P/ float32 dtest_data.Length)
            if System.Single.IsNaN r' then failwith "Nan error"

            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed.

        printfn "The validation cost at iteration %i is %f" i r'
        let r2 = r'
        r' <- 0.0f
        yield r1,r2
    |]
let num_iters = 30
let learning_rate = 0.001f
#time
let s = train_mnist_sgd num_iters learning_rate [|l1|]
#time

let l = [|for l,_ in s do yield l|]
let r = [|for _,r in s do yield r|]

(Chart.Combine [|Chart.Line l;Chart.Line r|]).ShowChart()

tape.Select -1
for x in base_nodes do tape.Add x
tape.Select 0
tape.DisposeAll()