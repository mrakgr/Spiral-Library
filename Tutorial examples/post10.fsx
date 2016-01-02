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

let l1 = FeedforwardLayer.createRandomLayer 1024 784 (WTA 6)
let l2 = FeedforwardLayer.createRandomLayer 1024 1024 (WTA 6)
let l3 = FeedforwardLayer.createRandomLayer 1024 1024 (WTA 6)
let l4 = InverseFeedforwardLayer.createRandomLayer l3 (fun x -> x) // No nonlinearity at the end. With a steep sigmoid the cost is much better, but the visualizations are less crisp.
let l5 = InverseFeedforwardLayer.createRandomLayer l2 (fun x -> x)
let l6 = InverseFeedforwardLayer.createRandomLayer l1 (fun x -> x)

let l1' = FeedforwardLayer.fromArray l1.ToArray relu // Makes supervised layers from the same weights.
let l2' = FeedforwardLayer.fromArray l2.ToArray relu
let l3' = FeedforwardLayer.fromArray l3.ToArray relu
let l_sig = FeedforwardLayer.createRandomLayer 10 1024 (clipped_steep_sigmoid 3.0f)

let layers_deep_autoencoder = [|[|l1;l2;l3|] |> Array.map (fun x -> x :> IFeedforwardLayer);[|l4;l5;l6|] |> Array.map (fun x -> x :> IFeedforwardLayer);|] |> Array.concat // Upcasting to the base type. The correct functions will get called with dynamic dispatch.
let layers_1 = [|[|l1|] |> Array.map (fun x -> x :> IFeedforwardLayer);[|l6|] |> Array.map (fun x -> x :> IFeedforwardLayer);|] |> Array.concat // Upcasting to the base type. The correct functions will get called with dynamic dispatch.
let layers_2 = [|[|l1;l2|] |> Array.map (fun x -> x :> IFeedforwardLayer);[|l5|] |> Array.map (fun x -> x :> IFeedforwardLayer);|] |> Array.concat // Upcasting to the base type. The correct functions will get called with dynamic dispatch.
let layers_3 = [|[|l1;l2;l3|] |> Array.map (fun x -> x :> IFeedforwardLayer);[|l4|] |> Array.map (fun x -> x :> IFeedforwardLayer);|] |> Array.concat // Upcasting to the base type. The correct functions will get called with dynamic dispatch.
let layers_fine_tune = [|l1';l2';l3';l_sig|] |> Array.map (fun x -> x :> IFeedforwardLayer)

let loop_1 data targets = // These loops are closures. I'll pass them into the function instead of the layers. This one is for the first autoencoder
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data layers_1 // Scan is like fold except it returns the intermediates.
    let inp = outputs.[outputs.Length-3]
    let out = outputs.[outputs.Length-1]
    squared_error_cost inp out, None

let loop_2 data targets = // The targets do nothing in autoencoders, they are here so the type for the supervised net squares out. This one is for the second.
    let l,r = layers_2 |> Array.splitAt 1
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data l // Scan is like fold except it returns the intermediates.
    tape.Add(BlockReverse()) // This blocks the reverse pass from running past this point. It is so the gradients get blocked and only the top two layers get trained.
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) (outputs |> Array.last) r // Scan is like fold except it returns the intermediates.
    let inp = outputs.[outputs.Length-3]
    let out = outputs.[outputs.Length-1]
    squared_error_cost inp out, None

let loop_3 data targets =
    let l,r = layers_3 |> Array.splitAt 2
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data l // Scan is like fold except it returns the intermediates.
    tape.Add(BlockReverse()) // This blocks the reverse pass from running past this point. It is so the gradients get blocked and only the top two layers get trained.
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) (outputs |> Array.last) r // Scan is like fold except it returns the intermediates.
    let inp = outputs.[outputs.Length-3]
    let out = outputs.[outputs.Length-1]
    squared_error_cost inp out, None

let loop_3b data targets = // This is not for the autoencoder, but for the final logistic regression layer. We train it separately first so it does not distrupt the pretrained weights below it.
    let l,r = layers_fine_tune |> Array.splitAt 3
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data l // Scan is like fold except it returns the intermediates.
    tape.Add(BlockReverse()) // This blocks the reverse pass from running past this point. It is so the gradients get blocked and only the top two layers get trained.
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) (outputs |> Array.last) r // Scan is like fold except it returns the intermediates.
    let out = outputs.[outputs.Length-1]
    squared_error_cost targets out, None

let loop_fine_tune data targets = // The full net with the pretrained weights.
    let outputs = Array.fold(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data layers_fine_tune
    cross_entropy_cost targets outputs, Some (lazy get_accuracy targets.r.P outputs.r.P)

// It might be possible to get more speed by not repeating needless calculations in the lower layers, but that would require switching
// branches and some modifying the training loop, but this is decent enough.
// Doing it like this is in fact the most effiecient from a memory standpoint.

let train_mnist_sgd num_iters learning_rate training_loop (layers: IFeedforwardLayer[]) =
    [|
    let mutable r' = 0.0f
    let base_nodes = layers |> Array.map (fun x -> x.ToArray) |> Array.concat // Stores all the base nodes of the layer so they can later be reset.
    for i=1 to num_iters do
        for x in dtrain do
            let data, target = x
            let (r:Df), _ = training_loop data target // Builds the tape.

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
            let r,lazy_acc = training_loop data target // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 dtest.Length) // Adds the cost to the accumulator.
            match lazy_acc with
            | Some (lazy_acc: Lazy<floatType>) -> acc <- acc+lazy_acc.Value // Here the accuracy calculation is triggered by accessing it through the Lazy property.
            | None -> ()

            if System.Single.IsNaN r' then failwith "Nan error"

            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed.

        printfn "The validation cost at iteration %i is %f" i r'
        if acc <> 0.0f then printfn "The accuracy is %i/10000" (int acc)
        let r2 = r'
        r' <- 0.0f
        yield r1,r2
    |]

let mutable loop_iter = 1

// For the autoencoders it seems 0.1f is a decent learning rate.
// They blow up with 0.2f.
// The lower learning rate in the final layer does not help, in fact the higher does.
// My record here is 99.09% after a few hours of playing around.
// Might be possible to do even better with max norm normalization.
for loop,layers,num_iters,learning_rate in [|loop_1,layers_1,10,0.1f;loop_2,layers_2,10,0.1f;loop_3,layers_3,10,0.1f;loop_3b,layers_fine_tune,10,0.1f;loop_fine_tune,layers_fine_tune,30,0.2f|] do
    printfn "Starting training loop %i..." loop_iter
    let s = train_mnist_sgd num_iters learning_rate loop layers

    let l = [|for l,_ in s do yield l|]
    let r = [|for _,r in s do yield r|]

    //(Chart.Combine [|Chart.Line l;Chart.Line r|]).ShowChart() |> ignore

    loop_iter <- loop_iter+1
