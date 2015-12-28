﻿// Spiral reverse AD example. Used for testing.

#load "ad_utils_spiral_v1.fsx"
open Ad_utils_spiral_v1
//#load "spiral_old.fsx"
//open Spiral_old

#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"

open FSharp.Charting

#load "embedded_reber.fsx"
open Embedded_reber

let reber_set = make_reber_set 3000

let make_data_from_set target_length =
    let twenties = reber_set |> Seq.filter (fun (a,b,c) -> a.Length = target_length) |> Seq.toArray
    let batch_size = (twenties |> Seq.length)

    let d_training_data =
        [|
        for i=0 to target_length-1 do
            let input = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield input.[i] |] |> Array.concat
            yield DM.makeConstantNode(7,batch_size,input)|]

    let d_target_data =
        [|
        for i=1 to target_length-1 do // The targets are one less than the inputs. This has the effect of shifting them to the left.
            let output = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield output.[i] |] |> Array.concat
            yield DM.makeConstantNode(7,batch_size,output)|]

    d_training_data, d_target_data


let lstm_embedded_reber_train num_iters learning_rate (data: DM[]) (targets: DM[]) (data_v: DM[]) (targets_v: DM[]) clip_coef (l1: LSTMLayer) (l2: Layer) =
    [|
    let l1 = l1
    let l2 = l2
    
    let base_nodes = [|l1.ToArray;l2.ToArray|] |> Array.concat

    let training_loop (data: DM[]) (targets: DM[]) i =
        tape.Select i
        let costs = [|
            let mutable a, c = l1.runLayerNoH data.[0]
            let b = l2.runLayerNoH a
            let r = squared_error_cost targets.[0] b
            yield r
    
            for i=1 to data.Length-2 do
                let a',c' = l1.runLayer data.[i] a c
                a <- a'; c <- c'
                let b = l2.runLayerNoH a
                let r = squared_error_cost targets.[i] b
                yield r
            |]
        scale (1.0f/float32 (costs.Length-1)) (sum_scalars costs)

    let ts = (data.Length-1)
    let vs = (data_v.Length-1)

    let r = training_loop data targets ts
    let rv = training_loop data_v targets_v vs

    let mutable r' = 0.0f
    let mutable i = 1
    while i <= num_iters && System.Single.IsNaN r' = false do
        tape.forwardpropTape ts
        printfn "The training cost is %f at iteration %i" !r.r.P i
        tape.forwardpropTape vs
        printfn "The validation cost is %f at iteration %i" !rv.r.P i
        yield !r.r.P, !rv.r.P

        tape.resetTapeAdjoint -1 // Resets base adjoints
        tape.resetTapeAdjoint ts // Resets the adjoints for the training select
        r.r.A := 1.0f
        tape.reversepropTape ts // Resets the adjoints for the test select
        add_gradients_to_weights base_nodes learning_rate clip_coef

        //tape.Dispose ts
        //tape.Dispose vs

        i <- i+1
        r' <- !r.r.P
    |]

let d_training_data_20, d_target_data_20 = make_data_from_set 20
let d_training_data_validation, d_target_data_validation = make_data_from_set 30

let hidden_size = 64

let l1 = LSTMLayer.createRandomLSTMLayer hidden_size 7 tanh_ tanh_
let l2 = Layer.createRandomLayer 7 hidden_size sigmoid

// Add the base nodes to the tape for easier resetting and disposal.
tape.Select -1
let base_nodes = [|l1.ToArray;l2.ToArray|] |> Array.concat |> Array.iter (fun x -> tape.Add x)

#time
let s = [|
        yield lstm_embedded_reber_train 100 5.0f d_training_data_20 d_target_data_20 d_training_data_validation d_target_data_validation 1.0f l1 l2
        |] |> Array.concat
#time
// On the GTX 970, I get 3-4s depending on how hot the GPU is.
let l = [|for l,_ in s do yield l|]
let r = [|for _,r in s do yield r|]

(Chart.Combine [|Chart.Line l;Chart.Line r|]).ShowChart()
tape.DisposeAll()
