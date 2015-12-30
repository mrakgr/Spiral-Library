// Spiral reverse AD example. Used for testing.
// Cross entropy bugged out on me so I had to bring in this.

#load "../Spiral Library/ad_utils_spiral_v1.fsx"
open Ad_utils_spiral_v1

#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"

open FSharp.Charting

let w1 = DM.makeNode(3,2,[|0.5f;0.4f;0.3f;0.2f;0.1f;0.0f|])
let bias1 = DM.makeNode(3,1,[|0.5f;0.4f;0.3f|])
let w2 = DM.makeNode(1,3,[|-0.55f;-0.4f;-0.25f|])
let bias2 = DM.makeNode(1,1,[|-0.8f|])

let input = DM.makeConstantNode(2,4,[|0.0f;0.0f;0.0f;1.0f;1.0f;0.0f;1.0f;1.0f;|])
let output = DM.makeConstantNode(1,4,[|0.0f;1.0f;1.0f;0.0f|])

let base_nodes = [|w1;bias1;w2;bias2|]

let z1 = addb (matmult w1 input) bias1
let a1 = tanh_ z1

let z2 = addb (matmult w2 a1) bias2
let a2 = sigmoid z2

let target = output
let activations = a2

let r = cross_entropy_cost target activations

(*
let log_activations = log_ activations
let neg_cross_ent_l = hadmult target log_activations

let neg_target = neg target
let neg_target_plus_one = scalar_add neg_target 1.0f

let neg_activations = neg activations
let neg_activations_plus_one = scalar_add neg_activations 1.0f
let log_neg_activations_plus_one = log_ neg_activations_plus_one

let neg_cross_ent_r = hadmult neg_target_plus_one log_neg_activations_plus_one
let cross_ent = add 1.0f neg_cross_ent_l 1.0f neg_cross_ent_r

let s = sum cross_ent
let r = scale (-1.0f/float32 (!target.r.P).num_cols) s
*)


let train =
    [|
    for i=1 to 1000 do
        tape.forwardpropTape 0

        printfn "The cost is %f at iteration %i" !r.r.P i

        // Resets the adjoints to zero.
        for x in base_nodes do x.r.A.contents.setZero()
        tape.resetTapeAdjoint 0
        r.r.A := 1.0f
        tape.reversepropTape 0

        add_gradients_to_weights' base_nodes 1.0f
        yield !r.r.P
    |]

(Chart.Line train).ShowChart()

tape.DisposeAll()
