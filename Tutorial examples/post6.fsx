let sigmoid a = 1.0f/(1.0f+exp(-a))

let target = -0.5f
let input = 1.0f

let W = 1.5f
let W2 = 2.0f
let bias = 0.25f
let bias2 = 0.0f

let y = (target-sigmoid(W2 * tanh(W*input+bias) + bias2))**2.0f

// These are the original assignments. In AD literature the starting variables are denoted from i up to 0, but here they will start at 0.
// Starting from scratch, this would be the evaluation trace of the program had I decompiled it.
let v0 = target
let v1 = input
let v2 = W
let v3 = W2
let v4 = bias
let v5 = bias2
// The first calculation is W*input = v2*v1.
let v6 = v2*v1
// Then comes v6+bias=v6+v4
let v7 = v6+v4
// Then comes tanh(v7)
let v8 = tanh(v7)
// Then comes W2*v8=v3*v8.
let v9 = v3*v8
// Then comes v9+bias2=v9+v5
let v10 = v9+v5
// Then comes sigmoid(v10)
let v11 = sigmoid(v10)
// Then comes target-v11=v0-v11
let v12 = v0-v11
// Then comes v12**2.0f
let v13 = v12**2.0f

module Isolated =
    let a = 3.0f
    let b = 2.0f
    let isolated_mult a b =
        let c = a*b
        let dc_a = b // dc/da = b - The derivative of c with respect to a
        let dc_b = a // dc/db = a - The derivative of c with respect to b
        c,dc_a,dc_b
    let c,dc_a,dc_b = isolated_mult a b

// The functions in here propagate the error from upwards to downwards.
module Propagating =
    let a = 3.0f
    let b = 2.0f
    let prop_mult a b =  
        let c = a*b
        // http://stackoverflow.com/questions/36636/what-is-a-closure
        let fdc_a error_c = error_c*b // dc/da = error*b - The derivative of c with respect to a. This is a function.
        let fdc_b error_c = error_c*a // dc/db = error*a - The derivative of c with respect to b. This is a function.
        c,fdc_a,fdc_b
    let c,fdc_a,fdc_b = prop_mult a b
    let dc_a = fdc_a 1.0f
    let dc_b = fdc_b 1.0f

module Propagating2 =
    let a = 3.0f
    let b = 2.0f
    let prop_mult a b =  
        let c = a*b
        // http://stackoverflow.com/questions/36636/what-is-a-closure
        let fdc_a error_c = error_c*b // dc/da = error*b - The derivative of c with respect to a. This is a function.
        let fdc_b error_c = error_c*a // dc/db = error*a - The derivative of c with respect to b. This is a function.
        c,fdc_a,fdc_b

    // (a*b)*(a*b)
    let c,fdc_a,fdc_b = prop_mult a b // c=a*b
    let d,fdc_a',fdc_b' = prop_mult a b // d=a*b
    let e,fde_c,fde_d = prop_mult c d // e=c*d

    let er_c, er_d = fde_c 1.0f, fde_d 1.0f // errors (or adjoints) of e with respect to c and d
    let er_a', er_b' = fdc_a' er_c, fdc_b' er_d // adjoints of a and b
    let er_a, er_b = fdc_a er_c, fdc_b er_d // adjoints of a and b

    let adjoint_a = er_a+er_a'
    let adjoint_b = er_b+er_b'

module Propagating3 =
    type Df =
        struct // Struct is similar to record or a class except it is stack allocated.
        val P : float32 // primal
        val A : float32 ref // adjoint (reference type)
        new p = {P=p;A=ref 0.0f}
        end
    let a = Df 3.0f
    let b = Df 2.0f
    let mult (a: Df) (b: Df) =  
        let c = Df (a.P*b.P)
        // http://stackoverflow.com/questions/36636/what-is-a-closure
        let fb() = //The function for the backwards pass.
            a.A := !c.A*b.P + !a.A 
            b.A := !c.A*a.P + !b.A 
        c,fb

    // (a*b)*(a*b)
    let c,fc = mult a b // c=a*b
    let d,fd = mult a b // d=a*b
    let e,fe = mult c d // e=c*d

    e.A := 1.0f // Feed the 1.0f at the top.

    fe() // Crank the machine
    fd() // Crank the machine
    fc() // Crank the machine

    let adjoint_a = !a.A
    let adjoint_b = !b.A

module Propagating4 =
    open System.Collections.Generic
    
    type Df =
        struct // Struct is similar to record or a class except it is stack allocated.
        val P : float32 // primal
        val A : float32 ref // adjoint (reference type)
        new p = {P=p;A=ref 0.0f}
        end
    let a = Df 3.0f
    let b = Df 2.0f

    let tape = List<unit -> unit>() // List is not really a list, but a dynamic array. unit -> unit is the function type that takes no parameters and returns nothing.
    let mult (a: Df) (b: Df) =  
        let c = Df (a.P*b.P)
        // http://stackoverflow.com/questions/36636/what-is-a-closure
        let fb() = //The function for the backwards pass.
            a.A := !c.A*b.P + !a.A 
            b.A := !c.A*a.P + !b.A 
        tape.Add(fb)
        c

    // (a*b)*(a*b)
    let c = mult a b // c=a*b
    let d = mult a b // d=a*b
    let e = mult c d // e=c*d

    e.A := 1.0f // Feed the 1.0f at the top.

    for i=tape.Count-1 downto 0 do tape.[i]() // Let the computer crank it for you from top to bottom.

    let adjoint_a = !a.A
    let adjoint_b = !b.A

module Propagating5 =
    open System.Collections.Generic
    
    type Df =
        struct // Struct is similar to record or a class except it is stack allocated.
        val P : float32 // primal
        val A : float32 ref // adjoint (reference type)
        new p = {P=p;A=ref 0.0f}
        end
    let a = Df 3.0f
    let b = Df 2.0f

    let tape = List<unit -> unit>() // List is not really a list, but a dynamic array. unit -> unit is the function type that takes no parameters and returns nothing.
    let mult (a: Df) (b: Df) =  
        let c = Df (a.P*b.P)
        // http://stackoverflow.com/questions/36636/what-is-a-closure
        let fb() = //The function for the backwards pass.
            a.A := !c.A*b.P + !a.A 
            b.A := !c.A*a.P + !b.A 
        tape.Add(fb)
        c

    type Df with
        static member inline (*)(a: Df, b: Df) = mult a b // The overloaded * operator

    // (a*b)*(a*b)
    let e = (a*b)*(a*b)

    e.A := 1.0f // Feed the 1.0f at the top.

    for i=tape.Count-1 downto 0 do tape.[i]() // Let the computer crank it for you from top to bottom.

    let adjoint_a = !a.A
    let adjoint_b = !b.A

module ReverseADExample =
    type Df =
        struct // Struct is similar to record or a class except it is stack allocated.
        val P : float32 // primal
        val A : float32 ref // adjoint (reference type)
        new p = {P=p;A=ref 0.0f}
        end with

        override t.ToString() = sprintf "(%f,%f)" t.P !t.A // To make F# Interactive print out the fields

    open System.Collections.Generic
    let tape = List<unit -> unit>() // List is not really a list, but a dynamic array. unit -> unit is the function type that takes no parameters and returns nothing.
    let sigmoid (a: Df) =  
        let c = Df (1.0f/(1.0f+exp(-a.P)))
        let fb() = // The function for the backwards pass.
            a.A := !c.A*c.P*(1.0f-c.P) + !a.A 
        tape.Add(fb)
        c
    let tanh (a: Df) =
        let c = Df (tanh a.P)
        let fb() = // The function for the backwards pass.
            a.A := !c.A*(1.0f-c.P*c.P) + !a.A 
        tape.Add(fb)
        c
    let pow (a: Df) b =
        let c = Df (a.P**b)
        let fb() = // The function for the backwards pass.
            a.A := !c.A*b*(c.P/a.P) + !a.A 
        tape.Add(fb)
        c
    let mult (a: Df) (b: Df) =  
        let c = Df (a.P*b.P)
        let fb() = //The function for the backwards pass.
            a.A := !c.A*b.P + !a.A 
            b.A := !c.A*a.P + !b.A 
        tape.Add(fb)
        c
    let add (a: Df) (b: Df) =  
        let c = Df (a.P+b.P)
        let fb() = //The function for the backwards pass.
            a.A := !c.A + !a.A 
            b.A := !c.A + !b.A 
        tape.Add(fb)
        c
    let sub (a: Df) (b: Df) =  
        let c = Df (a.P-b.P)
        let fb() = //The function for the backwards pass.
            a.A := !c.A + !a.A 
            b.A := !b.A - !c.A
        tape.Add(fb)
        c

    type Df with
        static member inline (*)(a: Df, b: Df) = mult a b // The overloaded * operator
        static member inline (+)(a: Df, b: Df) = add a b // The overloaded + operator
        static member inline (-)(a: Df, b: Df) = sub a b // The overloaded - operator
        static member inline Pow(a: Df, b) = pow a b // The overloaded ** operator

    let target = Df -0.5f
    let input = Df 1.0f

    let W = Df 1.5f
    let W2 = Df 2.0f
    let bias = Df 0.25f
    let bias2 = Df 0.0f

    //let y = (target-sigmoid(W2 * tanh(W*input+bias) + bias2))**2.0f

    // These are the original assignments. In AD literature the starting variables are denoted from i up to 0, but here they will start at 0.
    // Starting from scratch, this would be the evaluation trace of the program had I decompiled it.
    let v0 = target
    let v1 = input
    let v2 = W
    let v3 = W2
    let v4 = bias
    let v5 = bias2
    // The first calculation is W*input = v2*v1.
    let v6 = v2*v1
    // Then comes v6+bias=v6+v4
    let v7 = v6+v4
    // Then comes tanh(v7)
    let v8 = tanh(v7)
    // Then comes W2*v8=v3*v8.
    let v9 = v3*v8
    // Then comes v9+bias2=v9+v5
    let v10 = v9+v5
    // Then comes sigmoid(v10)
    let v11 = sigmoid(v10)
    // Then comes target-v11=v0-v11
    let v12 = v0-v11
    // Then comes v12**2.0f
    let v13 = v12**2.0f

    v13.A := 1.0f // Feed the 1.0f at the top.

    for i=tape.Count-1 downto 0 do tape.[i]() // Let the computer crank it for you from top to bottom.

    let adjoint_W = !W.A
    let adjoint_W2 = !W2.A
    let adjoint_bias = !bias.A
    let adjoint_bias2 = !bias2.A

    // Once more from the top.
    let l = [|v5;v4;v3;v2;v1;v0|] |> Array.iter (fun x -> x.A := 0.0f) // Reset the adjoints of the base variables.
    tape.Clear()
    let y = (target-sigmoid(W2 * tanh(W*input+bias) + bias2))**2.0f
    y.A := 1.0f
    for i=tape.Count-1 downto 0 do tape.[i]() // Let the computer crank it for you from top to bottom.

    let adjoint_W' = !W.A
    let adjoint_W2' = !W2.A
    let adjoint_bias' = !bias.A
    let adjoint_bias2' = !bias2.A

module Forward1 =
    open System.Collections.Generic
    
    type Df =
        struct // Struct is similar to record or a class except it is stack allocated.
        val P : float32 // primal
        val T : float32 ref // tangent (reference type)

        new (p) = {P=p; T=ref 0.0f}
        new (p,t) = {P=p; T=ref t}
        end with
        override t.ToString() = sprintf "(%f,%f)" t.P !t.T // To make F# Interactive print out the fields
    let a = Df 3.0f
    let b = Df 2.0f

    let mult (a: Df) (b: Df) =  
        let cp = a.P*b.P // primal
        let ct = !a.T * b.P + a.P * !b.T // tangent
        let c = Df(cp,ct)
        c

    type Df with
        static member inline (*)(a: Df, b: Df) = mult a b // The overloaded * operator

    // In order to get the derivative of the cost with respect to a, set a's tangent to 1.0f and every others' to 0.0f
    a.T := 1.0f
    b.T := 0.0f
    let c = a*a*b*b
    // In order to get the derivative of the cost with respect to b, set a's tangent to 1.0f and every others' to 0.0f
    a.T := 0.0f
    b.T := 1.0f
    let c' = a*a*b*b

module Hessian =
    open System.Collections.Generic
    
    type Df =
        struct // Struct is similar to record or a class except it is stack allocated.
        val P : float32 // primal
        val T : float32 ref // tangent (reference type)
        val A : float32 ref // adjoint (reference type)
        val TA : float32 ref // tangent of the adjoint (reference type)

        new (p) = {P=p; T=ref 0.0f; A=ref 0.0f; TA=ref 0.0f}
        new (p,t) = {P=p; T=ref t; A=ref 0.0f; TA=ref 0.0f}
        end with
        override t.ToString() = sprintf "(primal=%f,tangent=%f,adjoint=%f,tangent of adjoint=%f)" t.P !t.T !t.A !t.TA // To make F# Interactive print out the fields
    let a = Df 3.0f
    let b = Df 2.0f

    let tape = List<unit -> unit>() // List is not really a list, but a dynamic array. unit -> unit is the function type that takes no parameters and returns nothing.
    let mult (a: Df) (b: Df) =  
        let cp = a.P*b.P // primal
        let ct = !a.T * b.P + a.P * !b.T // tangent
        let c = Df(cp,ct)
        let fb() = 
            a.A := !c.A*b.P + !a.A 
            b.A := !c.A*a.P + !b.A 

            // The derivative of !c.A*b.P is !c.TA*b.P + !c.A* !b.T
            // We also run the forward mode during the reverse.
            // This calculates the Hessian multiplied by a vector.
            a.TA := !c.TA*b.P + !c.A* !b.T + !a.TA 
            b.TA := !c.TA*a.P + !c.A* !a.T + !b.TA 
        tape.Add(fb)
        c

    type Df with
        static member inline (*)(a: Df, b: Df) = mult a b // The overloaded * operator

    // In order to get the derivative of the cost with respect to a, set a's tangent to 1.0f and every others' to 0.0f
    a.T := 1.0f
    b.T := 0.0f
    let c = a*a*b*b
    c.A := 1.0f
    for i=tape.Count-1 downto 0 do tape.[i]() // Let the computer crank it for you from top to bottom.
    printfn "The elements of the Hessian are inside the tangent of the adjoint."
    printfn "a=%A" a
    printfn "b=%A" b


    // Once more from the top.
    [|a;b|] |> Array.iter (fun x -> x.A := 0.0f;x.T := 0.0f;x.TA := 0.0f) // Reset the adjoints and the tangents of the base variables.
    a.T := 0.0f
    b.T := 1.0f
    tape.Clear()
    let c' = a*a*b*b
    c'.A := 1.0f
    for i=tape.Count-1 downto 0 do tape.[i]() // Let the computer crank it for you from top to bottom.
    printfn "a=%A" a
    printfn "b=%A" b