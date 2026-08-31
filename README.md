# Femtochat

Femtochat is a copy of [nanochat](https://github.com/karpathy/nanochat) in Julia, trying to minimize dependencies. 

The main dependencies are:

  * [Enzyme](https://github.com/EnzymeAD) for auto-differentiation. Enzyme (like tapanade) is a compiler-level autodiff engine, which supports custom rules (e.g. Flash Attention reverse)
  * [DuckDB](https://duckdb.org/) for handling Parquet files. 
  * [CUDA.jl](https://cuda.juliagpu.org/stable/) for running on GPUs.

We do steal a the `BinaryMaxHeap` data structure from [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) (+ inline it into codebase) because importing the full dependency for one 30-line implementation feels like a lot. In a "real" (non-toy) codebase we'd import the full dependency for flexibility. [DataStructures.jl](https://juliacollections.github.io/DataStructures.jl/latest/) is a great package.

Everything is coded "from scratch" in an effort to really understand the full (pre/mid/post-training) stack without layers of misdirection (e.g. PyTorch). I've used PyTorch since ~2019 (_since it was more popular on ASAPP's research team than Keras_) and it took me a long time to understand what was going on under the hood. 

Using Julia allows some of the benefits of Python (strong REPL, interactive experimentation) while eliminating the two-language problem. 

## Notes:

  * Julia is columnar orientated, so the batch shape is different. We use `D x T x B` i.e. each token is a column, and then we group by batches. 
  * We add `parameters.jl` to keep track of parameters, which is something PyTorch handles in `nanochat`. We create a flat parameter vector and map the model into it; this allows us to accumulate gradients in a second flat parameter vector. 