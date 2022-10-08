+++
title = "A probabilistic programming language from the future"
date = 2022-10-08

[taxonomies]
categories = ["ppl"]
tags = ["probability", "programming", "compiler", "ppl", "probabilistic", "web"]
+++

Lately I've been interested in probabilistic programming languages.

I struggle to precisely target what is the biggest, while doable, achievement I could do. I'm going to introduce what's been boiling in my head.

# State of the art

## Probabilistic programming

In a nutshell, probabilistic programming is about making computer reason about statistical bayesian inference. It refers to a domain-specific language, embedded in a host language or not, where the user declares a model of some kind, and plug-and-play samplers.

This ia called a probabilistic programming language, or __PPL__. The two most popular ones are `Stan` and `PyMC`.

The model is usually _generative_, i.e. a data-generating process that allows of the build a _log-probability_, that is a _log-likelihood_ plus a _log-prior_ (thanks to Bayes' Theorem).

The _sampler_ is an inference algorithm, whose goal is to find the _posterior distribution_ of model parameters. It often belongs to the __Markov Chain Monte Carlo (MCMC)__ family: Metropolis-Hastings, Gibbs, Hamiltonian Monte-Carlo, No-U-Turn Sampler etc.

So the __model definition__ procedure is like this:
- Define _priors_ for each model parameter. This is an a priori belief you have, such as *the $\alpha$ parameter is roughly between 5 and 10*, or _the $b$ parameters is greater than 0_, or _the $r$ parameter is between 0 and 1_.
- Combine parameters with operations such as addition, multiplication, division, $\exp$, $\log$, $\sqrt.$ etc. to generate a _synthetic observation_, or _prediction_.

The __inference__ of the _sampler_ consists in updating the distribution of parameters towards a value that _minimizes_ the difference between observations and prediction. The MCMC algorithm is fundamentally sequential. Indeed, it's a Markov Chain so each state depends on the previous one. Which means it's not embarissingly parallel. 

However, often scientist launch _multiple chains_ in parallel, one on each CPU core, in order to assess the converge of the algorithm and improve the posterior sampling.

<!-- A nice property of Bayes' rule is that when you want to update your beliefs twice, you can do it once and then again with the updated belief as the new prior. -->

### The Hamiltonian Monte-Carlo algorithm

While MCMC has been increasingly popular in scientific domains for the past 50 years, a newish (2011) variant is all the rage right now: HMC for Hamiltonian Monte-Carlo, and its offspring __NUTS__. These are the workhorses of _Stan_ or _PyMC_.

These algorithms are very efficient, although they require the __gradient__ of the log-probability. 

## Automatic differentiation

To achieve this while having total flexibility over the model definition, one can leverage __automatic differentiation__. This means being able to _efficiently_ compute the gradient of a function programmed in the PPL.

The most well-known automatic differentiation engines are included in tensor computation (and often deep learning) libraries. In Python, examples are Tensorflow, PyTorch, JAX, Aesara etc.

### Enzyme

Enter Enzyme. It differentiates directly the LLVM intermediate representation (IR). This means that one can differentiate _any program_ compiled with LLVM. This gives access to gradient of programs written in Fortran, C, C++, Rust, Julia, Swift...

Compared to python frameworks, this is very appealing: it is framework-agnostic ! Better, the programmer doesn't even have to be aware that someone is going to differentiate their program ! Which means arbitrary program differentiation.

## Compilers

To translate a program written in textual form, such as Stan-lang or Python, to machine code, one needs a compiler (or an interpreter, but we'll say compiler here).

- `Stanc`, the official compiler for the Stan language, transpiles to `C++`. In this low-level language, it leverages the Stan math library for automatic differentiation, linear algebra, ODE solver, probability distributions, etc. OpenCL is used to accelerate linear-algebra heavy workloads on general-purpose GPUs.

- `PyMC` uses the `Aesara` tensor library to define a computation graph, and perform symbolic and computational operations on it. As of now, Aesara has two backends: one with Numba, that compiles a subset of python Just-in-time with LLVM, and one with `JAX`, that compiles to `XLA` (_accelerated linear algebra_) just-in-time as well. 

- `Numpyro` uses JAX directly as its tensor library.

- `Tensorflow-probability` targets XLA as well, piggybacking on tensorflow.

XLA targets most CPUs, Nvidia GPUs (with CUDA), or TPUs. And when a model is compiled to XLA, it runs _very fast_. I've seen some of my predictions in _100 $\mu$s_ on CPU. Numba can be very fast as well on CPU.

However, in order to accelerate with XLA out of the CPU, one needs an Nvidia GPU. These are very expensive, power-hungry, and well it's very monopolistic. Or a TPU, which is even worse since it's impossible to own one.

# Big ideas

In this section, I'm going to present the big ideas for a 2022+ scientific programming language such as a PPL.

## Compiler

One can write a compiler in any language. However, some are a better fit for it. One classic language for compilers is `OCaml`. The ML language lends itself well to lex, tokenize, parse, and generate code. 
It is also well-known that the Ocaml compiler is very fast, and there is an OCaml bytecode to javascript compiler (`Js_of_OCaml`, or jsoo). So a compiler written in OCaml could run inside the browser.

Stan has recently (2020-ish ?) swapped its old C++ compiler to a brand new OCaml one: Stanc3. 

`Rust` is the other candidate. More fashionable, modern, is definitely lower-level. One could compile the compiler to WASM and run it inside a browser as well. This is also where the cool kids are, so lots of friendly libraries such as `chumsky` are just waiting for us to leverage them!

From [https://matklad.github.io/2018/06/06/modern-parser-generator.html](https://matklad.github.io/2018/06/06/modern-parser-generator.html) :

> I think today it is the best language for writing compiler-like stuff (yes, better than OCaml!),


## Compiler target

Now, the question is: what kind of code would be generated by this compiler?

I think it could be nice to have two targets, loosely speaking:
- CPU
- GPU

Fortunately, web standards have emerged and we now have two perfectly suitable targets for both. Both of them are in the _write once, run anywhere_ category, including the web!

### WebAssembly

The main target is WebAssembly (WASM). The advantages is portability: run anywhere, web or native.

The compiler may leverage `binaryen` or just the `wasm-encode` Rust crate to compile the PPL to WASM. Then, `wastime` could be used to interpret or compile ahead-of-time (AOT) wasm to native code, or if needed the browser can directly execute the assembly file.

### WebGPU

One can write _shaders_ in WGSL, the __WebGPU__ language. It gives acess to general-purpose GPU (GPGPU) computing, on _any device_. This means my laptop's integrated graphics card as well as the latest RTX 4090.

How to leverage the GPU in the case of a PPL?   
One classic possibility is to optimize for linear algebra, e.g. matric multiplication (GEMM).
Another one would be to run thousands of somewhat short MCMC chains in parallel on the GPU.

### Rust

This is where Rust must come into play. Indeed, Rust is the language with the best support for WebAssembly. And the best-in-class webgpu implementation, that also have native backends, is the __wgpu-rs__ rust library.

Also, it must be very convenient to use a low-level language to manipulate... well, low-level code.

Now, in order to use HMC and NUTS, which are required today to be competitive with Stan or PyMC, one needs to take the gradients.

## Automatic differentiation in WASM

Inspired by Enzyme and its autodiff on LLVM IR, we could imagine differentiating WASM code directly. I have no idea if this is possible, but why not?

The even more tricky part is to differentiate both the CPU (WASM) and GPU (WebGPU) backends.

A probably simpler task would be to design the language and the compiler such that it is easy to differentiate on e.g. the Compiler IR level, before the assembly code is actually generated.

A good reading of [https://github.com/google-research/dex-lang/issues/454](https://github.com/google-research/dex-lang/issues/454) could help I guess.

The alternative is to target LLVM IR, that can emit WASM, and leverage Enzyme to compute the gradients. One would need to write a WebGPU LLVM backend for this. 

## The vision

So we have a PPL that runs either natively or entirely in the browser, on any CPU and any (i)GPU. The whole language is differentiable, which allows one to build efficient gradients and run fast algorithms requiring them such as NUTS.
On any (i)GPU, it can run thousands of (short) chains in parallel, greatly improving the reliability of scientific analysis.
