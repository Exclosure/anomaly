# Anomaly

Differentiable orbital dynamics accelerated with JAX.


## Installation

In one line:

```bash
pip install anomaly[cpu]
```

This will install the CPU-version of JAX which is readily supported on the most machines. To take advantage of GPU or TPU optimizations, you can replace `cpu` above with `gpu` or `tpu`. These options correspond precisely to the JAX options.


## Development

Check out this code and from the base folder run

```bash
pip install -e ".[cpu,dev]"
```

This will give you the CPU-only version of the package. Of course, if you have a CUDA-enabled GPU or TPU, you can replace `cpu` with either `gpu` or `tpu`.

Install the pre-commit hooks with

```bash
pre-commit install
```


### Linting

In order to run black,

## Troubleshooting

### Apple M1

If you see the error
```
E   RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support. You may be able work around this issue by building jaxlib from source.
```
this may be due to binaries not being availalbe for the Apple M1 chip. You may have luck installing older versions of `jax` and `jaxlib`, e.g.,

```
pip install jaxlib==0.1.60 jax==0.2.10
```

See this [Github issue](https://github.com/google/jax/issues/5501) for the current status.
