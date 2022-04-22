Minimal example using MNIST for easy implementation of highly configurable ray
tune CLIs based on [jsonargparse](https://github.com/omni-us/jsonargparse). Only
requires a function that uses
[LightningCLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html)
to run fit. The example is an adaptation of
[ray/tune/examples/mnist_pytorch_lightning.py](https://github.com/ray-project/ray/python/ray/tune/examples/mnist_pytorch_lightning.py).
To try out the code first install the packages listed in
[requirements.txt](requirements.txt).

The normal MNIST pytorch-lightning CLI is implemented in file
[mnist_lightning_cli.py](mnist_lightning_cli.py). For example, to run a fit you
could do:

    ./mnist_lightning_cli.py fit --config mnist_fit_config.yaml

The ray tune CLI is implemented in file
[mnist_ray_tune_cli.py](mnist_ray_tune_cli.py). To run a hyperparameter search,
this CLI first receives ray options, followed by "--" and then LightningCLI fit
options. For example:

    ./mnist_ray_tune_cli.py --config mnist_tune_config.yaml -- --config mnist_fit_config.yaml

This repo is only intended to illustrate the concept. The idea would be to
contribute an improved version either to
[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) or
[ray](https://github.com/ray-project/ray) or make it an independent pypi
package. The improved version would probably be a class similar to
[LightningCLI](https://github.com/PyTorchLightning/pytorch-lightning/blob/9b2b1bb494f928137be67e325b3fc8544a3bf321/pytorch_lightning/utilities/cli.py#L460-L461)
to allow customization. It would also automatically save the tune config to a
standard location to ease experiment reporting and reproducibility.
