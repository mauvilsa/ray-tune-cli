Minimal example using MNIST for easy implementation of highly configurable ray
tune CLIs based on jsonargparse. Only requires a function based LightningCLI to
run fit. The example is an adaptation of
https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch_lightning.py

The normal MNIST pytorch-lightning CLI is implemented in file
`mnist_lightning_cli.py`. For example, to run a fit you could do:

    ./mnist_lightning_cli.py fit --config mnist_fit_config.yaml

The ray tune CLI is implemented in file `mnist_ray_tune_cli.py`. To run a
hyperparameter search, this CLI first receives ray options, followed by "--" and
then LightningCLI fit options. For example:

    ./mnist_ray_tune_cli.py --config mnist_tune_config.yaml -- --config mnist_fit_config.yaml
