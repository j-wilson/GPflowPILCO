# GPflowPILCO
A modern implementation of [PILCO](https://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf) based on [GPflow](https://github.com/GPflow/GPflow). This package focuses on model-based reinforcement learning with Gaussian processes. Alongside [GPflowSampling](https://github.com/j-wilson/GPflowSampling), this code is intended as a companion for the paper [Pathwise Conditioning of Gaussian Processes](https://arxiv.org/abs/2011.04026). 


## Package content
| Module | Description |
| --- | --- |
| `components` | Miscellania such as objective functions and state encoders |
| `dynamics` | Generic classes for working with dynamical systems |
| `envs` | Various enviroments based on [Gym](https://github.com/openai/gym) |
| `loops` | High-level algorithm suites, such as variants of PILCO
| `models` | Gaussian processes and associated methods |
| `moment_matching` | Compute the moments of, e.g., nonlinear functions of Gaussian rvs |
| `utils` | Assorted utility methods |


## Installation
```
git clone git@github.com:j-wilson/GPflowPILCO.git
cd GPflowPILCO
pip install .
```
Note that this package relies on [GPflowSampling](https://github.com/j-wilson/GPflowSampling), which should be installed first.


## Examples
Code for experiments discussed in Section 7.4 of [Pathwise Conditioning of Gaussian Processes](https://arxiv.org/abs/2011.04026) is provided in `examples/cartpole_swingup`.

## Citing Us
If our work helps you in a way that you feel warrants reference, please cite the following paper:
```
@article{wilson2021pathwise,
    title={Pathwise Conditioning of Gaussian Processes},
    author={James T. Wilson
            and Viacheslav Borovitskiy
            and Alexander Terenin
            and Peter Mostowsky
            and Marc Peter Deisenroth},
    booktitle={Journal of Machine Learning Research},
    year={2021},
    url={https://arxiv.org/abs/2002.09309}
}
```
