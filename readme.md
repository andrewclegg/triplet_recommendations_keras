
# MovieLens recommendations in Keras and TensorFlow

An example of training a triplet network for personalized and non-personalized recommendations using [BPR loss](https://arxiv.org/abs/1205.2618).

Derived originally from [an example by Maciej Kula](https://github.com/maciejkula/triplet_recommendations_keras) with a number of changes.

See the [main Jupyter notebook](triplet_keras.ipynb) for full details. You can [view this on GitHub](https://github.com/andrewclegg/triplet_recommendations_keras/blob/master/triplet_keras.ipynb) or re-run it from scratch to reproduce the results.

In addition to the notebook, there are also three library files providing helper functions: [data.py](data.py), [metrics.py](metrics.py) and [net_helpers.py](net_helpers.py).

## Requirements

Tested in Python 2.7.13 from conda 4.3.22 distribution, plus:

* tensorflow=1.2.0rc0 [built from source](http://www.andrewclegg.org/tech/TensorFlowLaptopCPU.html)
* tensorflow-tensorboard=0.1.2 (pip)
* keras=2.0.4 (pip)
* pydot-ng=1.0.0.15 (conda)
* graphviz=2.38.0 (conda)
* graphviz=0.7.1 (pip)

Why are there two packages called graphviz? Well, confusingly, `conda install graphviz` only installs some binaries, not the actual Python graphviz package. `pip install graphviz` installs this. More details [here](https://stackoverflow.com/a/33433735).

TensorBoard is optional, you don't need this to run the notebook. But depending on how you install TensorFlow, you might have it already, without needing to install it separately.


