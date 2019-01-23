# The HMM filter package

This small Python package lets you improve the accuracy of sequences of predictions in classification problems by combining regular classifiers and Hidden Markov Models (HMMs). Assumptions about your problem (or, in which scenarios this method might be useful):

* Your data is organised into one or more sequences identified by session IDs
* Each sequence is composed by ordered rows containing a timestamp, one or more features, and a categorical attribute to be predicted, e.g., "state"
* You have a classifier to estimate the state given the feature(s)
* The classifier is agnostic about features and predicted states at neighboring timestamps
* You have a training dataset, a (possibly very large) unlabeled dataset, and a test dataset (labeled, for testing)

The HMM filter requires a sufficiently good base classifier to provide meaningful predictions (the HMM filter improves the accuracy of the base classifier, does not substitute it). The HMM filter provides diminishing improved accuracies as the accuracy of the base classifier increases (it becomes incresingly more difficult to be better than the base classifier). There is a sweet spot between these two extremes, where the HMM filter contributes significanly to the overall accuracy, as in the synthetic dataset provided as example. 


## Usage

* Step 1: Train a classifier on training dataset (multi-class or binary)
* Step 2: Predict labels for unlabeled dataset using trained classifier
* Step 3: Estimate HMM state transition matrix from predicted labels of unlabeled dataset
* Step 4: Estimate class probability distributions for test dataset using trained random forest
* Step 5: Predict most likely sequence of states for each session in test dataset using HMM filter


## How does it work


[HMMs](https://en.wikipedia.org/wiki/Hidden_Markov_model) are defined by hidden states, state transition probabilities, possible observations and their emission probabilities. In our problem, the HMM parameters are the following:

* Hidden states are drawn from the categorical distribution of the classification class labels. E.g., `"0:0"` where the pair of numbers identifies a cell in a 2D grid at position `x=0` and `y=0`
* State transition probabilities are represented by the state transition matrix. E.g., `P("0:0 -> "0:1") = 0.2`
* Possible observations are drawn from the categorical distribution of the classification class labels. E.g., `"0:0"`
* Emission probabilities are estimated by the prediction probability estimates. E.g., `{"0:0": 0.4, "0:1": 0.6}`

The HMM filter revises the predictions accordingly to their uncertainty and the state transition matrix estimated from unlabeled data using the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm). E.g, it might suggest to revise the sequence of predictions `["0:0", "1:1", "0:0"]` to `["0:0", "0:0", "0:0"]` since it is more likely to remain in the same cell (accordingly to the transition matrix) and the classifier was uncertain about the correct label in the 2nd position (E.g., `{"0:0": 0.8, "1:1": 0.2}`)

## Example

Jupyter notebooks `dataset.ipynb` and `demo.ipynb` in the `notebooks` folder provide a complete demo: the first defines, and generates, a synthetic dataset. The latter shows how to train a random forest as base classifier, and how to use the the HMM filter to improve the quality of the predictions.

## Installation

You can install `hmm_filter` with pip:

```
pip install hmm-filter
```

## Credits and license

The hmm_filter project is released under the MIT license. Please see [LICENSE.txt](https://github.com/minodes/hmm_filter/blob/master/LICENSE.txt).

Contributors include:

* Michele Dallachiesa: https://github.com/elehcimd
* Adrian Loy: https://github.com/adrianloy


## Development

Tests, builds and releases are managed with `Fabric`.
The build, test and release environment is managed with `Docker`.
Install Docker and Fabric in your system. To install Fabric:

```
pip install Fabric3
```

### Dependencies

For ease of development, the file `requirements.txt` includes the package dependencies.
Any changes to the package dependencies in `setup.py` must be reflected in `requirements.txt`.

### Jupyter Lab

Jupyter Lab is reachable at [http://127.0.0.1:8889](http://127.0.0.1:8889) and
points to the `notebooks` directory.

### Building and publishing a new release

Create a file `secrets.py` in the project directory with the Pypi credentials in this format:

```
pypi_auth = {
    'user': 'youruser',
    'pass': 'yourpass'
}
```

To release a new version:

```
fab release
```


### Running the tests

To run the py.test tests:

```
fab test
```

To run a single test:

```
fab test:tests/test_pep8.py::test_pep8
```

To run tests printing output and stopping at first error:

```
fab test_sx
```

To run the pep8 test:

```
fab test_pep8
```

To fix some common pep8 errors in the code:

```
fab fix_pep8
```

To test the pip package after a new release (end-to-end test):
```
fab test_pip
```

### Docker container

To build the Docker image:

```
fab docker_build
```

To force a complete rebuild of the Docker image without using the cache:

```
fab docker_build:--no-cache
```

To start the daemonized Docker container:

```
fab docker_start
```

To stop the Docker container:

```
fab docker_stop
```

To open a shell in the Docker container:

```
fab docker_sh
```

## Contributing

1. Fork it
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Create a new Pull Request
