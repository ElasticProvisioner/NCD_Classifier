## K-Nearest Neighbors Classifier with Normalized Compression Distance (KNN_NCD)

This module implements a K-Nearest Neighbors classifier that uses Normalized Compression Distance (NCD) to measure similarity between instances. The class `KNN_NCD` provides methods for fitting the model to the training data and predicting the class of test instances.

Author: Asher Bond
Contact: source@elasticprovisioner.com

## Dependencies

Ensure that you have `gzip` and `numpy` installed in your Python environment. You can install them using pip:

```bash
pip install numpy
```

Install various compression algorithms if you're using one not built into Python.
```bash
pip install zstandard python-snappy lz4
```

## Using different compression algorithms

The classifier uses a compression algorithm to compute the Normalized Compression Distance. By default, it uses the gzip algorithm. However, you can use a different algorithm by passing its name as a string to the KNN_NCD constructor, like so:

```python
knn = KNN_NCD(k=1, compression='lzma')
```

The currently supported compression algorithms are: "gzip", "bz2", "lzma", "zstd", "snappy", and "lz4".

These compression algorithms are built into Python:
- gzip
- bz2
- lzma

These compression algorithms require the following Python libraries:
- zstd: Install with `pip install zstandard`
- snappy: Install with `pip install python-snappy`
- lz4: Install with `pip install lz4`

If you try to use a compression algorithm without the required library installed, or if you specify a compression algorithm that is not supported, the KNN_NCD constructor will raise a ValueError.

## Usage

Let's say you have a classification task where you need to classify text strings into different categories (for example, spam and not-spam). Here's how you can use the NCD Classifier:

```python
from knn_ncd import KNN_NCD

# Initialize training data
training_data = [("spam text", "spam"), ("not spam text", "not-spam"), ...] 

# Initialize test data
test_data = ["this is a spam text", "this is not spam", ...]

# Create an instance of KNN_NCD with k as 3
knn_ncd = KNN_NCD(k=3)

# Train the model
knn_ncd.fit(training_data)

# Predict the class labels for test data
predictions = knn_ncd.predict(test_data)
```

In this case, the `knn_ncd` object is trained to classify text strings as "spam" or "not-spam". The number of neighbors `k` is set to `3`, meaning the classifier will consider the 3 nearest neighbors according to the Normalized Compression Distance to make its prediction.


## Testing

You can run the tests for this project using Python's built-in unittest module. Navigate to the project directory and run the following command:

```bash
python -m unittest test_knn_ncd.py

```

The tests check the functionality of the NCD calculation and the prediction mechanism.

## Contribution

Feel free to fork the project, open a pull request, or submit suggestions and bugs in the issue tracker.

## License

Apache 2.0 License is included (LICENSE.txt)

