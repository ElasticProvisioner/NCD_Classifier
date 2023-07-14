# NCD Classifier

A Python implementation of the K-Nearest Neighbors (KNN) classifier using Normalized Compression Distance (NCD). This unique approach to KNN leverages the idea of compressing data to measure similarity, providing a powerful tool for text classification tasks.

Author: Asher Bond
Contact: source@elasticprovisioner.com

### Dependencies

Ensure that you have `gzip` and `numpy` installed in your Python environment. You can install them using pip:

```bash
pip install numpy
```

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

