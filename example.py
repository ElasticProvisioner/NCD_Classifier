from knn_ncd import KNN_NCD

# Initialize training data
training_data = [("string1", "class1"), ("string2", "class2"), ...] 

# Initialize test data
test_data = ["test_string1", "test_string2", ...]

# Create an instance of KNN_NCD (considering 3 neighbors)
knn_ncd = KNN_NCD(k=3)

# Train the model
knn_ncd.fit(training_data)

# Predict the class labels for test data
predictions = knn_ncd.predict(test_data)

