import unittest
from knn_ncd import KNN_NCD


class TestKNN_NCD(unittest.TestCase):

    def setUp(self):
        self.knn = KNN_NCD(k=1)
        self.training_data = [("apple", "fruit"), ("dog", "animal"), ("cat", "animal")]

    def test_fit(self):
        self.knn.fit(self.training_data)
        self.assertEqual(self.knn.training_set, self.training_data)

    def test_ncd(self):
        self.knn.fit(self.training_data)
        ncd_value = self.knn.calc("apple", "dog")
        self.assertTrue(0 <= ncd_value <= 1)

    def test_predict_single_instance(self):
        self.knn.fit(self.training_data)
        prediction = self.knn.predict(["apple"])
        self.assertEqual(prediction[0], "fruit")

    def test_predict_multiple_instances(self):
        self.knn.fit(self.training_data)
        predictions = self.knn.predict(["apple", "dog"])
        self.assertListEqual(predictions, ["fruit", "animal"])

    def test_k_value(self):
        with self.assertRaises(ValueError):
            KNN_NCD(k=0)

    def test_empty_training_set(self):
        with self.assertRaises(ValueError):
            self.knn.fit([])

    def test_empty_test_set(self):
        self.knn.fit(self.training_data)
        with self.assertRaises(ValueError):
            self.knn.predict([])

if __name__ == '__main__':
    unittest.main()

