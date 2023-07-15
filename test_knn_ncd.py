import unittest
from knn_ncd import KNN_NCD


class TestKNN_NCD(unittest.TestCase):

    def setUp(self):
        self.comp_algos = ['gzip', 'bz2', 'lzma']
        self.knns = {comp: KNN_NCD(k=1, comp_algo=comp) for comp in self.comp_algos}
        self.training_data = [("apple", "fruit"), ("dog", "animal"), ("cat", "animal")]

    def test_fit(self):
        for knn in self.knns.values():
            knn.fit(self.training_data)
            self.assertEqual(knn.training_set, self.training_data)

    def test_ncd(self):
        for knn in self.knns.values():
            knn.fit(self.training_data)
            ncd_value = knn.calc("apple", "dog")
            self.assertTrue(0 <= ncd_value <= 1)

    def test_predict_single_instance(self):
        for knn in self.knns.values():
            knn.fit(self.training_data)
            prediction = knn.predict(["apple"])
            self.assertEqual(prediction[0], "fruit")

    def test_predict_multiple_instances(self):
        for knn in self.knns.values():
            knn.fit(self.training_data)
            predictions = knn.predict(["apple", "dog"])
            self.assertListEqual(predictions, ["fruit", "animal"])

    def test_k_value(self):
        with self.assertRaises(ValueError):
            KNN_NCD(k=0)

    def test_empty_training_set(self):
        for knn in self.knns.values():
            with self.assertRaises(ValueError):
                knn.fit([])

    def test_empty_test_set(self):
        for knn in self.knns.values():
            knn.fit(self.training_data)
            with self.assertRaises(ValueError):
                knn.predict([])


if __name__ == '__main__':
    unittest.main()

