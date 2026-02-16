import unittest
import numpy as np

from src.knn.classifier import KNNClassifier


class TestKNNClassifier(unittest.TestCase):

    def setUp(self):
        """
        Creo un piccolo dataset sintetico controllato.
        Serve per avere un ambiente stabile e deterministico,
        così posso verificare il comportamento del modello
        senza dipendere da dati esterni.
        """
        self.X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        self.y = np.array([0, 0, 1])

    # =========================
    # __init__
    # =========================

    def test_invalid_k(self):
        """
        Verifico che il modello non accetti k=0.
        È un caso limite importante perché k deve essere >= 1.
        Serve a garantire robustezza dei parametri iniziali.
        """
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)

    def test_invalid_distance_name(self):
        """
        Controllo che venga sollevato un errore se
        viene passata una metrica di distanza non valida.
        Questo evita comportamenti indefiniti.
        """
        with self.assertRaises(ValueError):
            KNNClassifier(distance="invalid")

    # =========================
    # fit
    # =========================

    def test_fit_invalid_X_dim(self):
        """
        Testo il caso in cui X non sia una matrice 2D.
        Il modello deve accettare solo array bidimensionali.
        """
        knn = KNNClassifier()
        with self.assertRaises(ValueError):
            knn.fit(np.array([1, 2, 3]), self.y)

    def test_fit_invalid_y_dim(self):
        """
        Testo il caso in cui y non sia un array 1D.
        Questo garantisce coerenza tra feature e label.
        """
        knn = KNNClassifier()
        with self.assertRaises(ValueError):
            knn.fit(self.X, np.array([[0, 1, 1]]))

    def test_fit_length_mismatch(self):
        """
        Verifico che venga sollevato errore se
        il numero di campioni in X e y non coincide.
        È un controllo fondamentale di consistenza.
        """
        knn = KNNClassifier()
        with self.assertRaises(ValueError):
            knn.fit(self.X, np.array([0, 1]))

    def test_fit_k_greater_than_samples(self):
        """
        Caso limite: k maggiore del numero di campioni.
        Non avrebbe senso cercare più vicini di quanti esistano.
        """
        knn = KNNClassifier(k=10)
        with self.assertRaises(ValueError):
            knn.fit(self.X, self.y)

    def test_fit_success(self):
        """
        Caso positivo: verifico che dopo il fit
        il modello memorizzi correttamente X_train e y_train.
        """
        knn = KNNClassifier()
        knn.fit(self.X, self.y)
        self.assertIsNotNone(knn.X_train)
        self.assertIsNotNone(knn.y_train)

    # =========================
    # _compute_distances
    # =========================

    def test_compute_distances_without_fit(self):
        """
        Verifico che non sia possibile calcolare distanze
        prima di aver fatto il fit.
        Questo evita uso scorretto del modello.
        """
        knn = KNNClassifier()
        with self.assertRaises(RuntimeError):
            knn._compute_distances(np.array([0, 0]))

    def test_compute_distances_shape(self):
        """
        Controllo che il numero di distanze restituite
        sia uguale al numero di campioni nel training set.
        Serve a garantire correttezza strutturale.
        """
        knn = KNNClassifier()
        knn.fit(self.X, self.y)
        dists = knn._compute_distances(np.array([0, 0]))
        self.assertEqual(dists.shape, (self.X.shape[0],))

    # =========================
    # _vote
    # =========================

    def test_vote_no_tie(self):
        """
        Caso normale: nessun pareggio tra le classi.
        Verifico che venga scelta la classe più frequente.
        """
        knn = KNNClassifier()
        labels = np.array([0, 0, 1])
        pred = knn._vote(labels)
        self.assertEqual(pred, 0)

    def test_vote_with_tie(self):
        """
        Caso di pareggio tra classi.
        Verifico che il modello scelga una delle due
        in modo controllato (usando random_state).
        """
        knn = KNNClassifier(random_state=42)
        labels = np.array([0, 1])
        pred = knn._vote(labels)
        self.assertIn(pred, [0, 1])

    # =========================
    # predict / predict_one
    # =========================

    def test_predict_without_fit(self):
        """
        Verifico che il modello non possa fare predizioni
        senza essere stato addestrato.
        """
        knn = KNNClassifier()
        with self.assertRaises(RuntimeError):
            knn.predict(np.array([0, 0]))

    def test_predict_one(self):
        """
        Caso standard: predizione di un singolo campione.
        Verifico che il modello restituisca la classe corretta.
        """
        knn = KNNClassifier(k=1)
        knn.fit(self.X, self.y)
        pred = knn.predict_one(np.array([0.1, 0.1]))
        self.assertEqual(pred, 0)

    def test_predict_multiple(self):
        """
        Verifico che predict su più campioni
        restituisca un numero di predizioni corretto.
        """
        knn = KNNClassifier(k=1)
        knn.fit(self.X, self.y)
        preds = knn.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    # =========================
    # predict_proba
    # =========================

    def test_predict_proba_sum_to_one(self):
        """
        Controllo che le probabilità sommino a 1.
        È una proprietà fondamentale di una distribuzione.
        """
        knn = KNNClassifier(k=2)
        knn.fit(self.X, self.y)
        proba = knn.predict_proba_one(np.array([0.5, 0.5]))
        self.assertAlmostEqual(np.sum(proba), 1.0)

    def test_predict_proba_shape(self):
        """
        Verifico che la shape della matrice delle probabilità
        sia coerente con il numero di campioni predetti.
        """
        knn = KNNClassifier(k=2)
        knn.fit(self.X, self.y)
        proba = knn.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])


if __name__ == "__main__":
    unittest.main()
