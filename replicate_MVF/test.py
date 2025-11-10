import unittest

import torch

from tadaconv.models.utils.losses import BalancedSoftTargetCrossEntropy, SoftTargetCrossEntropy


class TestMetrics(unittest.TestCase):
    def test_balanced_accuracy(self):
        import torch
        from tadaconv.utils.metrics import balanced_accuracy

        preds = {
            'task1': torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]]),
            'task2': torch.tensor([[0.7, 0.3], [0.9, 0.1], [0.5, 0.5], [0.9, 0.1]])
        }
        labels = {
            'task1': torch.tensor([1, 0, 2, 1]),
            'task2': torch.tensor([0, 1, 0, 0])
        }
        ks = {'task1': 3, 'task2': 2}

        result = balanced_accuracy(preds, labels, ks)
        expected = {
            'task1': 100.0,
            'task2': 50.0,
            'balanced_acc_joint': 75.0
        }

        for key in expected:
            self.assertAlmostEqual(result[key], expected[key], places=5)

    def test_balanced_soft_target_cross_entropy(self):
        criterion = BalancedSoftTargetCrossEntropy()
        inputs = torch.tensor([[2.0, 0.5, 0.3],
                               [0.1, 0.2, 3.0],
                               [1.0, 2.0, 0.1],
                               [0.5, 0.5, 0.5]])
        targets = torch.tensor([[0.15, 0.7, 0.15],
                                [0.05, 0.15, 0.8],
                                [0.2, 0.7, 0.2],
                                [0.0, 1.0, 0.0]])
        
        loss = criterion(inputs, targets)
        expected_loss = 1.0949  # Precomputed expected loss value
        self.assertAlmostEqual(loss.item(), expected_loss, places=4)

        criterion = SoftTargetCrossEntropy()

        loss2 = criterion(inputs, targets)

        self.assertTrue(loss2.item() > loss.item(), "Balanced loss should be less than standard soft target loss")


if __name__ == '__main__':
    unittest.main()