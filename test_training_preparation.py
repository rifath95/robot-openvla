"""Training split isolation, normalization degeneracy, and upstream loop checks."""
import unittest
import ast
import numpy as np

from prepare_training_dataset import training_statistics,normalize
from training.integrate_upstream import integrated_source


class TrainingPreparationTests(unittest.TestCase):
    def test_held_out_outliers_do_not_change_statistics(self):
        train=dict(split='train',rows=[{'action':[0,0,0,0,0,0,0]},{'action':[.01,.01,.01,0,0,.05,1]}])
        heldout=dict(split='test',rows=[{'action':[999]*7}])
        self.assertEqual(training_statistics([train]),training_statistics([train,heldout]))

    def test_constant_dimensions_remain_finite_and_gripper_is_absolute(self):
        episodes=[dict(split='train',rows=[{'action':[0,0,0,0,0,0,g]} for g in [0,1]])]
        stats=training_statistics(episodes)
        for g in [0,1]:
            result=normalize([0,0,0,0,0,0,g],stats)
            np.testing.assert_array_equal(result,[0,0,0,0,0,0,g])

    def test_upstream_training_preserved_and_stops_at_completed_updates(self):
        source=integrated_source()
        ast.parse(source)
        self.assertIn('normalized_loss.backward()',source)
        self.assertIn('target_modules="all-linear"',source)
        self.assertIn('(batch_idx + 1) // cfg.grad_accumulation_steps',source)
        self.assertIn('gradient_step_idx >= cfg.max_steps',source)
        self.assertNotIn('merged_vla =',source)


if __name__=='__main__':unittest.main()
