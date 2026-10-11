import unittest
from training.loss_plots import moving_average


class SmoothingTests(unittest.TestCase):
    def test_centered_neighborhood_and_clipped_edges(self):
        self.assertEqual(moving_average([0,2,4,6,8],3),[1,2,4,6,7])
        self.assertEqual(moving_average([0,2,4,6,8],4),[1,2,3,5,6])

    def test_short_runs_and_single_update_window(self):
        self.assertEqual(moving_average([1,3],100),[2,2])
        self.assertEqual(moving_average([1,3],1),[1,3])
        self.assertEqual(moving_average([],100),[])
        with self.assertRaises(ValueError):moving_average([1],0)


if __name__=='__main__':unittest.main()
