"""A lift must retain physical grip for simulator time, without requiring idle motion."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from simulation.lift_scene import LiftSuccess


class LiftSuccessTests(unittest.TestCase):
    def setUp(self):
        self.cube=np.array([.5,0,.33]);self.env=SimpleNamespace(model=None,data=SimpleNamespace(time=0.))
        self.env.controller=SimpleNamespace(end_effector_pose=lambda:(self.cube+np.array([0,0,.003]),None))
        self.contacts=[dict(finger='left_finger'),dict(finger='right_finger')]
        self.position=patch('data_tools.panda_demonstrations.cube_position',side_effect=lambda env:self.cube.copy())
        self.contact=patch('simulation.lift_scene.finger_cube_contacts',side_effect=lambda model,data:self.contacts)
        self.position.start();self.contact.start();self.addCleanup(self.position.stop);self.addCleanup(self.contact.stop)
        self.tracker=LiftSuccess([.5,0,.27])

    def test_continuing_upward_motion_qualifies(self):
        for t in (0.,.4,.8,1.2):
            self.env.data.time=t;self.cube[2]+=.004
            result=self.tracker.observe(self.env)
        self.assertTrue(result['success']);self.assertAlmostEqual(result['continuous_grasp_seconds'],1.2)

    def test_prediction_waits_do_not_count(self):
        for _ in range(50):result=self.tracker.observe(self.env)
        self.assertFalse(result['success']);self.assertEqual(result['continuous_grasp_seconds'],0.)

    def test_lost_contact_or_low_height_resets_the_clock(self):
        self.tracker.observe(self.env);self.env.data.time=.8;self.tracker.observe(self.env)
        self.contacts=[dict(finger='left_finger')];self.env.data.time=1.2
        self.assertFalse(self.tracker.observe(self.env)['success'])
        self.contacts.append(dict(finger='right_finger'));self.env.data.time=1.6
        self.assertEqual(self.tracker.observe(self.env)['continuous_grasp_seconds'],0.)
        self.cube[2]=.31;self.env.data.time=3.
        self.assertFalse(self.tracker.observe(self.env)['success'])
