"""Offline playback preserves saved poses, including old-run endpoints."""
from pathlib import Path
import tempfile
import unittest
import mujoco
import numpy as np
from replay_run import ROOT, load_frames, write_launcher

class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.step = self.folder / 'step_001'
        self.step.mkdir()
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
        self.start = self.model.key_qpos[0].copy()
        self.end = self.start.copy()
        self.end[0] += 0.1

    def test_recorded_frames_are_used_without_recomputing(self):
        mid = self.start.copy(); mid[0] -= 0.03
        np.savez(self.step / 'motion_frames.npz', qpos=[self.start, mid, self.end], time=[0, .2, .4])
        frames, approximate = load_frames(self.folder, self.model)
        self.assertFalse(approximate)
        np.testing.assert_array_equal(frames[1][2], mid)
        np.testing.assert_array_equal(frames[-1][2], self.end)
        self.assertEqual(frames[-1][1], .4)

    def test_legacy_endpoints_and_incomplete_observation_are_preserved(self):
        np.savez(self.step / 'scene_state.npz', qpos=self.start, time=0)
        np.savez(self.step / 'after_state.npz', qpos=self.end, time=.4)
        next_step = self.folder / 'step_002'; next_step.mkdir()
        np.savez(next_step / 'scene_state.npz', qpos=self.end, time=.4)
        frames, approximate = load_frames(self.folder, self.model)
        self.assertTrue(approximate)
        np.testing.assert_array_equal(frames[0][2], self.start)
        np.testing.assert_array_equal(frames[-1][2], self.end)
        self.assertEqual(frames[-1][0], 'step_002')

    def test_invalid_frames_rejected(self):
        np.savez(self.step / 'motion_frames.npz', qpos=np.zeros((2, 1)), time=[0, .4])
        with self.assertRaises(ValueError):
            load_frames(self.folder, self.model)

if __name__ == '__main__':
    unittest.main()
