"""Real contact pick/place + sequential replay; no model or paid cloud."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from panda_actions import action_contract, from_controller_action, to_controller_action
from panda_demonstrations import record_episode, replay_episode
from prepare_panda_dataset import prepare_dataset


class DemonstrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.directory.name) / 'episode'
        cls.metadata = record_episode(cls.folder)
        cls.replay = replay_episode(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_successful_contact_task_and_sequential_replay(self):
        self.assertTrue(self.metadata['success'])
        self.assertGreater(self.replay['maximum_cube_lift_metres'], .08)
        self.assertLess(self.replay['placement_xy_error_metres'], .015)
        self.assertLess(self.replay['maximum_qpos_difference'], 1e-6)
        rows = [json.loads(line) for line in (self.folder / 'transitions.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), self.replay['replayed_transitions'])
        self.assertTrue(all(row['executed'] and row['physics_steps'] == 200 for row in rows))
        self.assertTrue(all(abs(row['physics_seconds'] - .4) < 1e-12 for row in rows))
        self.assertTrue(any(np.linalg.norm(np.array(row['proposed_action'])[:3]) > .01 for row in rows))
        self.assertTrue(all(np.linalg.norm(np.array(row['action'])[:3]) <= .0100000001 for row in rows))
        for before, after in zip(rows, rows[1:]):
            self.assertEqual((self.folder / before['next_observation']).read_bytes(),
                             (self.folder / after['observation']).read_bytes())

    def test_statistics_use_executed_actions_and_panda_key(self):
        output = Path(self.directory.name) / 'dataset'
        result = prepare_dataset([self.folder], output)
        stats = json.loads((output / 'normalization_stats.json').read_text())
        self.assertEqual(set(stats), {'panda_grasp_v1'})
        self.assertEqual(stats['panda_grasp_v1']['action']['mask'], [True] * 6 + [False])
        self.assertLessEqual(max(stats['panda_grasp_v1']['action']['max'][:3]), .0100000001)
        self.assertFalse(result['ready_for_finetuning'])

    def test_modified_or_rejected_transition_cannot_enter_dataset(self):
        path = self.folder / 'transitions.jsonl'
        original = path.read_bytes()
        try:
            rows = [json.loads(line) for line in original.decode().splitlines()]
            rows[0]['executed'] = False
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'changed since replay'):
                prepare_dataset([self.folder], Path(self.directory.name) / 'bad_dataset')
        finally:
            path.write_bytes(original)

    def test_incompatible_contract_rejected(self):
        path = self.folder / 'episode.json'
        original = path.read_bytes()
        try:
            value = json.loads(original)
            value['action_contract']['translation_frame'] = 'gripper local'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Incompatible convention'):
                prepare_dataset([self.folder], Path(self.directory.name) / 'wrong_contract')
        finally:
            path.write_bytes(original)

    def test_gripper_and_reduced_command_round_trip(self):
        proposed = np.array([.03, .04, 0, 0, 0, .1, 0])
        accepted = to_controller_action(proposed)
        accepted[:6] *= .125
        label = from_controller_action(accepted)
        np.testing.assert_allclose(to_controller_action(label), accepted)
        self.assertEqual(label[6], 0)
        self.assertEqual(action_contract()['control_offset_body_metres'], [0, 0, .103])

    def test_combined_world_rotation_keeps_grasp_point_target_fixed(self):
        import mujoco
        from panda_env import PandaEnv
        with PandaEnv() as env:
            position, _ = env.controller.end_effector_pose()
            before = env.data.xmat[env.controller.hand_id].reshape(3, 3).copy()
            x, y, z = .015, -.02, .025
            cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            result = env.controller.apply_action(to_controller_action([0, 0, 0, x, y, z, 1]))
            self.assertTrue(result.converged)
            target = np.empty(9)
            mujoco.mju_quat2Mat(target, env.controller.target_quaternion)
            np.testing.assert_allclose(target.reshape(3, 3), rz @ ry @ rx @ before, atol=1e-12)
            np.testing.assert_allclose(env.controller.target_position, position, atol=1e-12)

    def test_modified_image_invalidates_dataset_eligibility(self):
        path = self.folder / 'step_0001/before.png'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'changed since replay'):
                prepare_dataset([self.folder], Path(self.directory.name) / 'changed_image')
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
