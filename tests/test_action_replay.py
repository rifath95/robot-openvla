"""Action playback must advance physics continuously, not copy recorded poses."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from data_tools.bowl_demonstrations import configure_recorded_scene, scene_fingerprint
from data_tools.panda_demonstrations import restore_state, save_state
from simulation.action_replay import replay_actions
from simulation.bowl_scene import SCENE_PATH,CAMERA_NAME
from simulation.panda_actions import action_contract,make_controller,to_controller_action
from simulation.replay_run import write_launcher


class ActionReplayTests(unittest.TestCase):
    def episode(self,folder):
        folder.mkdir()
        metadata=dict(scene_profile='panda_bowl_v1',scene_xml_sha256=scene_fingerprint(),
            action_contract=action_contract(camera=CAMERA_NAME),
            scenario=dict(scene=dict(bowl_position=[.62,-.12,.254])))
        (folder/'episode.json').write_text(json.dumps(metadata))
        model=mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        configure_recorded_scene(model,metadata)
        data=mujoco.MjData(model)
        home=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'scene_home')
        mujoco.mj_resetDataKeyframe(model,data,home)
        env=SimpleNamespace(model=model,data=data,controller=make_controller(model,data))
        save_state(folder/'initial_state.npz',env)
        rows=[]
        for index in (1,2):
            action=[.001,0,0,0,0,0,1]
            result=env.controller.apply_action(to_controller_action(action))
            self.assertTrue(result.converged)
            env.controller.step(40)
            directory=folder/f'step_{index:04d}';directory.mkdir()
            save_state(directory/'after_state.npz',env)
            rows.append(dict(index=index,executed=True,action=action,physics_steps=40,stage='move'))
        expected=data.qpos.copy()
        # Deliberately make the saved endpoint wrong. Replay may compare it,
        # but must not restore it into the running simulation.
        path=folder/'step_0001/after_state.npz'
        with np.load(path) as state:contents={k:state[k].copy() for k in state.files}
        contents['qpos'][0]+=.5
        np.savez(path,**contents)
        (folder/'transitions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return expected

    def test_restore_once_execute_labels_and_keep_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'source';expected=self.episode(folder)
            original={p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            output=Path(tmp)/'output'
            with patch('simulation.action_replay.restore_state',wraps=restore_state) as restore:
                report=replay_actions(folder,no_view=True,output_dir=output)
            restore.assert_called_once()
            self.assertEqual(report['completed_actions'],2)
            self.assertGreater(report['maximum_recorded_qpos_difference'],.4)
            with np.load(output/'step_0002/after_state.npz') as state:
                np.testing.assert_allclose(state['qpos'],expected,rtol=0,atol=1e-10)
            for path,content in original.items():self.assertEqual((folder/path).read_bytes(),content)

    def test_partial_replay_reports_actual_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'source';self.episode(folder)
            report=replay_actions(folder,no_view=True,max_steps=1,output_dir=Path(tmp)/'output')
            self.assertEqual(report['outcome'],'partial')
            self.assertEqual(report['completed_actions'],1)

    def test_demo_launcher_selects_action_playback(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);write_launcher(folder,demonstration=True)
            self.assertIn('simulation.action_replay',(folder/'replay_actions.py').read_text())
            self.assertIn('from simulation.replay_run import main',(folder/'replay.py').read_text())

    def test_new_demonstrations_reject_controller_provenance_mismatch(self):
        from data_tools.bowl_demonstrations import controller_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'source';self.episode(folder)
            path=folder/'episode.json';metadata=json.loads(path.read_text())
            metadata['controller_sha256']='different';path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'Controller differs'):
                replay_actions(folder,no_view=True,output_dir=Path(tmp)/'mismatch')
            metadata['controller_sha256']=controller_fingerprint();path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'Controller differs'):
                replay_actions(folder,no_view=True,controller_mode='collection',output_dir=Path(tmp)/'old')


if __name__=='__main__':unittest.main()
