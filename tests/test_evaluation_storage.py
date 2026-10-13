from pathlib import Path
import tempfile
import unittest
from cloud.evaluation_storage import cleanup
from cloud.cloud_session import bootstrap_script


class EvaluationStorageTests(unittest.TestCase):
    def test_only_disposable_directories_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disposable = root / 'panda-evaluation-A1b2C3d4'
            disposable.mkdir()
            (disposable / 'partial-model').write_text('partial')
            for name in ('openvla-7b', 'panda-training', 'panda-evaluation-backup'):
                (root / name).mkdir()
            (root / 'panda-evaluation-12345678').symlink_to(root / 'panda-training', target_is_directory=True)
            self.assertEqual(cleanup(root), [disposable.name])
            self.assertTrue((root / 'panda-training').exists())
            self.assertTrue((root / 'panda-evaluation-12345678').is_symlink())
            self.assertTrue((root / 'panda-evaluation-backup').exists())
            self.assertEqual(cleanup(root), [])

    def test_cleanup_after_stop_before_merge_and_after_reuse_exit(self):
        script = bootstrap_script(1800, '/workspace/checkpoint')
        self.assertLess(script.index('exit 0'), script.index('cloud.evaluation_storage'))
        self.assertLess(script.index('bash scripts/cloud_stop.sh'), script.index('cloud.evaluation_storage'))
        self.assertLess(script.index('cloud.evaluation_storage'), script.index('training.merge_adapter'))
