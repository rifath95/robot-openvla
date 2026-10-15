import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_tools.download_sweep_archive import download, retained_bytes


class DownloadProgressTests(unittest.TestCase):
    def test_resume_and_completion_do_not_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / 'archive'
            partial = Path(directory) / 'archive.incomplete'
            rows = [(str(final), str(partial), 100)]
            self.assertEqual(retained_bytes(rows), 0)
            partial.write_bytes(b'x' * 40)
            self.assertEqual(retained_bytes(rows), 40)
            partial.write_bytes(b'x' * 100)
            partial.replace(final)
            self.assertEqual(retained_bytes(rows), 100)
            partial.write_bytes(b'x' * 40)
            self.assertEqual(retained_bytes(rows), 100)

    def test_wrong_size_final_and_oversized_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / 'archive'
            partial = Path(directory) / 'archive.incomplete'
            final.write_bytes(b'x' * 5)
            rows = [(str(final), str(partial), 100)]
            self.assertEqual(retained_bytes(rows), 0)
            partial.write_bytes(b'x' * 150)
            self.assertEqual(retained_bytes(rows), 100)

    def test_download_reuses_hf_cache_pins_revision_and_limits_workers(self):
        with patch.dict(os.environ), patch('huggingface_hub.snapshot_download') as snapshot:
            download('owner/repo', 'commit', 'existing-folder', ['sweep/file'], 2)
            self.assertEqual(os.environ['HF_HUB_DISABLE_XET'], '1')
            self.assertEqual(os.environ['HF_HUB_ENABLE_HF_TRANSFER'], '0')
            snapshot.assert_called_once_with('owner/repo', repo_type='dataset',
                revision='commit', local_dir='existing-folder',
                allow_patterns=['sweep/file'], max_workers=2)


if __name__ == '__main__':
    unittest.main()
