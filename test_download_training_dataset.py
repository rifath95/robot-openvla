"""Verify checksum gates and non-overwriting dataset publication locally."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from download_training_dataset import verify_bundle,finalize_download


class DownloadDatasetTests(unittest.TestCase):
    def bundle(self,root):
        root.mkdir()
        (root/'verification.json').write_text('{"passed": true}')
        (root/'data.bin').write_bytes(b'image-instruction-action')
        checksums=''.join(f'{hashlib.sha256((root/name).read_bytes()).hexdigest()}  {name}\n'
            for name in ('verification.json','data.bin'))
        (root/'SHA256SUMS').write_text(checksums)
        return checksums

    def test_verified_bundle_moves_to_final_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'staging';expected=self.bundle(source)
            self.assertEqual(finalize_download(source,root/'training_v2',expected),2)
            self.assertFalse(source.exists())
            self.assertTrue((root/'training_v2/data.bin').exists())

    def test_corruption_leaves_bundle_in_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'staging';expected=self.bundle(source)
            (source/'data.bin').write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError,'Checksum failed'):
                finalize_download(source,root/'training_v2',expected)
            self.assertFalse((root/'training_v2').exists())

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'staging';expected=self.bundle(source)
            destination=root/'training_v2';destination.mkdir();(destination/'keep').write_text('original')
            with self.assertRaises(ValueError):finalize_download(source,destination,expected)
            self.assertEqual((destination/'keep').read_text(),'original')

    def test_checksum_revision_must_match_and_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'bundle';expected=self.bundle(source)
            with self.assertRaises(ValueError):verify_bundle(source,expected+'other\n')
            invalid='0'*64+'  ../escape\n';(source/'SHA256SUMS').write_text(invalid)
            with self.assertRaises(ValueError):verify_bundle(source,invalid)


if __name__=='__main__':unittest.main()
