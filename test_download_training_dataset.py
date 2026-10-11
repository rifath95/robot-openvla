"""Verify checksum gates and non-overwriting dataset publication locally."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from download_training_dataset import verify_bundle,finalize_download,load_hf_token


class DownloadDatasetTests(unittest.TestCase):
    def test_recovers_only_hf_token_from_container_when_ssh_omits_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'environ'
            path.write_bytes(b'RUNPOD_API_KEY=other-secret\0HF_TOKEN=test-read-token\0PATH=/bin\0')
            env={}
            self.assertEqual(load_hf_token(env,path),'container environment')
            self.assertEqual(env,{'HF_TOKEN':'test-read-token'})

    def test_explicit_shell_token_takes_precedence(self):
        env={'HF_TOKEN':' shell-token '}
        self.assertEqual(load_hf_token(env,Path('/missing')),'shell environment')
        self.assertEqual(env['HF_TOKEN'],'shell-token')

    def test_missing_and_unresolved_secrets_are_reported_without_values(self):
        with self.assertRaisesRegex(ValueError,'missing'):
            load_hf_token({},Path('/missing'))
        value='{{ RUNPOD_SECRET_huggingface-read }}'
        with self.assertRaises(ValueError) as exc:
            load_hf_token({'HF_TOKEN':value})
        self.assertNotIn(value,str(exc.exception))

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
