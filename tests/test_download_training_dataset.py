"""Verify checksum gates and non-overwriting dataset publication locally."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from cloud.download_training_dataset import verify_bundle,finalize_download,load_hf_token,copy_verified_bundle
from cloud.download_training_dataset import bundle_source, BOWL_REPOSITORY, BOWL_REVISION


class DownloadDatasetTests(unittest.TestCase):
    def test_bowl_uses_verified_root_repository_and_old_bundles_keep_prefixes(self):
        self.assertEqual(bundle_source('training_bowl_v1'),(BOWL_REPOSITORY,BOWL_REVISION,''))
        self.assertEqual(bundle_source('training_v1')[2],'training_v1/')
        self.assertEqual(bundle_source('training_v2')[2],'training_v2/')
        self.assertEqual(bundle_source('training_bowl_v1','custom/repo','custom-sha'),
                         ('custom/repo','custom-sha',''))

    def test_global_copy_keeps_source_and_publishes_verification_last(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'staging';expected=self.bundle(source)
            destination=root/'global'/'training_v1';copied=[]
            copy=shutil.copyfile
            def record(src,dst):
                copied.append(src.name)
                if src.name!='verification.json':self.assertFalse((destination/'verification.json').exists())
                return copy(src,dst)
            with patch('cloud.download_training_dataset.shutil.copyfile',side_effect=record), \
                 patch.object(Path,'rename',side_effect=AssertionError('Global copy must not rename')):
                self.assertEqual(copy_verified_bundle(source,destination,expected),2)
            self.assertTrue(source.exists())
            self.assertEqual(copied[-1],'verification.json')
            self.assertEqual((destination/'data.bin').read_bytes(),b'image-instruction-action')

    def test_global_copy_rejects_corruption_before_creating_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'staging';expected=self.bundle(source)
            (source/'data.bin').write_bytes(b'broken')
            with self.assertRaises(ValueError):copy_verified_bundle(source,root/'global',expected)
            self.assertFalse((root/'global').exists())

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
