"""Verify evidence rejection and publication recovery without network access."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import sync_results as sync


class EvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = sync.DESTINATION
        if not (cls.evidence/'transfer.json').exists():
            raise unittest.SkipTest('Run sync first to obtain real server evidence')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.destination = Path(self.temporary.name)/'evidence'
        shutil.copytree(self.evidence, self.destination)
        self.override = patch.object(sync, 'DESTINATION', self.destination)
        self.override.start()
        self.snapshot = sync.load('transfer.json')

    def tearDown(self):
        self.override.stop()
        self.temporary.cleanup()

    def change(self, name, mutate):
        value = sync.load(name)
        mutate(value)
        sync.json_write(self.destination/name, value)

    def test_real_snapshot(self):
        self.assertEqual(sync.verify(self.snapshot)['status'], 'passed')

    def test_missing_metric_rejected(self):
        self.change(sync.RUNS['t2m']+'/summary.json', lambda v: v['metrics'].pop('fid'))
        with self.assertRaisesRegex(ValueError, 'metrics'):
            sync.verify(self.snapshot)

    def test_changed_binding_rejected(self):
        self.change(sync.RUNS['kit']+'/binding.json', lambda v: v.update(seed=1))
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            sync.verify(self.snapshot)

    def test_wrong_aggregate_rejected(self):
        self.change(sync.RUNS['kit']+'/summary.json', lambda v: v['metrics']['fid'].update(mean=999))
        with self.assertRaisesRegex(ValueError, 'summary mismatch'):
            sync.verify(self.snapshot)


class PublicationChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)/'repo'
        self.root.mkdir()
        subprocess.run(['git','init','-b','main',str(self.root)],check=True,capture_output=True)
        (self.root/'reproduction').mkdir()
        (self.root/'.gitignore').write_text('reproduction/artifacts/\n',encoding='utf-8')
        self.status = self.root/'reproduction/STATUS_zh.md'
        self.status.write_text('Initial status\n',encoding='utf-8')
        self.overrides = [patch.object(sync,'REPO',self.root),
                          patch.object(sync,'DESTINATION',self.root/'reproduction/artifacts/evidence'),
                          patch.object(sync,'describe',return_value=('Report\n',[],'')),
                          patch.object(sync,'load',return_value={'fixture':True})]
        for override in self.overrides:
            override.start()
        self.raw_evidence = b'{\n  "fixture": true\n}\n'
        for folder in sync.RUNS.values():
            for name in ['summary.json','binding.json','repetitions.json','metric_audit.json']:
                path = sync.DESTINATION/folder/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(self.raw_evidence)
        sync.json_write(sync.DESTINATION/'transfer.json', {'checked_at':'before'})
        sync.git('add','.')
        sync.git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                 'commit','-m','Initial fixture')
        self.snapshot = {'checked_at':'2026-10-16T00:00:00+00:00'}
        self.verification = {'complete':True}

    def tearDown(self):
        for override in reversed(self.overrides):
            override.stop()
        self.temporary.cleanup()

    def test_dirty_repository_preserved_before_writes(self):
        self.status.write_text('User changes\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'no tracked files written'):
            sync.finalize(self.snapshot,self.verification,False)
        self.assertEqual(self.status.read_text(encoding='utf-8'),'User changes\n')
        self.assertFalse((self.root/'reproduction/results').exists())

    def test_prepared_user_edits_preserved(self):
        sync.finalize(self.snapshot,self.verification,False)
        self.status.write_text('User changes\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Prepared result changed'):
            sync.finalize(self.snapshot,self.verification,False)
        self.assertEqual(self.status.read_text(encoding='utf-8'),'User changes\n')

    def test_server_evidence_bytes_preserved(self):
        sync.finalize(self.snapshot,self.verification,False)
        self.assertEqual((self.root/'reproduction/results/t2m/summary.json').read_bytes(),self.raw_evidence)

    def test_failed_pull_retries_without_duplicate_commit(self):
        real_git = sync.git
        def git_without_network(*arguments, **kwargs):
            return '' if arguments[0]=='push' else real_git(*arguments, **kwargs)
        with patch.object(sync,'git',side_effect=git_without_network):
            with patch.object(sync,'remote',return_value=json.dumps({'exit_code':1,'head':'old'})):
                with self.assertRaisesRegex(ConnectionError,'pull not verified'):
                    sync.finalize(self.snapshot,self.verification,True)
            head = sync.git('rev-parse','HEAD')
            status = self.status.read_text(encoding='utf-8')
            sync.json_write(sync.DESTINATION/'transfer.json', {'checked_at':'after'})
            receipt = {'exit_code':0,'head':head,'output':'Already up to date.','error':''}
            with patch.object(sync,'remote',return_value=json.dumps(receipt)):
                sync.finalize(self.snapshot,self.verification,True)
            self.assertEqual(sync.git('rev-parse','HEAD'),head)
            self.assertEqual(self.status.read_text(encoding='utf-8'),status)
            value = json.loads((sync.DESTINATION/'publication.json').read_text(encoding='utf-8'))
            self.assertEqual(value['status'],'complete')
            self.assertEqual(value['server_commit'],head)

    def test_server_pull_command_is_valid_python(self):
        real_git = sync.git
        def git_without_network(*arguments, **kwargs):
            return '' if arguments[0]=='push' else real_git(*arguments, **kwargs)
        def remote_without_network(code, **kwargs):
            compile(code,'<server-pull>','exec')
            return json.dumps({'exit_code':0,'head':real_git('rev-parse','HEAD')})
        with patch.object(sync,'git',side_effect=git_without_network):
            with patch.object(sync,'remote',side_effect=remote_without_network):
                sync.finalize(self.snapshot,self.verification,True)


if __name__ == '__main__':
    unittest.main()
