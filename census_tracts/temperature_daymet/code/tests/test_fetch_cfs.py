"""Verify durable fetch promotion, failure containment, authentication stop and resume."""
import hashlib, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import daymet_fetch as fetch

class DirectArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.archive=self.root/'archive'
        self.target=self.archive/'daily/1980/source.nc';self.payload=b'original-netcdf-byte-stream'
        self.record=dict(filename='source.nc',year=1980,status='planned',url='https://example.invalid/source.nc',cfs_path=str(self.target),cmr_bytes=len(self.payload),cmr_checksum_algorithm='SHA-256',cmr_checksum=hashlib.sha256(self.payload).hexdigest())
        self.manifest=self.root/'manifest.json';fetch.atomic_json(self.manifest,dict(files=[self.record]))
        self.args=SimpleNamespace(command='fetch-cfs',manifest=str(self.manifest),staging=str(self.root/'scratch'))
        self.cfs_patch=patch.object(fetch,'CFS',self.archive);self.cfs_patch.start();fetch._stop.clear()
    def tearDown(self):
        fetch._stop.clear();self.cfs_patch.stop();self.temp.cleanup()
    def session(self,payload=None,status=200):
        response=MagicMock();response.__enter__.return_value=response
        response.status_code=status;response.headers={'Content-Type':'application/octet-stream'}
        response.iter_content.return_value=[self.payload if payload is None else payload]
        session=MagicMock();session.__enter__.return_value=session;session.get.return_value=response
        return session
    def test_original_bytes_promoted_and_manifest_verified(self):
        with patch.object(fetch.requests,'Session',return_value=self.session()):fetch.fetch_one(self.record,self.args)
        self.assertEqual(self.target.read_bytes(),self.payload)
        got=json.loads(self.manifest.read_text())['files'][0]
        self.assertEqual(got['status'],'verified_cfs');self.assertEqual(got['sha256'],self.record['cmr_checksum'])
        self.assertFalse(self.target.with_name('source.nc.part').exists());self.assertFalse(Path(self.args.staging).exists())
    def test_checksum_failure_cannot_promote_source(self):
        with patch.object(fetch.requests,'Session',return_value=self.session(b'x'*len(self.payload))):
            with self.assertRaises(ValueError):fetch.fetch_one(self.record,self.args)
        self.assertFalse(self.target.exists());self.assertEqual(json.loads(self.manifest.read_text())['files'][0]['status'],'planned')
    def test_authentication_failure_stops_without_source(self):
        with patch.object(fetch.requests,'Session',return_value=self.session(status=401)):
            with self.assertRaises(fetch.AuthenticationFailure):fetch.fetch_one(self.record,self.args)
        self.assertTrue(fetch._stop.is_set());self.assertFalse(self.target.exists())
    def test_verified_archive_resume_never_requests_network(self):
        self.target.parent.mkdir(parents=True);self.target.write_bytes(self.payload)
        self.record.update(status='verified_cfs',sha256=self.record['cmr_checksum']);fetch.atomic_json(self.manifest,dict(files=[self.record]))
        with patch.object(fetch.requests,'Session') as session:fetch.fetch_one(self.record,self.args)
        session.assert_not_called()
