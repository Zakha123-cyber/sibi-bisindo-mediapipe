"""Checks for leakage protection and reproducible split construction."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_data', ROOT/'scripts/prepare_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SplitSafetyTests(unittest.TestCase):
    def test_sibi_backgrounds_and_variants_share_capture_group(self):
        keys = [module.identify_source(dict(language='SIBI', origin=value))[0]
                for value in ['A_10','bg_01__A_10','bg_05__A_10']]
        self.assertEqual(len(set(keys)), 1)

    def test_bisindo_series_is_grouped_across_letters(self):
        a = module.identify_source(dict(language='BISINDO',origin='A_0001_2',letter='A'))
        z = module.identify_source(dict(language='BISINDO',origin='Z_0030_2',letter='Z'))
        self.assertEqual(a[0],z[0])
        self.assertIn('not_person_id',a[1])

    def test_unknown_bisindo_series_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            module.identify_source(dict(language='BISINDO',origin='A_0001_new',letter='A'))

    def test_split_is_independent_of_input_order(self):
        config=dict(seed=42,sibi=dict(validation_fraction=.15,test_fraction=.15))
        groups={'A':[f'sibi:A_{i}' for i in range(60)]}
        first=module.assign_sibi_splits(groups,config)
        second=module.assign_sibi_splits({'A':list(reversed(groups['A']))},config)
        self.assertEqual(first,second)
        self.assertEqual(sum(v=='train' for v in first.values()),42)
        self.assertEqual(sum(v=='val' for v in first.values()),9)
        self.assertEqual(sum(v=='test' for v in first.values()),9)

    def test_too_few_groups_fails(self):
        with self.assertRaises(ValueError):
            module.assign_sibi_splits({'A':['g1','g2']},dict(seed=42,sibi=dict(validation_fraction=.15,test_fraction=.15)))

    def test_input_fingerprint_tolerates_path_separators_and_record_order(self):
        windows=[{'path':'folder\\b.jpg','sha256':'b'},{'path':'folder\\a.jpg','sha256':'a'}]
        posix=[{'path':'folder/a.jpg','sha256':'a'},{'path':'folder/b.jpg','sha256':'b'}]
        first=module.input_fingerprints(windows,[{'path':'folder\\a.jpg'}],{'orphan_labels':['labels\\a.txt']},{})
        second=module.input_fingerprints(posix,[{'path':'folder/a.jpg'}],{'orphan_labels':['labels/a.txt']},{})
        self.assertEqual(first,second)
        posix[0]['sha256']='different_content'
        self.assertNotEqual(first,module.input_fingerprints(posix,[{'path':'folder/a.jpg'}],{'orphan_labels':['labels/a.txt']},{}))

    def sample_manifest(self):
        rows=[]
        for language in ['sibi','bisindo']:
            for split in ['train','val','test']:
                key=language+':'+split
                rows.append(dict(language=language,label='A',split=split,source_group=key,
                    sha256=key,include='1',exclusion_reason='',variant='original',signer_id='',session_id=''))
        return rows

    def label_map(self):
        return dict(sibi=dict(annotation_id_to_letter={'0':'A'}),bisindo=dict(candidate_letters=['A']))

    def test_source_leakage_is_rejected(self):
        rows=self.sample_manifest()
        rows[1]['source_group']=rows[0]['source_group']
        with self.assertRaisesRegex(ValueError,'Source group'):
            module.validate_manifest(rows,self.label_map())

    def test_exact_duplicate_leakage_is_rejected(self):
        rows=self.sample_manifest()
        rows[1]['sha256']=rows[0]['sha256']
        with self.assertRaisesRegex(ValueError,'Identical image bytes cross'):
            module.validate_manifest(rows,self.label_map())

    def test_pre_generated_augmentation_is_rejected(self):
        rows=self.sample_manifest()
        rows[0]['variant']='sibi_alphabet_mono_bg_with_augmentation'
        with self.assertRaisesRegex(ValueError,'augmentation'):
            module.validate_manifest(rows,self.label_map())

    def test_missing_class_coverage_is_rejected(self):
        rows=self.sample_manifest()[:-1]
        with self.assertRaisesRegex(ValueError,'Missing class coverage'):
            module.validate_manifest(rows,self.label_map())

    def test_split_lock_rejects_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            lock={'version':'v1','assignment_sha256':'first'}
            (out/'split_lock.json').write_text(json.dumps(lock))
            module.enforce_lock(out,lock)
            with self.assertRaisesRegex(ValueError,'Locked snapshot changed'):
                module.enforce_lock(out,dict(lock,assignment_sha256='different'))

    def test_dataset_path_cannot_escape_root(self):
        for path in ['../secret','C:/secret','/secret','folder/../../secret']:
            with self.subTest(path=path),self.assertRaises(ValueError):
                module.normalized_path(path)

    def test_new_version_can_update_latest_report_but_not_reuse_snapshot_directory(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            (out/'split_lock.json').write_text(json.dumps({'version':'v1','assignment_sha256':'first'}))
            new_lock={'version':'v2','assignment_sha256':'second'}
            with self.assertRaisesRegex(ValueError,'Locked snapshot changed'):
                module.enforce_lock(out,new_lock)
            module.enforce_lock(out,new_lock,latest_report=True)
            with self.assertRaisesRegex(ValueError,'Locked snapshot changed'):
                module.enforce_lock(out,{'version':'v1','assignment_sha256':'changed'},latest_report=True)

    @unittest.skipUnless((ROOT/'reports/dataset_audit/inventory.csv').exists(), 'Local inventory required')
    def test_full_local_manifest_is_order_independent_and_quarantines_sources(self):
        config=json.loads((ROOT/'configs/data_v1.json').read_text())
        labels=json.loads((ROOT/config['label_map']).read_text())
        inventory=module.read_csv(ROOT/config['inventory'])
        decisions=module.read_csv(ROOT/config['curation_decisions'])
        audit=json.loads((ROOT/config['audit_summary']).read_text())
        rows,groups,_=module.build_manifest(inventory,decisions,config,labels,audit['orphan_labels'])
        reversed_rows,_,_=module.build_manifest(list(reversed(inventory)),decisions,config,labels,audit['orphan_labels'])
        self.assertEqual(module.csv_bytes(rows),module.csv_bytes(reversed_rows))
        self.assertEqual(len(groups),7)
        self.assertTrue(all(r['include']=='0' and r['split']=='excluded' for r in rows if r['source_group'] in groups))
        module.validate_manifest(rows,labels)
        changed=copy.deepcopy(decisions)
        changed[0]['image_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'snapshot'):
            module.build_manifest(inventory,changed,config,labels,audit['orphan_labels'])


if __name__ == '__main__':
    unittest.main()
