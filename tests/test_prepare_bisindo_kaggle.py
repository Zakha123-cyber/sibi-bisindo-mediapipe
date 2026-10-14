import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_bisindo_kaggle import component_ids,assign_group_splits,LETTERS


def row(path,label,split,phash,digest):
    return {'path':path,'label':label,'original_split':split,'phash64':f'{phash:016x}',
        'sha256':digest,'include':'1','split':'','source_group':''}


class KaggleBisindoGroupingTests(unittest.TestCase):
    def test_near_duplicate_components_stay_within_class(self):
        rows=[row('a1','A','train',0,'0'*64),row('a2','A','val',3,'1'*64),
              row('a3','A','train',63,'2'*64),row('b1','B','val',0,'3'*64)]
        groups,stats=component_ids(rows,max_distance=2)
        self.assertEqual(groups['a1'],groups['a2'])
        self.assertNotEqual(groups['a1'],groups['a3'])
        self.assertNotEqual(groups['a1'],groups['b1'])
        self.assertEqual(stats['A']['cross_original_split_groups'],1)

    def test_split_assignment_is_deterministic_and_keeps_groups_intact(self):
        rows=[]
        for letter in LETTERS:
            for i in range(10):
                r=row(f'{letter}{i}',letter,'train',i,f'{i:064x}')
                r['source_group']=f'{letter}:group{i}'
                rows.append(r)
        groups={r['path']:r['source_group'] for r in rows}
        config={'seed':42,'split_fractions':{'train':.7,'val':.15,'test':.15}}
        first=assign_group_splits([dict(r) for r in rows],groups,config)
        second=assign_group_splits([dict(r) for r in reversed(rows)],groups,config)
        a={r['path']:r['split'] for r in first};b={r['path']:r['split'] for r in second}
        self.assertEqual(a,b)
        for letter in LETTERS:
            self.assertEqual({a[f'{letter}{i}'] for i in range(10)},{'train','val','test'})

    def test_balanced_v2_assigns_source_groups_to_each_split(self):
        rows=[];groups={}
        for group_index in range(40):
            size=30 if group_index==0 else 1
            for member in range(size):
                path=f'A-{group_index}-{member}'
                rows.append(row(path,'A','train',group_index,f'{group_index:064x}'))
                groups[path]=f'A:group{group_index}'
        config={'seed':42,'split_method':'balanced_group_count_v2',
            'split_fractions':{'train':.7,'val':.15,'test':.15}}
        assigned=assign_group_splits(rows,groups,config)
        grouped={}
        for item in assigned:grouped.setdefault(item['source_group'],set()).add(item['split'])
        self.assertTrue(all(len(splits)==1 for splits in grouped.values()))
        counts={split:sum(next(iter(splits))==split for splits in grouped.values())
            for split in ('train','val','test')}
        self.assertEqual(counts,{'train':28,'val':6,'test':6})


if __name__=='__main__':unittest.main()
