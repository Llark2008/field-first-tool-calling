import copy
import unittest
from experiments.fields_epochs import train,evaluate

class PipelineTests(unittest.TestCase):
    def test_resume_event_cursor_discards_only_uncheckpointed_updates(self):
        events=[{'microstep':i,'optimizer_updates':i//8,'loss':.1,'supervised_tokens':10} for i in range(8,625,8)]
        retained=train.resume_events(events,600)
        self.assertEqual(len(retained),75);self.assertEqual(retained[-1]['microstep'],600)
        with self.assertRaises(ValueError):train.resume_events(events[1:],600)
        bad=copy.deepcopy(events);bad[3]['loss']=float('nan')
        with self.assertRaises(ValueError):train.resume_events(bad,600)

    def test_checkpoint_selection_does_not_reinitialize_A_at_epoch_two(self):
        from pathlib import Path
        root=Path('/tmp/epoch-test');parent=Path('/tmp/parent')
        self.assertEqual(train.resume_origin(root,parent,'A',[]),(parent/'train-F/step-0600',600,True))
        self.assertEqual(train.resume_origin(root,parent,'B',[]),(None,0,False))
        states=[root/'train-A/step-0800',root/'train-A/step-1000']
        self.assertEqual(train.resume_origin(root,parent,'A',states),(states[-1],1000,False))

    def test_raw_resume_is_bound_to_same_case_order_and_prompt(self):
        cases=[{'id':'one'}];prompts={'one':'hash'}
        row={'case_id':'one','prompt_sha256':'hash','output':'{}','finish_reason':'stop'}
        self.assertEqual(evaluate.validate_raw(cases,[row],prompts),[row])
        for key,value in [('case_id','other'),('prompt_sha256','changed')]:
            with self.assertRaises(ValueError):evaluate.validate_raw(cases,[dict(row,**{key:value})],prompts)
        with self.assertRaises(ValueError):evaluate.validate_raw(cases,[],prompts)

if __name__=='__main__':unittest.main()
