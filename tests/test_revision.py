import json
import tempfile
import unittest
from pathlib import Path
from programs import protocol, judge, agent_eval as agent, safety_eval as safety
class RevisionTests(unittest.TestCase):
 def test_manifest(self):
  with tempfile.TemporaryDirectory() as t:
   f=protocol.demo(t);self.assertEqual(json.loads((Path(t)/'manifest.json').read_text())['configuration_id'],f)
   rows=[json.loads(s) for s in (Path(t)/'observations.jsonl').read_text().splitlines()]
   self.assertEqual([r['status'] for r in rows],['completed','timeout']);self.assertEqual(rows[1]['score'],0)
 def test_bad_judge(self):
  task=judge.Task('t','q','a',{},'1')
  for s in ['{','{}','[]']:self.assertEqual(judge.evaluate_result(s,task)['status'],'invalid_judge_output')
 def test_agent_error(self):
  def broken(h):raise RuntimeError()
  r,e=agent.run_episode(agent.Environment('a','s'),broken,'x','e')
  self.assertEqual(r['status'],'agent_error');self.assertEqual((e[0]['episode_id'],e[0]['scenario_id'],e[0]['step_id']),('e','s',0))
 def test_invalid_action(self):
  _,e=agent.run_episode(agent.Environment('a','s'),lambda _:None,'x','e')
  self.assertEqual(e[0]['error_type'],'InvalidAction');self.assertEqual(e[0]['episode_id'],'e')
 def test_progress_and_save(self):
  with tempfile.TemporaryDirectory() as t:
   r=agent.demo(t);self.assertEqual(r['good']['progress'],1);self.assertEqual(r['early']['progress'],0)
   es=[json.loads(s) for s in (Path(t)/'agent_events.jsonl').read_text().splitlines()]
   self.assertTrue(es);self.assertTrue(all({'episode_id','scenario_id','step_id'}<=e.keys() for e in es))
 def test_canary(self):
  def a(**kw):return safety.Attempt('a','c','f',False,True,'MARK','MARK',(),False,**kw)
  self.assertFalse(safety.assess(a())['violation']);self.assertFalse(safety.assess(a(input_text='MARK',disclosure_confirmed=True))['violation']);self.assertTrue(safety.assess(a(disclosure_confirmed=True))['violation'])
