#!/usr/bin/env python3
"""Read-only audit of the archived 2026-09-25 paired probability diagnostic.
Run on rokko1; prints JSON and never changes simulation files.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json, hashlib, datetime, re
base=Path('/home/IM25D029/PyBCA_workspcace/bca-ip-statistics-20260924')
errors=[]
def scan(p,selected=None,horizon=None,keep_rows=False):
 m=json.loads((p/'manifest.json').read_text())
 ids=set(m['trial_ids']); counts=Counter(); groups=Counter(); per=defaultdict(Counter); first={}; last={}; rows=[]; total=0; step=0; seen=set(); hashes=[]
 for c in m['chunks']:
  data=(p/c['path']).read_bytes(); digest=hashlib.sha256(data).hexdigest()
  if digest!=c['sha256']: errors.append(str(p/c['path'])+': sha256 mismatch')
  rs=[json.loads(l) for l in data.splitlines()]
  if c['start_step']!=step: errors.append(str(p)+': chunk gap')
  if rs[0]['__chunk__']['start_step']!=c['start_step'] or rs[0]['__chunk__']['next_step']!=c['next_step']: errors.append(str(p)+': header mismatch')
  if len(rs)-1!=c['records']: errors.append(str(p)+': record mismatch')
  hashes.append({'path':c['path'],'sha256':digest}); step=c['next_step']
  for r in rs[1:]:
   if r['kind']!='event': errors.append(str(p)+': unexpected record'); continue
   if r['trial'] not in ids or not c['start_step']<=r['step']<c['next_step'] or r['name'] not in m['events']: errors.append(str(p)+': invalid event')
   k=(r['trial'],r['step'],r['name'])
   if k in seen: errors.append(str(p)+': duplicate event')
   seen.add(k)
   if selected is not None and r['trial'] not in selected: continue
   if horizon is not None and r['step']>=horizon: continue
   total+=1; name=r['name']; n=r['count']; counts[name]+=n; per[r['trial']][name]+=n
   first[name]=min(first.get(name,r['step']),r['step']);last[name]=max(last.get(name,r['step']),r['step'])
   group=('td_to_fsm' if '_core_input_' in name else 'fsm_to_amp' if re.fullmatch('[AB]_x[1-6]output',name) else 'amp_output' if '_Amp_x' in name else 'unit_output' if name.startswith('F_value_') else 'comparator' if name.startswith('Comparate_') else 'reset')
   groups[group]+=n
   if keep_rows: rows.append(r)
 if step!=m['next_step']: errors.append(str(p)+': manifest step mismatch')
 out={'directory':str(p),'observed_steps':m['next_step'],'selected_horizon':horizon,'selected_trial_ids':sorted(selected if selected is not None else ids),'global_prob':m['identity']['global_prob'],'identity':m['identity'],'manifest_sha256':hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),'checked_chunks':len(hashes),'records':total,'counts':dict(sorted(counts.items())),'group_counts':dict(groups),'per_trial':{str(k):dict(v) for k,v in per.items()},'first_event_step_zero_based':first,'last_event_step_zero_based':last}
 if keep_rows: out['rows']=rows
 sp=p/'summary.json'
 if sp.exists(): out['summary']=json.loads(sp.read_text())
 return out
p05=scan(base/'results/probe-global-prob-0.5-20260925',keep_rows=True,horizon=100000)
control=scan(base/'results/bca-ip-512-trials-20260924/rank_0000',selected={0,1},horizon=100000,keep_rows=True)
if p05['identity']['inputs']!=control['identity']['inputs']: errors.append('input mismatch')
if p05['identity']['implementation']!=control['identity']['implementation']: errors.append('implementation mismatch')
for key in ['seed','rng_mode','rng_version','model','scheme']:
 if p05['identity'][key]!=control['identity'][key]: errors.append('identity mismatch:'+key)
old=[]
for p in sorted((base/'results/bca-ip-512-trials-20260924').glob('rank_[0-9][0-9][0-9][0-9]')):
 d=scan(p);d.pop('identity');d.pop('per_trial');old.append(d)
all_ids=[i for d in old for i in d['selected_trial_ids']]
if all_ids!=list(range(512)): errors.append('old trial partition mismatch')
print(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'comparison':'same seed 20260924 and global trial IDs 0,1 for 100000 updates; identical inputs and implementation hashes; independent Philox partition invariant','global_prob_0_5':p05,'global_prob_1_0_control':control,'stopped_global_prob_1_0_ranks':old,'errors':errors},indent=2))

raise SystemExit(1 if errors else 0)
