from pathlib import Path
import sys,time,json
sys.path[:0]=[str(Path.cwd()/'src'),str(Path.cwd()/'scripts')]
import numpy as np,torch
from PyBCA.core.simulator import BCA_Simulator
from generate_bca_ip_cellspaces import DEFAULT_OUTPUT,sha256
r=Path('results/cellspace-variants-20261006/smoke');r.mkdir(exist_ok=True)
m=json.loads((DEFAULT_OUTPUT/'manifest.json').read_text());out=[];torch.set_num_threads(1)
for item in m['variants']:
 f=DEFAULT_OUTPUT/item['file'];start=time.monotonic()
 s=BCA_Simulator(str(f),['Sample/rule/base-rule.yaml'],device='cpu',spatial_event_filePath=str(DEFAULT_OUTPUT/m['events']),execution_mode='torch_sparse',rng_mode='independent',trial_ids=[0],quiet=True)
 s.Allocate_torch_Tensors_on_Device();s.set_ParallelTrial(1);initial=s.TCHW.cpu().numpy().copy()
 for j in range(128):s.step(.5,seed=20261006)
 final=s.TCHW.cpu().numpy()
 assert np.array_equal(initial!=0,final!=0)
 assert np.array_equal(initial==-1,final==-1)
 changed=int(np.sum(initial!=final));assert changed>0
 state=r/(f.stem+'.npz');np.savez_compressed(state,cells=final,offset=np.array([s.offset_x,s.offset_y]))
 record={'file':f.name,'sha256':sha256(f),'steps':128,'trials':1,'seed':20261006,'global_prob':.5,'execution_mode':'torch_sparse','rng_mode':'independent','device':'cpu','shape':list(final.shape),'changed_cells':changed,'static_geometry_preserved':True,'event_counts':{k:len(v) for k,v in s.event_history[0].items()},'elapsed_sec':time.monotonic()-start,'passed':True}
 out.append(record);print(json.dumps({k:record[k] for k in ['file','steps','changed_cells','elapsed_sec']}),flush=True)
 (DEFAULT_OUTPUT/'smoke-validation.json').write_text(json.dumps({'scope':'Load and update every integrated map with its companion events. This is not long-run validation.','script_sha256':sha256(Path(__file__)),'runs':out},indent=2)+'\n')
