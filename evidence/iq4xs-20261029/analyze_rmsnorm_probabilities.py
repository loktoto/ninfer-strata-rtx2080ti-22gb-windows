import json,struct,pathlib,numpy as np
p=pathlib.Path(__file__).parent/'rmsnorm-integration'
def load(n):
 with (p/(n+'.logits')).open('rb') as f:
  magic=f.read(8); ver,vocab,prompt,steps=struct.unpack('<IIII',f.read(16)); f.seek(4*(prompt+steps),1); return np.fromfile(f,dtype='<f4').reshape(steps,vocab).astype(np.float64)
a,b=load('baseline'),load('candidate'); out=[]
for x,y in zip(a,b):
 x=x-np.max(x);y=y-np.max(y); px=np.exp(x);py=np.exp(y);px/=px.sum();py/=py.sum();
 out.append({'kl_baseline_candidate':float(np.sum(px*(np.log(px+1e-300)-np.log(py+1e-300)))), 'total_variation':float(np.abs(px-py).sum()/2), 'baseline_top1_probability':float(px.max()),'candidate_top1_probability':float(py.max()),'top1_equal':bool(np.argmax(px)==np.argmax(py))})
result={'temperature':1.0,'interpretation':'Probability diagnostics on teacher-forced stored logits, not a root-cause isolation or sampled-output quality proof.','steps':out}
(pathlib.Path(__file__).parent/'rmsnorm-probability-analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8'); print('KL max',max(x['kl_baseline_candidate'] for x in out),'TV max',max(x['total_variation'] for x in out),'TV median',np.median([x['total_variation'] for x in out]))

