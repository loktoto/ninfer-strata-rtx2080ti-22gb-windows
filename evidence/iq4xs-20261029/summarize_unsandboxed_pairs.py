import json,statistics,pathlib
p=pathlib.Path(__file__).parent
names=[f'unsandboxed-{i}-{n}' for i,n in [(1,'nographs'),(2,'graphs'),(3,'graphs'),(4,'nographs')]]
data={n:json.loads((p/(n+'-comparison.json')).read_text(encoding='utf8')) for n in names if (p/(n+'-comparison.json')).exists()}
s={}
for name,d in data.items():
  s[name]={}
  for case in ('arithmetic','python','retrieval'):
    rows=[x for x in d['results'] if x['case']==case and x['iteration']>1]
    if not rows: continue
    s[name][case]={'decode_tps':[x['timings']['predicted_per_second'] for x in rows], 'prompt_tps':[x['timings']['prompt_per_second'] for x in rows], 'ttft':[x['ttft_seconds'] for x in rows], 'tokens':[(x['usage']['prompt_tokens'],x['usage']['completion_tokens']) for x in rows], 'all_complete':all(x['done'] and x['finish_reason']=='stop' for x in rows)}
if len(data)==4:
  s['outputs_equal']=all([x['content'] for x in d['results']]==[x['content'] for x in data[names[0]]['results']] for d in data.values())
(p/'unsandboxed-paired-summary.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf8')
print(json.dumps(s,indent=2))

