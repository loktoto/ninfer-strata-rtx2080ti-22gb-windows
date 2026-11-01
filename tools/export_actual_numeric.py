"""Export numeric-only long-context diagnostics, preserving limitations and all samples."""
import argparse,json,hashlib,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
result={'schema_version':1,'scope':'Exploratory one-start-per-profile diagnostics, not a qualified 2x benchmark.',
'limitations':['Synthetic long prompt; no replay of private user tool conversation.','81920 configured context instead of live 262144; same 70038 formatted prompt tokens in all observed profiles.','128-token bounded replies clamp Strata thinking to 64 and inject a close; llama.cpp handling differs.','Second Strata request reused prefix despite cache_prompt=false; llama.cpp remained cold.','Sampling outputs can differ. Q8 KV changes precision.','Telemetry includes startup and prefill; minimum free RAM is not a decode-only metric.'],
'profiles':[]}
for folder in a.folders:
 x=json.loads((folder/'results.json').read_text())
 for prof in x['profiles']:
  rows=[]
  for r in prof['requests']:
   rows.append({k:v for k,v in r.items() if k not in ('content','reasoning')}|{'content_sha256':hashlib.sha256(r.get('content','').encode()).hexdigest(),'reasoning_sha256':hashlib.sha256(r.get('reasoning','').encode()).hexdigest()})
  t=prof['telemetry']
  result['profiles'].append({'profile':prof['profile'],'requests':rows,'error':prof.get('error'),
   'min_available_ram_mib':min((v['available_ram']/1048576 for v in t),default=None),
   'gpu_samples':t,'decode_median':statistics.median([r['timings']['predicted_per_second'] for r in rows if r.get('done') and r.get('timings')]) if any(r.get('done') and r.get('timings') for r in rows) else None})
r=Path(a.folders[0]/'user-run-numeric.json')
if r.exists():
 u=json.loads(r.read_text());result['user_run']={k:u[k] for k in ('source_sha256','context_bins','limitations')}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{'profile':p['profile'],'median':p['decode_median'],'error':p['error']} for p in result['profiles']]))

