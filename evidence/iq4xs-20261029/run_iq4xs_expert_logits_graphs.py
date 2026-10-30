import json, os, pathlib, subprocess, time, importlib.util
base=pathlib.Path(__file__).resolve().parent; root=base.parents[1]; out=base/'iq4xs-expert-logits-graphs';out.mkdir(exist_ok=True)
report=json.loads((base/'rmsnorm-integration/report.json').read_text(encoding='utf8')); results={}
for name in ['baseline','candidate']:
 launch=report['launches']['baseline']; argv=launch['argv'].copy();argv.remove('--no-graphs');argv.remove('--dump-layers');argv[argv.index('--logits-file')+1]=str(out/(name+'.logits'))
 env=dict(os.environ,**launch['controlled_environment']);env['PATH']=str(root/'.cuda/bin/x64')+os.pathsep+env['PATH'];env['STRATA_SAGE_EXPERT_IQ4XS']='1' if name=='candidate' else '0';env['STRATA_SAGE_TRACE']='1'
 with (out/(name+'.stdout.txt')).open('wb') as stdout,(out/(name+'.stderr.txt')).open('wb') as stderr:
  t=time.monotonic(); p=subprocess.run(argv,env=env,stdout=stdout,stderr=stderr,timeout=600);results[name]={'exit':p.returncode,'seconds':time.monotonic()-t,'argv':argv,'env':{k:v for k,v in env.items() if k.startswith('STRATA_')}}
 if p.returncode: raise RuntimeError(str(results[name]))
spec=importlib.util.spec_from_file_location('compare_logits',root/'tools/compare_logits.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);results['comparison']=m.compare(out/'baseline.logits',out/'candidate.logits',1e-4,1e-3);(out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf8'); print(json.dumps(results['comparison'],indent=2))
