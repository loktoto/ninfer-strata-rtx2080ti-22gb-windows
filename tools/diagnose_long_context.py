"""Bounded long-context diagnostic using synthetic data; never executes model tool calls."""
import argparse,json,os,sys,time,subprocess,socket,threading,urllib.request,hashlib
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/PolyStrata/tools'))
from strata_tokenizer import Tokenizer
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--profiles',nargs='+',default=['D-fp16','D-reserve','A-fp16','D-q8'])
p.add_argument('--tokens',type=int,default=70000)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
tok=Tokenizer.from_dir(cfg['tokenizer'])
text='\n'.join(f'Record {i}: component {(i*37)%997}; state checked; value {(i*19)%7919}.' for i in range(12000))
ids=tok.encode(text)[:a.tokens]
context=tok.decode(ids)
prompt=context+'\n\nExplain how to validate this inventory systematically. Provide a detailed numbered procedure with at least 30 steps. Do not call tools.'
(a.output/'prompt.txt').write_text(prompt,encoding='utf8')
env=os.environ.copy();env['PATH']=str(ROOT/'.cuda/bin/x64')+os.pathsep+env['PATH'];env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
summary={'synthetic_context_tokens':len(ids),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'profiles':[]}
def save(): (a.output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
for ix,profile in enumerate(a.profiles):
 port=18540+ix
 with socket.socket() as sock:
  if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('port occupied')
 rec={'profile':profile,'requests':[],'telemetry':[]};summary['profiles'].append(rec);save()
 c=json.loads(json.dumps(cfg));c['args'][c['args'].index('--max-context')+1]='81920'
 if profile=='D-reserve': c['args']+=['--vram-reserve-mib','4096']
 if profile=='D-lowram': c['args']+=['--prompt-cache','2','--conversation-cache-mib','0']
 if profile=='D-no-draft':
  c['args']+=['--no-draft']
  c['args'][c['args'].index('--window')+1]='1'
 if profile=='D-q8':c['args'][c['args'].index('--kv')+1]='q8_0'
 c['port']=port
 if profile.startswith('A'):
  cmd=[str(ROOT.parent/'LLM/llama.cpp/llama-server.exe'),'-m',c['args'][1],'-ngl','99','-c','81920','-t','16','-np','1','-b','512','-ub','128','-fa','on','-ctk','f16','-ctv','f16','--fit','off','--host','127.0.0.1','--port',str(port),'--no-webui','--no-cache-prompt']
  cwd=ROOT
 else:
  cf=a.output/(profile+'-config.json');cf.write_text(json.dumps(c,indent=2))
  cmd=[sys.executable,'-m','serve.server','--engine','strata','--config',str(cf.resolve()),'--host','127.0.0.1','--port',str(port)]
  cwd=ROOT/'.deps/PolyStrata'
 rec['command']=cmd
 stop=threading.Event()
 def monitor():
  while not stop.is_set():
   try:
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,clocks.sm,power.draw,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
    rec['telemetry'].append({'time':time.time(),'gpu':gpu,'available_ram':psutil.virtual_memory().available})
   except Exception:pass
   stop.wait(2)
 with (a.output/(profile+'-server.log')).open('w',encoding='utf8') as log:
  proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:
   deadline=time.monotonic()+180
   while time.monotonic()<deadline:
    if proc.poll() is not None:raise RuntimeError('startup exit '+str(proc.returncode))
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2):break
    except OSError:time.sleep(1)
   else:raise TimeoutError('startup')
   print(profile,'ready',flush=True)
   for iteration in range(2):
    body={'model':'sage-a1.5','messages':[{'role':'user','content':prompt}],'max_tokens':128,'temperature':1.0,'top_p':.95,'top_k':20,'presence_penalty':1.5,'repeat_penalty':1.0,'seed':1,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':True},'reasoning_effort':'high','stream':True,'stream_options':{'include_usage':True}}
    r={'iteration':iteration,'content':'','reasoning':'','done':False};rec['requests'].append(r)
    start=time.monotonic();timer=threading.Timer(240,lambda:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True));timer.start()
    try:
     req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
     with urllib.request.urlopen(req,timeout=180) as response:
      for raw in response:
       line=raw.decode().strip()
       if not line.startswith('data:'):continue
       s=line[5:].strip()
       if s=='[DONE]':r['done']=True;break
       data=json.loads(s)
       if data.get('error'):r['error']=data['error']
       for ch in data.get('choices',[]):
        for field,key in [('content','content'),('reasoning_content','reasoning')]:
         v=ch.get('delta',{}).get(field)
         if isinstance(v,str) and v:
          r.setdefault('ttft',time.monotonic()-start);r[key]+=v
        if ch.get('finish_reason'):r['finish_reason']=ch['finish_reason']
       for key in ('usage','timings'):
        if data.get(key) is not None:r[key]=data[key]
    finally:timer.cancel();r['wall_seconds']=time.monotonic()-start;save()
    print(profile,iteration,r.get('timings'),r.get('usage'),'wall',r['wall_seconds'],flush=True)
  except Exception as ex:rec['error']=repr(ex);print(profile,'ERROR',repr(ex),flush=True)
  finally:
   subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
   stop.set();thread.join(timeout=8);save()
print('completed',flush=True)

