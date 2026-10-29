import json,os,pathlib,socket,subprocess,time,urllib.request
root=pathlib.Path(r'D:\AI\ninfer-strata-test'); out=root/'evidence/iq4xs-20261029'; py=root/'.venv/Scripts/python.exe'; env=os.environ.copy(); env['PATH']=str(root/'.cuda/bin/x64')+os.pathsep+env['PATH']; port=18105
with socket.socket() as s:
 if s.connect_ex(('127.0.0.1',port))==0: raise RuntimeError('port occupied')
cfg=json.loads((root/'comparison-strata-native/strata-sage.json').read_text());cfg['args'].remove('--no-graphs');cfg['env']['STRATA_SAGE_RMSNORM']='1';cfg['env']['STRATA_SAGE_TRACE']='1';cfg['port']=port;cfg['log']=str(out/'unsandboxed-5-rms-graphs-engine.log')
p=out/'unsandboxed-5-rms-graphs-config.json';p.write_text(json.dumps(cfg,indent=2))
with (out/'unsandboxed-5-rms-graphs-server.log').open('w',encoding='utf8') as log:
 server=subprocess.Popen([str(py),'-m','serve.server','--engine','strata','--config',str(p),'--host','127.0.0.1','--port',str(port)],cwd=root/'.deps/PolyStrata',env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  start=time.monotonic();deadline=start+180
  while time.monotonic()<deadline:
   if server.poll() is not None:raise RuntimeError(f'server exited {server.returncode}')
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as res:
     if res.status==200:break
   except Exception:time.sleep(2)
  else:raise RuntimeError('startup timeout')
  print('candidate ready',round(time.monotonic()-start,1),flush=True)
  subprocess.run([str(py),str(out/'compare_engines.py'),'--name','unsandboxed-5-rms-graphs','--port',str(port),'--process','strata-poly.exe','--output',str(out/'unsandboxed-5-rms-graphs-comparison.json')],cwd=root,env=env,check=True,timeout=240)
 finally:subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],capture_output=True,timeout=15)



