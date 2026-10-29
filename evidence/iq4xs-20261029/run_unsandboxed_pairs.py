import json, os, pathlib, socket, subprocess, time, urllib.request
root=pathlib.Path(r'D:\AI\ninfer-strata-test'); out=root/'evidence/iq4xs-20261029'; py=root/'.venv/Scripts/python.exe'; env=os.environ.copy(); env['PATH']=str(root/'.cuda/bin/x64')+os.pathsep+env['PATH']
for iteration, graphs in enumerate([False, True, True, False],1):
    label=f'unsandboxed-{iteration}-'+('graphs' if graphs else 'nographs'); port=18100+iteration
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',port))==0: raise RuntimeError(f'port {port} already used')
    cfg=json.loads((root/'comparison-strata-native/strata-sage.json').read_text()); cfg['port']=port
    if graphs: cfg['args'].remove('--no-graphs')
    p=out/(label+'-config.json'); p.write_text(json.dumps(cfg,indent=2))
    with (out/(label+'-server.log')).open('w',encoding='utf8') as log:
        server=subprocess.Popen([str(py),'-m','serve.server','--engine','strata','--config',str(p),'--host','127.0.0.1','--port',str(port)],cwd=root/'.deps/PolyStrata',env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic(); deadline=start+180
            while time.monotonic()<deadline:
                if server.poll() is not None: raise RuntimeError(f'{label}: server exited {server.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as res:
                        if res.status==200: break
                except Exception: time.sleep(2)
            else: raise RuntimeError(f'{label}: startup timeout')
            print(label,'ready seconds',round(time.monotonic()-start,1),flush=True)
            subprocess.run([str(py),str(out/'compare_engines.py'),'--name',label,'--port',str(port),'--process','strata-poly.exe','--output',str(out/(label+'-comparison.json'))],cwd=root,env=env,check=True,timeout=240)
        finally:
            subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],capture_output=True,timeout=15)

