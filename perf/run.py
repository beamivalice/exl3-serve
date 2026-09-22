import datetime,json,os,re,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent.parent
name=sys.argv[1]
state=root/'perf'/f'{name}.state.jsonl'
def snapshot():
    p=subprocess.run(['pgrep','-x','mlx-serve'],capture_output=True,text=True)
    c=subprocess.run(['pgrep','-f','imatrix_serve|convert_mimo_v26_exl3|convert_qwen38_flash_next_exl3'],capture_output=True,text=True)
    conv=[]
    for pid in c.stdout.split():
        comm=subprocess.run(['ps','-p',pid,'-o','comm='],capture_output=True,text=True).stdout.strip()
        if 'python' in comm.lower():
            argv=subprocess.run(['ps','-p',pid,'-o','args='],capture_output=True,text=True).stdout.strip()
            conv.append({'pid':pid,'args':argv})
    i=subprocess.run(['ioreg','-r','-d','1','-c','IOAccelerator'],capture_output=True,text=True)
    util=[int(v) for v in re.findall(r'"Device Utilization %"=(\d+)',i.stdout)]
    d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),servers=p.stdout.strip(),converters=conv,util=util)
    with state.open('a') as f: f.write(json.dumps(d)+'\n')
    return d
before=snapshot()
if before['servers'] or before['converters'] or not before['util'] or max(before['util'])>10:
    raise SystemExit('Box busy; no measurement started')
env=os.environ.copy()
env['PERF_REPORT']='1'
if 'model' not in name: env['MLX_SERVE_EXL3_LAYER_UBENCH']='1'
else: env.pop('MLX_SERVE_EXL3_LAYER_UBENCH',None)
cmd=sys.argv[2:]
with (root/'perf'/f'{name}.log').open('w') as out:
    p=subprocess.Popen(cmd,env=env,stdout=out,stderr=out,cwd=root)
    start=time.monotonic()
    while p.poll() is None:
        time.sleep(1)
        d=snapshot()
        if d['servers'] or d['converters'] or time.monotonic()-start>(1800 if "model" in name else 600):
            p.terminate();p.wait();raise SystemExit('INVALID: concurrent model/converter or timeout')
    code=p.returncode
after=snapshot()
(root/'perf'/f'{name}.meta.json').write_text(json.dumps(dict(before=before,after=after,command=cmd,exit_code=code,env={k:env[k] for k in ['PERF_REPORT','PERF_MIMO_MODEL','MLX_ENABLE_TF32','MLX_SERVE_EXL3_LAYER_UBENCH'] if k in env},boot=subprocess.check_output(['sysctl','-n','kern.boottime'],text=True).strip()),indent=2))
raise SystemExit(code)
