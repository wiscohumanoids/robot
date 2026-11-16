"""Read-only, CPU-only attestation of the currently deployed remote base."""
import json
from pathlib import Path
import re
import subprocess
import uuid

REMOTE_AUDIT = r'''
import hashlib,json,struct,subprocess,urllib.request
from pathlib import Path
def health():
 with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=5) as r:return json.load(r)
def identity():
 return subprocess.check_output(['systemctl','--user','show','panthera-inference.service','--property=MainPID,ExecMainStartTimestampMonotonic,InvocationID,ActiveState'],text=True)
a=health(); service=identity(); root=Path(a['checkpoint']).resolve(strict=True)
assert a['status']=='ready' and root.is_relative_to('/mnt/pi05/inference/checkpoints')
files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors'))
assert all(not p.is_symlink() for p in files)
def sig(p):
 s=p.stat();return [s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
signatures={str(p.relative_to(root)):sig(p) for p in files}; hashes={}
for p in files:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 hashes[str(p.relative_to(root))]=h.hexdigest()
cid=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
pid=hashlib.sha256(json.dumps({k:v for k,v in hashes.items() if 'processor' in k or 'normalizer' in k},sort_keys=True).encode()).hexdigest()
pre=json.loads((root/'policy_preprocessor.json').read_text())
steps=[s for s in pre['steps'] if s['registry_name']=='normalizer_processor'];assert len(steps)==1
step=steps[0];assert all(step['config']['norm_map'][k]=='QUANTILES' for k in ('STATE','ACTION'))
p=(root/step['state_file']).resolve(strict=True);assert p.is_relative_to(root)
data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==hashes[str(p.relative_to(root))]
n=struct.unpack('<Q',data[:8])[0];header=json.loads(data[8:8+n])
def values(key):
 v=header[key];assert v['dtype']=='F32' and v['shape']==[14]
 lo,hi=v['data_offsets'];return list(struct.unpack('<14f',data[8+n+lo:8+n+hi]))
norm={'checkpoint_id':cid,'processor_id':pid}
for label,key in [('action','action'),('state','observation.state')]:
 norm[label]={'q01':values(key+'.q01'),'q99':values(key+'.q99'),'mode':'QUANTILES','eps':step['config']['eps']}
assert signatures=={str(p.relative_to(root)):sig(p) for p in files}
assert health()==a and identity()==service
print(json.dumps({'schema_version':1,'synthetic':False,'verified':True,'source_kind':'verified_deployed_pi05_checkpoint','checkpoint':a['checkpoint'],'checkpoint_id':cid,'processor_id':pid,'file_manifest':hashes,'normalization':norm,'normalization_sha256':hashlib.sha256(json.dumps(norm,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'contract':a['contract'],'backend_service_identity':dict(line.split('=',1) for line in service.splitlines()),'file_signatures':signatures}))
'''


def audit_manifest(checkpoint, directory, ssh_host):
    if not re.fullmatch(r'[A-Za-z0-9_@.][A-Za-z0-9_@.-]*', ssh_host):
        raise ValueError('Invalid configured SSH host')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             ssh_host, 'nice -n 19 ionice -c 3 python3 -'],
                            input=REMOTE_AUDIT, text=True, capture_output=True, timeout=180, check=True)
    manifest = json.loads(result.stdout)
    if manifest['checkpoint'] != checkpoint:
        raise ValueError('Backend changed during automatic verification; retry connection')
    from .remote_server import read_manifest
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (manifest['checkpoint_id'] + '-' + uuid.uuid4().hex + '.json')
    with path.open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    read_manifest(path)
    return path
