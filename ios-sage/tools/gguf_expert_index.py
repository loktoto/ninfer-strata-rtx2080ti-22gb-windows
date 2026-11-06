#!/usr/bin/env python3
"""Create GGUF expert slice offset manifest, without reading weights."""
import argparse, json, os, re, struct
from pathlib import Path

BLOCK = {0:(1,4),1:(1,2),2:(32,18),8:(32,34),10:(256,84),
         11:(256,110),12:(256,144),13:(256,176),14:(256,210),
         16:(256,66),17:(256,74),18:(256,98),19:(256,50),
         22:(256,82),23:(256,136),29:(256,56),30:(1,2)}
FIXED = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
PAT = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight')

def exact(f, n):
    v=f.read(n)
    if len(v)!=n: raise ValueError('truncated GGUF header')
    return v
def u32(f):return struct.unpack('<I',exact(f,4))[0]
def u64(f):return struct.unpack('<Q',exact(f,8))[0]
def skip(f,n,lim):
    if n>lim or f.tell()+n>lim: raise ValueError('GGUF metadata beyond file')
    f.seek(n,os.SEEK_CUR)
def string(f,lim,read=False):
    n=u64(f)
    if n>32_000_000 and read:raise ValueError('oversized string')
    if read:return exact(f,n).decode('utf8')
    skip(f,n,lim)
def value(f,typ,lim,keep=False):
    if typ==8:return string(f,lim,keep)
    if typ in FIXED:
        n=FIXED[typ]
        return int.from_bytes(exact(f,n),'little') if keep and typ in (4,10) else skip(f,n,lim)
    if typ==9:
        t=u32(f);n=u64(f)
        if n>10_000_000:raise ValueError('oversized array')
        if t in FIXED:skip(f,n*FIXED[t],lim)
        else:
            for _ in range(n):value(f,t,lim)
        return
    raise ValueError(f'unknown metadata type {typ}')

def index_model(p,layers=40,experts=256):
    lim=p.stat().st_size
    with p.open('rb') as f:
        if exact(f,4)!=b'GGUF' or u32(f)!=3:raise ValueError('not GGUF v3')
        nt,nkv=u64(f),u64(f)
        if nt>100000 or nkv>100000:raise ValueError('implausible header')
        meta={}
        targets={'general.architecture','general.alignment','qwen35moe.expert_count','qwen35moe.expert_used_count'}
        for _ in range(nkv):
            key=string(f,lim,True);typ=u32(f)
            v=value(f,typ,lim,key in targets)
            if key in targets:meta[key]=v
        if meta.get('general.architecture')!='qwen35moe':raise ValueError('not Sage qwen35moe')
        if meta.get('qwen35moe.expert_count')!=experts or meta.get('qwen35moe.expert_used_count')!=8:
            raise ValueError('expert configuration mismatch')
        alignment=meta.get('general.alignment') or 32
        if alignment<1 or alignment>4096 or alignment&(alignment-1):raise ValueError('bad alignment')
        tensors=[]
        for _ in range(nt):
            name=string(f,lim,True);nd=u32(f)
            if not 1<=nd<=8:raise ValueError('invalid rank')
            dims=[u64(f) for _ in range(nd)]
            kind=u32(f);rel=u64(f)
            m=PAT.fullmatch(name)
            if not m or int(m[1])>=layers:continue
            if len(dims)!=3 or dims[-1]!=experts or kind not in BLOCK:raise ValueError('expert shape/type unknown')
            ne=dims[0]*dims[1];bl,bs=BLOCK[kind]
            if not ne or ne%bl:raise ValueError('GGML block straddles expert')
            tensors.append(dict(layer=int(m[1]),part=m[2],name=name,
                                ggml_type=kind,stride_bytes=ne//bl*bs,
                                relative_offset=rel,experts=experts))
        base=(f.tell()+alignment-1)//alignment*alignment
    groups={}
    for t in tensors:
        groups.setdefault(t['layer'],set()).add(t['part'])
        t['offset_bytes']=base+t.pop('relative_offset')
        if t['offset_bytes']+t['stride_bytes']*experts>lim:raise ValueError('tensor beyond GGUF EOF')
    if any(groups.get(i)!={'gate','down','up'} for i in range(layers)) or len(tensors)!=layers*3:
        raise ValueError('incomplete or duplicate expert tensor groups')
    return dict(format='sage-expert-index-v1',source_file=p.name,
                file_size_bytes=lim,architecture='qwen35moe',model_layers=layers,
                expert_count=experts,expert_top_k=8,tensors=tensors)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('gguf',type=Path)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--layers',type=int,default=40)
    a.add_argument('--experts',type=int,default=256)
    x=a.parse_args()
    if x.layers<1 or x.experts<1:a.error('positive layers/experts required')
    with x.out.open('x') as f:json.dump(index_model(x.gguf,x.layers,x.experts),f,indent=2)
    print('Index created; token inference is not implemented.')
if __name__=='__main__':main()
