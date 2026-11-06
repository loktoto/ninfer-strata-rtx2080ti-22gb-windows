import sys, tempfile, unittest, struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from gguf_expert_index import index_model

def u32(n):return struct.pack('<I',n)
def u64(n):return struct.pack('<Q',n)
def s(x):return u64(len(x.encode()))+x.encode()
class IndexTests(unittest.TestCase):
  def test_low_bit_expert_ranges(self):
    with tempfile.TemporaryDirectory() as tmp:
      p=Path(tmp)/'tiny.gguf'
      meta=[('general.architecture',8,s('qwen35moe')),
            ('qwen35moe.expert_count',4,u32(4)),
            ('qwen35moe.expert_used_count',4,u32(8))]
      names=[f'blk.0.ffn_{x}_exps.weight' for x in ('gate','down','up')]
      stride=256*512//256*84
      with p.open('wb') as f:
        f.write(b'GGUF'+u32(3)+u64(3)+u64(len(meta)))
        for key,t,v in meta:f.write(s(key)+u32(t)+v)
        for n,name in enumerate(names):
          f.write(s(name)+u32(3)+u64(256)+u64(512)+u64(4)+u32(10)+u64(n*stride*4))
        f.write(b'\x00'*((-f.tell())%32))
        f.write(b'\x7f'*stride*4*3)
      index=index_model(p,layers=1,experts=4)
      self.assertEqual(len(index['tensors']),3)
      self.assertEqual({t['stride_bytes'] for t in index['tensors']},{stride})
if __name__=='__main__': unittest.main()
