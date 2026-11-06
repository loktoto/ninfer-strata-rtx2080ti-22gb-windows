import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

public struct SageManifest: Decodable {
  public struct Tensor: Decodable {
    public let layer: Int
    public let part: String
    public let strideBytes: Int
    public let experts: Int
    public let offsetBytes: UInt64
  }
  public let format: String
  public let fileSizeBytes: UInt64
  public let architecture: String
  public let expertCount: Int
  public let expertTopK: Int
  public let modelLayers: Int
  public let tensors: [Tensor]
}
public struct SageBlob {
  public let bytes: Data
  public let ranges: [String: Range<Int>]
  public func part(_ name: String) -> Data? {
    guard let r = ranges[name] else { return nil }
    return bytes.subdata(in: r)
  }
}
public struct SageStats {
  public let hits: Int
  public let misses: Int
  public let readBytes: Int
  public let cacheBytes: Int
  public let cachedExperts: Int
}
public enum SagePagerError: Error {
  case invalidManifest, wrongFile, cannotOpen, badExpert, unexpectedEOF, readFailure(Int32)
}
public final class SagePagedExperts {
  private struct Key: Hashable { let layer:Int; let expert:Int }
  private struct Entry { let blob:SageBlob; var count:Int; var recent:UInt64 }
  private let fd:Int32
  private let groups:[Int:[SageManifest.Tensor]]
  private let experts:Int
  private let lock=NSLock()
  private var entries:[Key:Entry]=[:]
  private var limit:Int
  private var used=0
  private var clock:UInt64=0
  private var hits=0
  private var misses=0
  private var readBytes=0

  public init(gguf:URL, index:URL, cacheBytes:Int) throws {
    let decoder=JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    let m=try decoder.decode(SageManifest.self,from:Data(contentsOf:index))
    guard cacheBytes>=0,m.format=="sage-expert-index-v1",m.architecture=="qwen35moe",
          m.expertTopK==8,m.expertCount>0,m.modelLayers>0,m.tensors.count==m.modelLayers*3
    else { throw SagePagerError.invalidManifest }
    var map:[Int:[SageManifest.Tensor]]=[:]
    for t in m.tensors {
      let (size,of1)=UInt64(max(0,t.strideBytes)).multipliedReportingOverflow(by:UInt64(max(0,t.experts)))
      let (end,of2)=t.offsetBytes.addingReportingOverflow(size)
      guard !of1,!of2,t.layer>=0,t.layer<m.modelLayers,t.experts==m.expertCount,
            t.strideBytes>0,t.strideBytes<64*1024*1024,
            ["gate","down","up"].contains(t.part),end<=m.fileSizeBytes
      else {throw SagePagerError.invalidManifest}
      map[t.layer,default:[]].append(t)
    }
    for l in 0..<m.modelLayers {
      guard let a=map[l],a.count==3,Set(a.map(\.part))==Set(["gate","down","up"])
      else {throw SagePagerError.invalidManifest}
      map[l]=a.sorted{$0.part<$1.part}
    }
    let f=open(gguf.path,O_RDONLY)
    guard f>=0 else {throw SagePagerError.cannotOpen}
    var st=stat()
    guard fstat(f,&st)==0,st.st_size>=0,UInt64(st.st_size)==m.fileSizeBytes else {
      close(f);throw SagePagerError.wrongFile
    }
    fd=f;groups=map;experts=m.expertCount;limit=cacheBytes
  }
  deinit{close(fd)}
  public func load(layer:Int,expert:Int)throws->SageBlob{
    guard expert>=0,expert<experts,let parts=groups[layer] else{throw SagePagerError.badExpert}
    let key=Key(layer:layer,expert:expert)
    lock.lock()
    clock &+= 1
    if var e=entries[key]{
      hits+=1;e.count+=1;e.recent=clock;entries[key]=e
      lock.unlock();return e.blob
    }
    misses+=1;lock.unlock()
    let size=parts.reduce(0){$0+$1.strideBytes}
    var bytes=Data(count:size)
    var ranges:[String:Range<Int>]=[:]
    try bytes.withUnsafeMutableBytes{ b in
      guard let ptr=b.baseAddress else{throw SagePagerError.unexpectedEOF}
      var pos=0
      for t in parts {
        let off=t.offsetBytes+UInt64(expert)*UInt64(t.strideBytes)
        guard off<=UInt64(Int64.max) else{throw SagePagerError.invalidManifest}
        var got=0
        while got<t.strideBytes {
          let n=pread(fd,ptr.advanced(by:pos+got),t.strideBytes-got,off_t(off+UInt64(got)))
          if n<0 {
            if errno==EINTR {continue}
            throw SagePagerError.readFailure(errno)
          }
          if n==0 {throw SagePagerError.unexpectedEOF}
          got+=n
        }
        ranges[t.part]=pos..<(pos+t.strideBytes);pos+=t.strideBytes
      }
    }
    let blob=SageBlob(bytes:bytes,ranges:ranges)
    lock.lock();defer{lock.unlock()}
    readBytes+=size
    if let existing=entries[key]{return existing.blob}
    if size<=limit {
      while used>limit-size,let victim=entries.min(by:{
        ($0.value.count,$0.value.recent).0 == ($1.value.count,$1.value.recent).0
          ? $0.value.recent<$1.value.recent : $0.value.count<$1.value.count
      }) {
        used-=victim.value.blob.bytes.count;entries.removeValue(forKey:victim.key)
      }
      entries[key]=Entry(blob:blob,count:1,recent:clock);used+=size
    }
    return blob
  }
  public func setCacheLimit(_ bytes:Int){
    lock.lock();defer{lock.unlock()};limit=max(0,bytes)
    while used>limit,let victim=entries.min(by:{$0.value.count<$1.value.count}){
      used-=victim.value.blob.bytes.count;entries.removeValue(forKey:victim.key)
    }
  }
  public func stats()->SageStats{
    lock.lock();defer{lock.unlock()}
    return SageStats(hits:hits,misses:misses,readBytes:readBytes,cacheBytes:used,cachedExperts:entries.count)
  }
}
