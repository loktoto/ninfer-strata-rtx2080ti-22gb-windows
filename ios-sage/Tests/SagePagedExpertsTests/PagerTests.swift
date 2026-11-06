import Foundation
import XCTest
@testable import SagePagedExperts
final class PagerTests:XCTestCase{
  func testExactExpertReadsAndRAMCache()throws{
    let base=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:base,withIntermediateDirectories:true)
    defer{try? FileManager.default.removeItem(at:base)}
    let model=base.appendingPathComponent("test.gguf"),idx=base.appendingPathComponent("index.json")
    var data=Data()
    for p in 0..<3{for e in 0..<4{data.append(Data(repeating:UInt8(10*p+e),count:12))}}
    try data.write(to:model)
    let tensors=["gate","down","up"].enumerated().map{n,p in
      ["layer":0,"part":p,"stride_bytes":12,"experts":4,"offset_bytes":n*48] as [String:Any]
    }
    let manifest:[String:Any]=["format":"sage-expert-index-v1","file_size_bytes":144,
      "architecture":"qwen35moe","expert_count":4,"expert_top_k":8,"model_layers":1,"tensors":tensors]
    try JSONSerialization.data(withJSONObject:manifest).write(to:idx)
    let pager=try SagePagedExperts(gguf:model,index:idx,cacheBytes:72)
    let b=try pager.load(layer:0,expert:2)
    XCTAssertEqual(b.part("gate"),Data(repeating:2,count:12))
    XCTAssertEqual(b.part("down"),Data(repeating:12,count:12))
    XCTAssertEqual(b.part("up"),Data(repeating:22,count:12))
    _=try pager.load(layer:0,expert:2)
    XCTAssertEqual(pager.stats().hits,1)
    XCTAssertEqual(pager.stats().readBytes,36)
    pager.setCacheLimit(0)
    XCTAssertEqual(pager.stats().cacheBytes,0)
  }
}
