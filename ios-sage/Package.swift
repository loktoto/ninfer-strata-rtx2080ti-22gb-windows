// swift-tools-version: 5.9
import PackageDescription
let package = Package(
  name: "SagePagedExperts",
  platforms: [.iOS(.v17), .macOS(.v14)],
  products: [.library(name: "SagePagedExperts", targets: ["SagePagedExperts"])],
  targets: [
    .target(name: "SagePagedExperts"),
    .testTarget(name: "SagePagedExpertsTests", dependencies: ["SagePagedExperts"])
  ]
)
