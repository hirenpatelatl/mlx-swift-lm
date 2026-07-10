// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "gemma4-mac-sandbox",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "Gemma4MacSandbox",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
            ]
        )
    ]
)
