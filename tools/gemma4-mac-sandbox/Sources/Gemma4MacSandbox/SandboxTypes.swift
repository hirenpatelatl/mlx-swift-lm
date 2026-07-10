import Foundation
import MLXLMCommon
import MLXVLM

let defaultModelDirectory = URL(fileURLWithPath:
    "/Users/hirenpatel/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/2c3e507453b4f218d05fe3cc97bea5c5a654257e")
let expectedModelRevision = "2c3e507453b4f218d05fe3cc97bea5c5a654257e"
let expectedTokenizerSHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
let expectedMetalLibrarySHA256 = "71f2ad788d86f29486315b55835ce6ed17ceb9b0302f6b80429306340dca28d1"
let colorPrompt =
    "Answer with exactly one lowercase word: red or blue. What color is this image?"
let fixedRows = [0, 1, 42, 258_881, 262_143]

struct Options {
    private var values: [String: String] = [:]

    init(_ args: [String]) {
        var index = 0
        while index < args.count {
            let key = args[index]
            if key.hasPrefix("--"), index + 1 < args.count {
                values[String(key.dropFirst(2))] = args[index + 1]
                index += 2
            } else {
                index += 1
            }
        }
    }

    var modelDirectory: URL { valueURL("model-dir") ?? defaultModelDirectory }
    var outputURL: URL? { valueURL("output") }
    var metalLibraryURL: URL? { valueURL("metallib") }
    var cacheRows: Int { Int(values["cache-rows"] ?? "512") ?? 512 }

    func required(_ key: String) throws -> String {
        guard let value = values[key] else {
            throw SandboxError.invalidOption("missing --\(key)")
        }
        return value
    }

    func requiredURL(_ key: String) throws -> URL {
        guard let value = valueURL(key) else {
            throw SandboxError.invalidOption("missing --\(key)")
        }
        return value
    }

    private func valueURL(_ key: String) -> URL? {
        values[key].map { URL(fileURLWithPath: $0) }
    }
}

struct SelfTestReceipt: Codable {
    let mode: String
    let modelRevision: String
    let tokenizerSHA256: String
    let metalLibrarySHA256: String
    let rows: [Gemma4PagedRowComparison]
    let rowStoreMetrics: SafeTensorRowStoreMetrics
    let rawRowsMatch: Bool
    let dequantizedRowsMatch: Bool
    let exitReason: String
}

struct RunReceipt: Codable {
    let mode: String
    let color: String
    let sourceRevision: String
    let modelRevision: String
    let tokenizerSHA256: String
    let metalLibrarySHA256: String
    let startPhysicalFootprint: UInt64
    let endPhysicalFootprint: UInt64
    let peakPhysicalFootprint: UInt64
    let startResidentSetSize: UInt64
    let endResidentSetSize: UInt64
    let peakResidentSetSize: UInt64
    let elapsedLoadSeconds: TimeInterval
    let elapsedPrefillSeconds: TimeInterval
    let elapsedGenerationSeconds: TimeInterval
    let elapsedTotalSeconds: TimeInterval
    let generatedTokenIDs: [Int]
    let outputText: String
    let exclusiveColorCorrect: Bool
    let thermalState: String
    let loaderDiagnostics: WeightLoadingDiagnostics
    let pagedMetrics: Gemma4PagedPerLayerEmbeddingMetrics?
    let exitReason: String
}

struct ComparisonReceipt: Codable {
    let result: String
    let receiptCount: Int
    let residentMedianPeakPhysicalFootprint: UInt64
    let pagedMedianPeakPhysicalFootprint: UInt64
    let medianReductionBytes: UInt64
    let minimumRequiredReductionBytes: UInt64
    let deterministicTokenIDs: Bool
    let exclusiveColorsCorrect: Bool
    let fixedRunCounts: Bool
    let audioEvaluatedBytes: UInt64
    let pagerReadsBounded: Bool
    let cacheBounded: Bool
}

enum SandboxError: LocalizedError {
    case usage
    case invalidOption(String)
    case invalidSnapshot(String)
    case comparisonFailed(String)

    var errorDescription: String? {
        switch self {
        case .usage:
            return "usage: Gemma4MacSandbox self-test|run|compare"
        case .invalidOption(let message), .invalidSnapshot(let message),
            .comparisonFailed(let message):
            return message
        }
    }
}

func printJSON<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    FileHandle.standardOutput.write(try encoder.encode(value))
    FileHandle.standardOutput.write(Data("\n".utf8))
}

func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(value).write(to: url, options: .atomic)
}
