// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

enum Gemma4PagedPerLayerEmbeddingTensorName {
    static let prefix = "language_model.model.embed_tokens_per_layer"
    static let weight = "\(prefix).weight"
    static let scales = "\(prefix).scales"
    static let biases = "\(prefix).biases"

    static func isExternalized(_ name: String) -> Bool {
        name == weight || name == scales || name == biases
    }
}

public struct Gemma4PagedPerLayerEmbeddingMetrics: Sendable, Codable, Equatable {
    public var rowStore: SafeTensorRowStoreMetrics
    public var cachedRows: Int
    public var dequantizedPayloadBytes: Int
}

public struct Gemma4PagedRowComparison: Sendable, Codable, Equatable {
    public let row: Int
    public let rawBytesMatch: Bool
    public let dequantizedValuesMatch: Bool
    public let maximumAbsoluteDifference: Float
}

protocol Gemma4PerLayerInputProviding: AnyObject {
    func values(for tokenIDs: MLXArray) -> MLXArray
    var metrics: Gemma4PagedPerLayerEmbeddingMetrics { get }
}

final class Gemma4PagedPerLayerEmbedding: Gemma4PerLayerInputProviding, @unchecked Sendable {
    private let rowStore: SafeTensorRowStore
    private let hiddenLayers: Int
    private let hiddenSizePerLayerInput: Int
    private let cacheLimit: Int
    private let lock = NSLock()
    private var cache: [Int: MLXArray] = [:]
    private var recency: [Int] = []

    init(
        modelDirectory: URL,
        hiddenLayers: Int,
        hiddenSizePerLayerInput: Int,
        cacheLimit: Int = 512
    ) throws {
        let safetensor = modelDirectory.appendingPathComponent("model.safetensors")
        self.rowStore = try SafeTensorRowStore(
            url: safetensor, allowedRoot: safetensorAllowedRoot(for: modelDirectory))
        self.hiddenLayers = hiddenLayers
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.cacheLimit = max(cacheLimit, 1)
        try Self.validate(store: rowStore, layers: hiddenLayers, perLayer: hiddenSizePerLayerInput)
    }

    var metrics: Gemma4PagedPerLayerEmbeddingMetrics {
        lock.withLock {
            Gemma4PagedPerLayerEmbeddingMetrics(
                rowStore: rowStore.metrics,
                cachedRows: cache.count,
                dequantizedPayloadBytes: cache.values.reduce(0) { $0 + $1.nbytes }
            )
        }
    }

    func values(for tokenIDs: MLXArray) -> MLXArray {
        let originalShape = Array(tokenIDs.shape)
        let tokens = tokenIDs.flattened().asArray(Int.self)
        var uniqueRows: [Int: MLXArray] = [:]
        for token in tokens where uniqueRows[token] == nil {
            do {
                uniqueRows[token] = try row(for: token)
            } catch {
                fatalError("Gemma4 paged per-layer embedding failed for token \(token): \(error)")
            }
        }
        let rows = tokens.map { uniqueRows[$0]! }
        return stacked(rows, axis: 0).reshaped(
            originalShape + [hiddenLayers, hiddenSizePerLayerInput]
        )
    }

    private func row(for token: Int) throws -> MLXArray {
        try lock.withLock {
            if let cached = cache[token] {
                rowStore.recordCacheHit()
                markRecentlyUsed(token)
                return cached
            }
            rowStore.recordCacheMiss()
            let loaded = try loadRow(token: token)
            eval(loaded)
            guard loaded.dtype == .bfloat16 else {
                throw SafeTensorRowStoreError.malformedTensor(
                    "paged row dequantized to \(loaded.dtype), expected bfloat16")
            }
            cache[token] = loaded
            markRecentlyUsed(token)
            evictIfNeeded()
            return loaded
        }
    }

    private func markRecentlyUsed(_ token: Int) {
        recency.removeAll { $0 == token }
        recency.append(token)
    }

    private func evictIfNeeded() {
        while cache.count > cacheLimit, let evicted = recency.first {
            recency.removeFirst()
            cache.removeValue(forKey: evicted)
            rowStore.recordEviction()
        }
    }

    private func loadRow(token: Int) throws -> MLXArray {
        let weightBytes = try rowStore.readRow(
            tensor: Gemma4PagedPerLayerEmbeddingTensorName.weight, row: token)
        let scalesBytes = try rowStore.readRow(
            tensor: Gemma4PagedPerLayerEmbeddingTensorName.scales, row: token)
        let biasesBytes = try rowStore.readRow(
            tensor: Gemma4PagedPerLayerEmbeddingTensorName.biases, row: token)

        let weight = MLXArray(weightBytes, [1, weightBytes.count / 4], dtype: .uint32)
        let groups = scalesBytes.count / 2
        let scales = MLXArray(scalesBytes, [1, groups], dtype: .bfloat16)
        let biases = MLXArray(biasesBytes, [1, groups], dtype: .bfloat16)
        return dequantized(weight, scales: scales, biases: biases, groupSize: 64, bits: 4)
            .reshaped(hiddenLayers, hiddenSizePerLayerInput)
    }

    fileprivate static func validate(
        store: SafeTensorRowStore, layers: Int, perLayer: Int
    ) throws {
        let weight = try store.tensor(named: Gemma4PagedPerLayerEmbeddingTensorName.weight)
        let scales = try store.tensor(named: Gemma4PagedPerLayerEmbeddingTensorName.scales)
        let biases = try store.tensor(named: Gemma4PagedPerLayerEmbeddingTensorName.biases)
        guard weight.dtype == "U32", scales.dtype == "BF16", biases.dtype == "BF16" else {
            throw SafeTensorRowStoreError.malformedTensor("unexpected per-layer tensor dtype")
        }
        guard weight.shape.count == 2, scales.shape.count == 2, biases.shape.count == 2,
            weight.shape[0] == scales.shape[0], weight.shape[0] == biases.shape[0],
            scales.shape == biases.shape
        else {
            throw SafeTensorRowStoreError.malformedTensor("unexpected per-layer tensor shape")
        }
        guard weight.rowByteCount == 4_480 else {
            throw SafeTensorRowStoreError.malformedTensor("unexpected quantized row width")
        }
        guard scales.rowByteCount + biases.rowByteCount + weight.rowByteCount == 5_040 else {
            throw SafeTensorRowStoreError.malformedTensor("unexpected total per-token row width")
        }
        guard layers * perLayer == scales.shape[1] * 64 else {
            throw SafeTensorRowStoreError.malformedTensor("unexpected dequantized row width")
        }
    }
}

public func compareGemma4PagedRows(
    modelDirectory: URL, rows: [Int]
) throws -> [Gemma4PagedRowComparison] {
    let safetensor = modelDirectory.appendingPathComponent("model.safetensors")
    let store = try SafeTensorRowStore(
        url: safetensor, allowedRoot: safetensorAllowedRoot(for: modelDirectory))
    try Gemma4PagedPerLayerEmbedding.validate(
        store: store, layers: 35, perLayer: 256)
    let resident = try loadArrays(url: safetensor)
    let names = [
        Gemma4PagedPerLayerEmbeddingTensorName.weight,
        Gemma4PagedPerLayerEmbeddingTensorName.scales,
        Gemma4PagedPerLayerEmbeddingTensorName.biases,
    ]
    return try rows.map { row in
        var rawMatches = true
        var pagedRows: [MLXArray] = []
        var residentRows: [MLXArray] = []
        for name in names {
            guard let residentTensor = resident[name] else {
                throw SafeTensorRowStoreError.tensorNotFound(name)
            }
            let raw = try store.readRow(tensor: name, row: row)
            let residentRow = residentTensor[row].reshaped(1, -1)
            rawMatches = rawMatches && raw == residentRow.asData().data
            let dtype = try store.tensor(named: name).dtype
            let rowArray = MLXArray(
                raw, [1, raw.count / (dtype == "U32" ? 4 : 2)],
                dtype: dtype == "U32" ? .uint32 : .bfloat16)
            pagedRows.append(rowArray)
            residentRows.append(residentRow)
        }
        let paged = dequantized(
            pagedRows[0], scales: pagedRows[1], biases: pagedRows[2],
            groupSize: 64, bits: 4)
        let reference = dequantized(
            residentRows[0], scales: residentRows[1], biases: residentRows[2],
            groupSize: 64, bits: 4)
        eval(paged, reference)
        let difference = abs(paged.asType(.float32) - reference.asType(.float32)).max()
            .item(Float.self)
        return Gemma4PagedRowComparison(
            row: row,
            rawBytesMatch: rawMatches,
            dequantizedValuesMatch: paged.allClose(reference, rtol: 0, atol: 0).item(Bool.self),
            maximumAbsoluteDifference: difference)
    }
}

extension Gemma4 {
    public func installPagedPerLayerEmbedding(
        modelDirectory: URL, cacheRows: Int = 512
    ) throws {
        let textConfig = config.textConfiguration
        let provider = try Gemma4PagedPerLayerEmbedding(
            modelDirectory: modelDirectory,
            hiddenLayers: textConfig.hiddenLayers,
            hiddenSizePerLayerInput: textConfig.hiddenSizePerLayerInput,
            cacheLimit: cacheRows
        )
        try languageModel.model.installPagedPerLayerEmbedding(provider)
    }

    public var pagedPerLayerEmbeddingMetrics: Gemma4PagedPerLayerEmbeddingMetrics? {
        languageModel.model.pagedPerLayerEmbedding?.metrics
    }
}

private func safetensorAllowedRoot(for modelDirectory: URL) -> URL {
    let snapshots = modelDirectory.deletingLastPathComponent()
    if snapshots.lastPathComponent == "snapshots" {
        return snapshots.deletingLastPathComponent()
    }
    return modelDirectory
}
