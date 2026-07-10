// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

public struct WeightLoadingDiagnostics: Sendable, Codable, Equatable {
    public let loadedTensorNames: [String]
    public let evaluatedTensorNames: [String]
    public let excludedTensorNames: [String]
    public let loadedBytesByComponent: [String: UInt64]
    public let evaluatedBytesByComponent: [String: UInt64]

    public var evaluatedAudioBytes: UInt64 {
        evaluatedBytesByComponent["audio"] ?? 0
    }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    _ = try loadWeightsImpl(
        modelDirectory: modelDirectory, model: model, quantization: quantization,
        perLayerQuantization: perLayerQuantization, collectDiagnostics: false)
}

/// Experiment-oriented loader entry point that preserves normal strict verification while
/// returning tensor/component accounting for a single model materialization.
public func loadWeightsForDiagnostics(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws -> WeightLoadingDiagnostics {
    try loadWeightsImpl(
        modelDirectory: modelDirectory, model: model, quantization: quantization,
        perLayerQuantization: perLayerQuantization, collectDiagnostics: true)!
}

private func loadWeightsImpl(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization?,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?,
    collectDiagnostics: Bool
) throws -> WeightLoadingDiagnostics? {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
    }

    let loadedNames = collectDiagnostics ? weights.keys.sorted() : []
    let loadedBytes = collectDiagnostics ? bytesByComponent(weights) : [:]

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)

    guard collectDiagnostics else { return nil }
    let evaluatedNames = weights.keys.sorted()
    return WeightLoadingDiagnostics(
        loadedTensorNames: loadedNames,
        evaluatedTensorNames: evaluatedNames,
        excludedTensorNames: Array(Set(loadedNames).subtracting(evaluatedNames)).sorted(),
        loadedBytesByComponent: loadedBytes,
        evaluatedBytesByComponent: bytesByComponent(weights))
}

private func bytesByComponent(_ weights: [String: MLXArray]) -> [String: UInt64] {
    weights.reduce(into: [
        "audio": 0,
        "per_layer_embedding": 0,
        "vision": 0,
        "language_model": 0,
        "other": 0,
    ]) { result, entry in
        result[weightComponent(entry.key), default: 0] += UInt64(entry.value.nbytes)
    }
}

private func weightComponent(_ name: String) -> String {
    if name.contains("audio") { return "audio" }
    if name.contains("embed_tokens_per_layer") { return "per_layer_embedding" }
    if name.contains("vision") { return "vision" }
    if name.contains("language_model") { return "language_model" }
    return "other"
}
