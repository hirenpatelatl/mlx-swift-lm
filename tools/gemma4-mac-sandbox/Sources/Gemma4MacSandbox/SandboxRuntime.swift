import CoreImage
import Foundation
import MLXLMCommon
import MLXVLM

func runSelfTest(modelDirectory: URL, metalLibrarySHA256: String) throws {
    try validateModelDirectory(modelDirectory)
    let tokenizer = try PinnedGemma4Tokenizer(modelDirectory: modelDirectory)
    let comparisons = try compareGemma4PagedRows(
        modelDirectory: modelDirectory, rows: fixedRows)
    let store = try SafeTensorRowStore(
        url: modelDirectory.appendingPathComponent("model.safetensors"),
        allowedRoot: allowedModelCacheRoot(modelDirectory))
    for row in fixedRows {
        for name in [
            "language_model.model.embed_tokens_per_layer.weight",
            "language_model.model.embed_tokens_per_layer.scales",
            "language_model.model.embed_tokens_per_layer.biases",
        ] {
            _ = try store.readRow(tensor: name, row: row)
        }
    }
    let rawMatch = comparisons.allSatisfy(\.rawBytesMatch)
    let dequantizedMatch = comparisons.allSatisfy(\.dequantizedValuesMatch)
    let receipt = SelfTestReceipt(
        mode: "self-test",
        modelRevision: expectedModelRevision,
        tokenizerSHA256: tokenizer.sha256,
        metalLibrarySHA256: metalLibrarySHA256,
        rows: comparisons,
        rowStoreMetrics: store.metrics,
        rawRowsMatch: rawMatch,
        dequantizedRowsMatch: dequantizedMatch,
        exitReason: rawMatch && dequantizedMatch ? "fixed-row-equivalence-passed" : "row-mismatch")
    try printJSON(receipt)
    guard rawMatch, dequantizedMatch else {
        throw SandboxError.comparisonFailed("fixed-row equivalence failed")
    }
    guard store.metrics.rowDataByteCount == UInt64(fixedRows.count * 5_040),
        store.metrics.largestRowDataRead == 4_480
    else {
        throw SandboxError.comparisonFailed("fixed-row I/O accounting exceeded bounds")
    }
}

func runOne(
    mode: String, color: String, modelDirectory: URL, output: URL?, cacheRows: Int,
    metalLibrarySHA256: String
) async throws {
    guard mode == "resident" || mode == "paged" else {
        throw SandboxError.invalidOption("mode must be resident or paged")
    }
    guard color == "red" || color == "blue" else {
        throw SandboxError.invalidOption("color must be red or blue")
    }
    guard cacheRows > 0, cacheRows <= 512 else {
        throw SandboxError.invalidOption("cache-rows must be between 1 and 512")
    }
    try validateModelDirectory(modelDirectory)

    let start = MemorySample.current()
    let sampler = PeakMemorySampler()
    let started = Date()
    let configData = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
    let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
    let config = try JSONDecoder.json5().decode(Gemma4Configuration.self, from: configData)
    let model = Gemma4(config)
    let weightLoadingStrategy: ModelWeightLoadingStrategy =
        mode == "paged"
        ? .gemma4PagedPerLayerEmbedding(cacheRows: cacheRows)
        : .resident
    if mode == "paged" {
        try model.installPagedPerLayerEmbedding(
            modelDirectory: modelDirectory, cacheRows: cacheRows)
    }

    let loadStart = Date()
    let diagnostics = try loadWeightsForDiagnostics(
        modelDirectory: modelDirectory,
        model: model,
        perLayerQuantization: baseConfig.perLayerQuantization,
        weightLoadingStrategy: weightLoadingStrategy)
    let loaded = Date()

    let tokenizer = try PinnedGemma4Tokenizer(modelDirectory: modelDirectory)
    let processorData = try Data(
        contentsOf: modelDirectory.appendingPathComponent("processor_config.json"))
    let processorConfig = try JSONDecoder.json5().decode(
        Gemma4ProcessorConfiguration.self, from: processorData)
    let processor = Gemma4Processor(processorConfig, tokenizer: tokenizer)
    let context = ModelContext(
        configuration: ModelConfiguration(
            directory: modelDirectory,
            defaultPrompt: colorPrompt,
            extraEOSTokens: ["<turn|>"],
            eosTokenIds: [1, 50, 106],
            weightLoadingStrategy: weightLoadingStrategy),
        model: model,
        processor: processor,
        tokenizer: tokenizer)

    let image = solidColorImage(color)
    let input = try await processor.prepare(
        input: UserInput(prompt: colorPrompt, images: [.ciImage(image)]))
    let stream = try generateTokens(
        input: input,
        parameters: GenerateParameters(maxTokens: 8, temperature: 0),
        context: context)
    var generatedTokenIDs: [Int] = []
    var completion: GenerateCompletionInfo?
    for await event in stream {
        switch event {
        case .token(let token): generatedTokenIDs.append(token)
        case .info(let info): completion = info
        }
    }
    guard let completion else {
        throw SandboxError.comparisonFailed("generation ended without completion telemetry")
    }

    let outputText = tokenizer.decode(tokenIds: generatedTokenIDs)
    let correct = isExclusiveColor(outputText, expected: color)
    let end = MemorySample.current()
    let peak = sampler.stop()
    let metrics = model.pagedPerLayerEmbeddingMetrics
    let receipt = RunReceipt(
        mode: mode,
        color: color,
        sourceRevision: currentGitRevision(),
        modelRevision: expectedModelRevision,
        tokenizerSHA256: tokenizer.sha256,
        metalLibrarySHA256: metalLibrarySHA256,
        startPhysicalFootprint: start.physicalFootprint,
        endPhysicalFootprint: end.physicalFootprint,
        peakPhysicalFootprint: peak.physicalFootprint,
        startResidentSetSize: start.residentSetSize,
        endResidentSetSize: end.residentSetSize,
        peakResidentSetSize: peak.residentSetSize,
        elapsedLoadSeconds: loaded.timeIntervalSince(loadStart),
        elapsedPrefillSeconds: completion.promptTime,
        elapsedGenerationSeconds: completion.generateTime,
        elapsedTotalSeconds: Date().timeIntervalSince(started),
        generatedTokenIDs: generatedTokenIDs,
        outputText: outputText,
        exclusiveColorCorrect: correct,
        thermalState: thermalStateDescription(),
        loaderDiagnostics: diagnostics,
        pagedMetrics: metrics,
        exitReason: correct ? "deterministic-color-generation-passed" : "incorrect-color-output")
    if let output { try writeJSON(receipt, to: output) } else { try printJSON(receipt) }

    guard correct else {
        throw SandboxError.comparisonFailed(
            "expected exactly \(color), generated \(String(reflecting: outputText))")
    }
    guard diagnostics.evaluatedAudioBytes == 0 else {
        throw SandboxError.comparisonFailed("audio tensors were evaluated")
    }
    if mode == "paged" {
        guard diagnostics.loadedBytesByComponent["per_layer_embedding"] == 0,
            diagnostics.excludedTensorNames.contains(
                "language_model.model.embed_tokens_per_layer.weight"),
            diagnostics.excludedTensorNames.contains(
                "language_model.model.embed_tokens_per_layer.scales"),
            diagnostics.excludedTensorNames.contains(
                "language_model.model.embed_tokens_per_layer.biases")
        else {
            throw SandboxError.comparisonFailed(
                "paged loader materialized externalized per-layer tensors")
        }
    }
    if let metrics {
        guard metrics.rowStore.rowDataByteCount <= metrics.rowStore.rowMisses * 5_040,
            metrics.rowStore.largestRowDataRead <= 4_480,
            metrics.cachedRows <= cacheRows,
            metrics.dequantizedPayloadBytes < 16 * 1_024 * 1_024
        else {
            throw SandboxError.comparisonFailed("pager I/O or cache bound failed")
        }
    }
}

private func solidColorImage(_ color: String) -> CIImage {
    let ciColor = color == "red"
        ? CIColor(red: 1, green: 0, blue: 0, alpha: 1)
        : CIColor(red: 0, green: 0, blue: 1, alpha: 1)
    return CIImage(color: ciColor).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
}

private func isExclusiveColor(_ output: String, expected: String) -> Bool {
    output == expected
}

func validateModelDirectory(_ modelDirectory: URL) throws {
    let resolvedModel = modelDirectory.standardizedFileURL
    let safetensor = resolvedModel.appendingPathComponent("model.safetensors")
        .resolvingSymlinksInPath()
    let attributes = try FileManager.default.attributesOfItem(atPath: safetensor.path)
    guard (attributes[.size] as? NSNumber)?.uint64Value == 3_581_101_896 else {
        throw SandboxError.invalidSnapshot("model.safetensors does not match pinned byte count")
    }
    if resolvedModel == defaultModelDirectory.standardizedFileURL,
        resolvedModel.lastPathComponent != expectedModelRevision
    {
        throw SandboxError.invalidSnapshot("default model snapshot revision drifted")
    }
}

func allowedModelCacheRoot(_ modelDirectory: URL) -> URL {
    let snapshots = modelDirectory.deletingLastPathComponent()
    return snapshots.lastPathComponent == "snapshots"
        ? snapshots.deletingLastPathComponent() : modelDirectory
}

private func currentGitRevision() -> String {
    var repositoryRoot = URL(fileURLWithPath: #filePath)
    for _ in 0 ..< 5 { repositoryRoot.deleteLastPathComponent() }
    guard let head = runGit(["rev-parse", "HEAD"], in: repositoryRoot) else {
        return "unknown"
    }
    let status = runGit(["status", "--porcelain", "--untracked-files=normal"], in: repositoryRoot)
    return status?.isEmpty == true ? head : "\(head)-dirty"
}

private func runGit(_ arguments: [String], in directory: URL) -> String? {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.arguments = ["-C", directory.path] + arguments
    process.standardOutput = pipe
    process.standardError = Pipe()
    guard (try? process.run()) != nil else { return nil }
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { return nil }
    return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
}
