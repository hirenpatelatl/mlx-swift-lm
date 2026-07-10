import Foundation

func compareReceipts(at directory: URL) throws {
    let files = try FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "json" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    let receipts = try files.map {
        try JSONDecoder().decode(RunReceipt.self, from: Data(contentsOf: $0))
    }

    let modes = ["resident", "paged"]
    let colors = ["red", "blue"]
    let fixedRunCounts = modes.allSatisfy { mode in
        colors.allSatisfy { color in
            receipts.filter { $0.mode == mode && $0.color == color }.count == 3
        }
    } && receipts.count == 12
    let exclusiveColorsCorrect = receipts.allSatisfy {
        $0.exclusiveColorCorrect && $0.outputText == $0.color
    }
    let deterministicTokenIDs = colors.allSatisfy { color in
        let colorRuns = receipts.filter { $0.color == color }
        guard let reference = colorRuns.first?.generatedTokenIDs, !reference.isEmpty else {
            return false
        }
        return colorRuns.allSatisfy { $0.generatedTokenIDs == reference }
    }
    let residentMedian = median(
        receipts.filter { $0.mode == "resident" }.map(\.peakPhysicalFootprint))
    let pagedMedian = median(
        receipts.filter { $0.mode == "paged" }.map(\.peakPhysicalFootprint))
    let reduction = residentMedian > pagedMedian ? residentMedian - pagedMedian : 0
    let audioBytes = receipts.map(\.loaderDiagnostics.evaluatedAudioBytes).max() ?? 0
    let pagedMetrics = receipts.filter { $0.mode == "paged" }.compactMap(\.pagedMetrics)
    let pagerReadsBounded = pagedMetrics.count == 6 && pagedMetrics.allSatisfy {
        $0.rowStore.rowDataByteCount <= $0.rowStore.rowMisses * 5_040
            && $0.rowStore.largestRowDataRead <= 4_480
    }
    let cacheBounded = pagedMetrics.count == 6 && pagedMetrics.allSatisfy {
        $0.cachedRows <= 512 && $0.dequantizedPayloadBytes < 16 * 1_024 * 1_024
    }
    let residentHasNoPager = receipts.filter { $0.mode == "resident" }
        .allSatisfy { $0.pagedMetrics == nil }
    let pinnedInputs = receipts.allSatisfy {
        $0.modelRevision == expectedModelRevision
            && $0.tokenizerSHA256 == expectedTokenizerSHA256
            && $0.metalLibrarySHA256 == expectedMetalLibrarySHA256
    }
    let passed = fixedRunCounts && exclusiveColorsCorrect && deterministicTokenIDs
        && reduction >= 1_073_741_824 && audioBytes == 0 && pagerReadsBounded
        && cacheBounded && residentHasNoPager && pinnedInputs
    let comparison = ComparisonReceipt(
        result: passed ? "pass" : "fail",
        receiptCount: receipts.count,
        residentMedianPeakPhysicalFootprint: residentMedian,
        pagedMedianPeakPhysicalFootprint: pagedMedian,
        medianReductionBytes: reduction,
        minimumRequiredReductionBytes: 1_073_741_824,
        deterministicTokenIDs: deterministicTokenIDs,
        exclusiveColorsCorrect: exclusiveColorsCorrect,
        fixedRunCounts: fixedRunCounts,
        audioEvaluatedBytes: audioBytes,
        pagerReadsBounded: pagerReadsBounded,
        cacheBounded: cacheBounded)
    try printJSON(comparison)
    guard passed else {
        throw SandboxError.comparisonFailed(
            "comparison failed: colors=\(exclusiveColorsCorrect), tokens=\(deterministicTokenIDs), reduction=\(reduction), reads=\(pagerReadsBounded), cache=\(cacheBounded), audio=\(audioBytes)")
    }
}

private func median(_ values: [UInt64]) -> UInt64 {
    let sorted = values.sorted()
    guard !sorted.isEmpty else { return 0 }
    if sorted.count % 2 == 1 { return sorted[sorted.count / 2] }
    let lower = sorted[sorted.count / 2 - 1]
    let upper = sorted[sorted.count / 2]
    return lower + (upper - lower) / 2
}
