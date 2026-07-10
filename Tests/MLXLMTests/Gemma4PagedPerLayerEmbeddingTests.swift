// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

struct Gemma4PagedPerLayerEmbeddingTests {
    @Test("Gemma4PagedPerLayerEmbedding discovers opt-in tests")
    func discoverySentinel() {
        #expect(Gemma4PagedPerLayerEmbeddingTensorName.isExternalized(
            "language_model.model.embed_tokens_per_layer.weight"))
        #expect(Gemma4PagedPerLayerEmbeddingTensorName.isExternalized(
            "language_model.model.embed_tokens_per_layer.scales"))
        #expect(Gemma4PagedPerLayerEmbeddingTensorName.isExternalized(
            "language_model.model.embed_tokens_per_layer.biases"))
        #expect(!Gemma4PagedPerLayerEmbeddingTensorName.isExternalized(
            "language_model.model.embed_tokens.weight"))
    }

    @Test("SafeTensorRowStore reads fixed rows with pread metrics")
    func rowStoreReadsFixedRows() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Gemma4PagedPerLayerEmbeddingTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("model.safetensors")
        let rows: [[UInt32]] = [
            [0, 1],
            [42, 43],
            [100, 101],
        ]
        try writeSafetensors(
            url: url,
            tensors: [
                (
                    name: "tensor.weight",
                    dtype: "U32",
                    shape: [3, 2],
                    bytes: rows.flatMap { $0 }.littleEndianBytes()
                )
            ])

        let store = try SafeTensorRowStore(url: url, allowedRoot: directory)
        let row = try store.readRow(tensor: "tensor.weight", row: 1)
        #expect(row == rows[1].littleEndianBytes())
        #expect(store.metrics.rowDataByteCount == 8)
        #expect(store.metrics.largestRowDataRead == 8)
        #expect(store.metrics.metadataByteCount > 8)
    }

    @Test("SafeTensorRowStore rejects row bounds")
    func rowStoreRejectsOutOfBoundsRows() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Gemma4PagedPerLayerEmbeddingBounds-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            url: url,
            tensors: [
                (
                    name: "tensor.weight",
                    dtype: "U32",
                    shape: [1, 1],
                    bytes: [UInt32(7)].littleEndianBytes()
                )
            ])
        let store = try SafeTensorRowStore(url: url, allowedRoot: directory)

        #expect(throws: SafeTensorRowStoreError.self) {
            _ = try store.readRow(tensor: "tensor.weight", row: 1)
        }
    }

    @Test("SafeTensorRowStore rejects duplicate top-level tensor names")
    func rowStoreRejectsDuplicateTensorNames() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Gemma4PagedDuplicate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        let tensor = #"{"dtype":"U32","shape":[1],"data_offsets":[0,4]}"#
        let header = Data("{\"tensor\":\(tensor),\"tensor\":\(tensor)}".utf8)
        var output = UInt64(header.count).littleEndianBytes()
        output.append(header)
        output.append([UInt32(7)].littleEndianBytes())
        try output.write(to: url)

        #expect(throws: SafeTensorRowStoreError.self) {
            _ = try SafeTensorRowStore(url: url, allowedRoot: directory)
        }
    }

    @Test("SafeTensorRowStore rejects a symlink target outside the allowed root")
    func rowStoreRejectsEscapingSymlink() throws {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("Gemma4PagedSymlink-\(UUID().uuidString)")
        let root = parent.appendingPathComponent("root")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parent) }
        let outside = parent.appendingPathComponent("outside.safetensors")
        try writeSafetensors(
            url: outside,
            tensors: [("tensor", "U32", [1], [UInt32(7)].littleEndianBytes())])
        let link = root.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)

        #expect(throws: SafeTensorRowStoreError.self) {
            _ = try SafeTensorRowStore(url: link, allowedRoot: root)
        }
    }

    @Test("SafeTensorRowStore rejects oversized headers before allocation")
    func rowStoreRejectsOversizedHeader() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Gemma4PagedHeader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        try UInt64(100_000_001).littleEndianBytes().write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 100_000_009)
        try handle.close()

        #expect(throws: SafeTensorRowStoreError.self) {
            _ = try SafeTensorRowStore(url: url, allowedRoot: directory)
        }
    }
}

private func writeSafetensors(
    url: URL,
    tensors: [(name: String, dtype: String, shape: [Int], bytes: Data)]
) throws {
    var offset = 0
    var header: [String: Any] = [:]
    for tensor in tensors {
        header[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, offset + tensor.bytes.count],
        ]
        offset += tensor.bytes.count
    }
    let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var output = UInt64(headerData.count).littleEndianBytes()
    output.append(headerData)
    for tensor in tensors {
        output.append(tensor.bytes)
    }
    try output.write(to: url)
}

extension Array where Element == UInt32 {
    fileprivate func littleEndianBytes() -> Data {
        var data = Data()
        for value in self {
            var littleEndian = value.littleEndian
            Swift.withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }
}

extension UInt64 {
    fileprivate func littleEndianBytes() -> Data {
        var value = littleEndian
        return withUnsafeBytes(of: &value) { Data($0) }
    }
}
