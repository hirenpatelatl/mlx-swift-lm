// Copyright © 2026 Apple Inc.

import Darwin
import Foundation

private let maximumSafeTensorHeaderByteCount: UInt64 = 100_000_000

public enum SafeTensorRowStoreError: LocalizedError {
    case invalidPath(URL)
    case pathEscapesRoot(path: URL, root: URL)
    case openFailed(String)
    case malformedHeader(String)
    case malformedTensor(String)
    case unsupportedDType(String)
    case tensorNotFound(String)
    case rowOutOfBounds(tensor: String, row: Int, rows: Int)
    case shortRead(tensor: String, expected: Int, actual: Int)
    case overflow(String)

    public var errorDescription: String? {
        switch self {
        case .invalidPath(let url):
            return "Invalid safetensors path: \(url.path)"
        case .pathEscapesRoot(let path, let root):
            return "Safetensors path \(path.path) escapes allowed root \(root.path)"
        case .openFailed(let message):
            return "Unable to open safetensors file: \(message)"
        case .malformedHeader(let message):
            return "Malformed safetensors header: \(message)"
        case .malformedTensor(let message):
            return "Malformed safetensors tensor metadata: \(message)"
        case .unsupportedDType(let dtype):
            return "Unsupported safetensors dtype: \(dtype)"
        case .tensorNotFound(let tensor):
            return "Tensor not found: \(tensor)"
        case .rowOutOfBounds(let tensor, let row, let rows):
            return "Row \(row) is out of bounds for \(tensor) with \(rows) rows"
        case .shortRead(let tensor, let expected, let actual):
            return "Short read for \(tensor): expected \(expected) bytes, got \(actual)"
        case .overflow(let message):
            return "Integer overflow while reading safetensors: \(message)"
        }
    }
}

public struct SafeTensorRowStoreMetrics: Sendable, Codable, Equatable {
    public var byteCount: UInt64 = 0
    public var largestRead: Int = 0
    public var metadataByteCount: UInt64 = 0
    public var largestMetadataRead: Int = 0
    public var rowDataByteCount: UInt64 = 0
    public var largestRowDataRead: Int = 0
    public var rowHits: UInt64 = 0
    public var rowMisses: UInt64 = 0
    public var evictions: UInt64 = 0
}

public struct SafeTensorRowStoreTensor: Sendable, Equatable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let dataStart: UInt64
    public let dataEnd: UInt64
    public let rowByteCount: Int

    public var rowCount: Int { shape.first ?? 1 }
    public var byteCount: UInt64 { dataEnd - dataStart }
}

public final class SafeTensorRowStore: @unchecked Sendable {
    public let url: URL
    public let root: URL
    public let tensors: [String: SafeTensorRowStoreTensor]

    private let fd: Int32
    private let lock = NSLock()
    private var mutableMetrics = SafeTensorRowStoreMetrics()

    public var metrics: SafeTensorRowStoreMetrics {
        lock.withLock { mutableMetrics }
    }

    public init(url: URL, allowedRoot: URL) throws {
        let resolvedURL = url.resolvingSymlinksInPath().standardizedFileURL
        let resolvedRoot = allowedRoot.resolvingSymlinksInPath().standardizedFileURL
        guard resolvedURL.isFileURL, resolvedRoot.isFileURL else {
            throw SafeTensorRowStoreError.invalidPath(url)
        }
        guard resolvedURL.path == resolvedRoot.path
            || resolvedURL.path.hasPrefix(resolvedRoot.path + "/")
        else {
            throw SafeTensorRowStoreError.pathEscapesRoot(path: resolvedURL, root: resolvedRoot)
        }

        let opened = Darwin.open(resolvedURL.path, O_RDONLY)
        guard opened >= 0 else {
            throw SafeTensorRowStoreError.openFailed(String(cString: strerror(errno)))
        }

        do {
            self.url = resolvedURL
            self.root = resolvedRoot
            self.fd = opened
            let index = try Self.readIndex(fd: opened, url: resolvedURL)
            self.tensors = index.tensors
            self.mutableMetrics.byteCount = index.metadataByteCount
            self.mutableMetrics.largestRead = index.largestMetadataRead
            self.mutableMetrics.metadataByteCount = index.metadataByteCount
            self.mutableMetrics.largestMetadataRead = index.largestMetadataRead
        } catch {
            Darwin.close(opened)
            throw error
        }
    }

    deinit {
        Darwin.close(fd)
    }

    public func tensor(named name: String) throws -> SafeTensorRowStoreTensor {
        guard let tensor = tensors[name] else {
            throw SafeTensorRowStoreError.tensorNotFound(name)
        }
        return tensor
    }

    public func readRow(tensor name: String, row: Int) throws -> Data {
        let tensor = try tensor(named: name)
        guard row >= 0, row < tensor.rowCount else {
            throw SafeTensorRowStoreError.rowOutOfBounds(
                tensor: name, row: row, rows: tensor.rowCount)
        }
        let rowOffset = try checkedMultiply(UInt64(row), UInt64(tensor.rowByteCount), "row offset")
        let offset = try checkedAdd(tensor.dataStart, rowOffset, "absolute row offset")
        _ = try checkedOffT(offset, byteCount: tensor.rowByteCount, label: name)
        return try readExactly(tensor: name, offset: offset, byteCount: tensor.rowByteCount)
    }

    public func recordCacheHit() {
        lock.withLock { mutableMetrics.rowHits += 1 }
    }

    public func recordCacheMiss() {
        lock.withLock { mutableMetrics.rowMisses += 1 }
    }

    public func recordEviction() {
        lock.withLock { mutableMetrics.evictions += 1 }
    }

    private func readExactly(tensor: String, offset: UInt64, byteCount: Int) throws -> Data {
        var data = Data(count: byteCount)
        let actual = data.withUnsafeMutableBytes { buffer in
            Darwin.pread(fd, buffer.baseAddress, byteCount, off_t(offset))
        }
        guard actual == byteCount else {
            throw SafeTensorRowStoreError.shortRead(
                tensor: tensor, expected: byteCount, actual: max(actual, 0))
        }
        lock.withLock {
            mutableMetrics.byteCount += UInt64(byteCount)
            mutableMetrics.largestRead = max(mutableMetrics.largestRead, byteCount)
            mutableMetrics.rowDataByteCount += UInt64(byteCount)
            mutableMetrics.largestRowDataRead = max(
                mutableMetrics.largestRowDataRead, byteCount)
        }
        return data
    }

    private struct IndexResult {
        let tensors: [String: SafeTensorRowStoreTensor]
        let metadataByteCount: UInt64
        let largestMetadataRead: Int
    }

    private static func readIndex(fd: Int32, url: URL) throws -> IndexResult {
        let fileSize = try fileByteCount(url: url)
        let prefix = try readPrefix(fd: fd, byteCount: 8)
        let headerLength = prefix.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard headerLength > 0,
            headerLength <= maximumSafeTensorHeaderByteCount,
            headerLength <= UInt64(Int.max)
        else {
            throw SafeTensorRowStoreError.malformedHeader("invalid header length \(headerLength)")
        }
        let dataBase = try checkedAdd(8, headerLength, "tensor data base")
        guard dataBase <= fileSize else {
            throw SafeTensorRowStoreError.malformedHeader("header exceeds file size")
        }

        let headerData = try readRange(fd: fd, offset: 8, byteCount: Int(headerLength))
        try rejectDuplicateTopLevelKeys(in: headerData)
        guard
            let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            throw SafeTensorRowStoreError.malformedHeader("header is not a JSON object")
        }

        var tensors: [String: SafeTensorRowStoreTensor] = [:]
        for (name, value) in object where name != "__metadata__" {
            guard let metadata = value as? [String: Any] else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) metadata is not an object")
            }
            guard let dtype = metadata["dtype"] as? String else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) has no dtype")
            }
            let bytesPerElement = try dtypeByteCount(dtype)
            guard let rawShape = metadata["shape"] as? [Any] else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) has no shape")
            }
            let shape = rawShape.compactMap(Self.int)
            guard shape.count == rawShape.count else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) has an invalid shape")
            }
            guard shape.allSatisfy({ $0 >= 0 }) else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) has a negative dimension")
            }
            guard let rawOffsets = metadata["data_offsets"] as? [Any], rawOffsets.count == 2,
                let relativeStart = Self.uint64(rawOffsets[0]),
                let relativeEnd = Self.uint64(rawOffsets[1])
            else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) has invalid data_offsets")
            }
            guard relativeStart <= relativeEnd else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) offsets are reversed")
            }
            let elementCount = try shape.reduce(UInt64(1)) {
                try checkedMultiply($0, UInt64($1), "\(name) element count")
            }
            let expectedBytes = try checkedMultiply(
                elementCount, UInt64(bytesPerElement), "\(name) byte count")
            guard relativeEnd - relativeStart == expectedBytes else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) byte count mismatch")
            }
            let absoluteStart = try checkedAdd(dataBase, relativeStart, "\(name) data start")
            let absoluteEnd = try checkedAdd(dataBase, relativeEnd, "\(name) data end")
            guard absoluteEnd <= fileSize else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) exceeds file size")
            }
            guard expectedBytes <= UInt64(Int.max) else {
                throw SafeTensorRowStoreError.overflow("\(name) tensor byte count")
            }
            _ = try checkedOffT(absoluteStart, byteCount: Int(expectedBytes), label: name)
            let rowCount = shape.first ?? 1
            guard rowCount > 0, expectedBytes % UInt64(rowCount) == 0 else {
                throw SafeTensorRowStoreError.malformedTensor("\(name) cannot be read by rows")
            }
            let rowBytes = expectedBytes / UInt64(rowCount)
            guard rowBytes <= UInt64(Int.max) else {
                throw SafeTensorRowStoreError.overflow("\(name) row byte count")
            }
            tensors[name] = SafeTensorRowStoreTensor(
                name: name,
                dtype: dtype,
                shape: shape,
                dataStart: absoluteStart,
                dataEnd: absoluteEnd,
                rowByteCount: Int(rowBytes)
            )
        }
        return IndexResult(
            tensors: tensors,
            metadataByteCount: try checkedAdd(8, headerLength, "metadata byte count"),
            largestMetadataRead: max(8, Int(headerLength)))
    }

    private static func uint64(_ value: Any) -> UInt64? {
        if let value = value as? UInt64 {
            return value
        }
        if let value = value as? Int, value >= 0 {
            return UInt64(value)
        }
        if let value = value as? NSNumber, value.int64Value >= 0 {
            return UInt64(value.uint64Value)
        }
        return nil
    }

    private static func int(_ value: Any) -> Int? {
        if let value = value as? Int {
            return value
        }
        if let value = value as? NSNumber {
            return value.intValue
        }
        return nil
    }
}

private func dtypeByteCount(_ dtype: String) throws -> Int {
    switch dtype {
    case "BOOL", "U8", "I8":
        return 1
    case "F16", "BF16", "I16", "U16":
        return 2
    case "F32", "I32", "U32":
        return 4
    case "F64", "I64", "U64":
        return 8
    default:
        throw SafeTensorRowStoreError.unsupportedDType(dtype)
    }
}

private func fileByteCount(url: URL) throws -> UInt64 {
    let values = try url.resourceValues(forKeys: [.fileSizeKey])
    guard let size = values.fileSize, size >= 0 else {
        throw SafeTensorRowStoreError.invalidPath(url)
    }
    return UInt64(size)
}

private func readPrefix(fd: Int32, byteCount: Int) throws -> Data {
    try readRange(fd: fd, offset: 0, byteCount: byteCount)
}

private func readRange(fd: Int32, offset: UInt64, byteCount: Int) throws -> Data {
    _ = try checkedOffT(offset, byteCount: byteCount, label: "metadata")
    var data = Data(count: byteCount)
    let actual = data.withUnsafeMutableBytes { buffer in
        Darwin.pread(fd, buffer.baseAddress, byteCount, off_t(offset))
    }
    guard actual == byteCount else {
        throw SafeTensorRowStoreError.shortRead(
            tensor: "header", expected: byteCount, actual: max(actual, 0))
    }
    return data
}

private func checkedOffT(_ offset: UInt64, byteCount: Int, label: String) throws -> off_t {
    guard byteCount >= 0, offset <= UInt64(Int64.max) else {
        throw SafeTensorRowStoreError.overflow("\(label) offset")
    }
    let end = try checkedAdd(offset, UInt64(byteCount), "\(label) read end")
    guard end <= UInt64(Int64.max) else {
        throw SafeTensorRowStoreError.overflow("\(label) read end")
    }
    return off_t(offset)
}

private func rejectDuplicateTopLevelKeys(in data: Data) throws {
    var scanner = JSONTopLevelKeyScanner(bytes: Array(data))
    let keys = try scanner.scan()
    var seen = Set<String>()
    for key in keys where !seen.insert(key).inserted {
        throw SafeTensorRowStoreError.malformedHeader("duplicate tensor name \(key)")
    }
}

private struct JSONTopLevelKeyScanner {
    let bytes: [UInt8]
    var index = 0

    mutating func scan() throws -> [String] {
        skipWhitespace()
        try consume(123, "expected top-level object")
        skipWhitespace()
        if consumeIf(125) { return [] }
        var keys: [String] = []
        while true {
            skipWhitespace()
            let keyBytes = try scanStringBytes()
            guard let key = try? JSONDecoder().decode(String.self, from: Data(keyBytes)) else {
                throw SafeTensorRowStoreError.malformedHeader("invalid object key")
            }
            keys.append(key)
            skipWhitespace()
            try consume(58, "expected colon after object key")
            skipWhitespace()
            try skipValue()
            skipWhitespace()
            if consumeIf(125) { break }
            try consume(44, "expected comma between tensors")
        }
        skipWhitespace()
        guard index == bytes.count else {
            throw SafeTensorRowStoreError.malformedHeader("trailing bytes after JSON object")
        }
        return keys
    }

    mutating func skipValue() throws {
        guard index < bytes.count else {
            throw SafeTensorRowStoreError.malformedHeader("missing value")
        }
        switch bytes[index] {
        case 34:
            _ = try scanStringBytes()
        case 123, 91:
            let opener = bytes[index]
            let closer: UInt8 = opener == 123 ? 125 : 93
            index += 1
            while true {
                skipWhitespace()
                if consumeIf(closer) { return }
                try skipValue()
                skipWhitespace()
                if consumeIf(closer) { return }
                if opener == 123 {
                    try consume(58, "expected colon in nested object")
                    skipWhitespace()
                    try skipValue()
                    skipWhitespace()
                    if consumeIf(closer) { return }
                }
                try consume(44, "expected comma in nested value")
            }
        default:
            while index < bytes.count, ![44, 93, 125].contains(bytes[index]) {
                index += 1
            }
        }
    }

    mutating func scanStringBytes() throws -> [UInt8] {
        guard index < bytes.count, bytes[index] == 34 else {
            throw SafeTensorRowStoreError.malformedHeader("expected JSON string")
        }
        let start = index
        index += 1
        var escaped = false
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if escaped {
                escaped = false
            } else if byte == 92 {
                escaped = true
            } else if byte == 34 {
                return Array(bytes[start ..< index])
            }
        }
        throw SafeTensorRowStoreError.malformedHeader("unterminated JSON string")
    }

    mutating func skipWhitespace() {
        while index < bytes.count, [9, 10, 13, 32].contains(bytes[index]) { index += 1 }
    }

    mutating func consume(_ byte: UInt8, _ message: String) throws {
        guard consumeIf(byte) else { throw SafeTensorRowStoreError.malformedHeader(message) }
    }

    mutating func consumeIf(_ byte: UInt8) -> Bool {
        guard index < bytes.count, bytes[index] == byte else { return false }
        index += 1
        return true
    }
}

private func checkedAdd(_ lhs: UInt64, _ rhs: UInt64, _ label: String) throws -> UInt64 {
    let (value, overflow) = lhs.addingReportingOverflow(rhs)
    if overflow {
        throw SafeTensorRowStoreError.overflow(label)
    }
    return value
}

private func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, _ label: String) throws -> UInt64 {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    if overflow {
        throw SafeTensorRowStoreError.overflow(label)
    }
    return value
}
