import CryptoKit
import Foundation
import MLXLMCommon

final class PinnedGemma4Tokenizer: Tokenizer, @unchecked Sendable {
    private let vocabulary: [String: Int]
    private let tokensByID: [Int: String]
    private let specialTokenIDs: Set<Int>
    let sha256: String

    let bosToken: String? = "<bos>"
    let eosToken: String? = "<eos>"
    let unknownToken: String? = "<unk>"

    init(modelDirectory: URL) throws {
        let data = try Data(contentsOf: modelDirectory.appendingPathComponent("tokenizer.json"))
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == expectedTokenizerSHA256 else {
            throw SandboxError.invalidSnapshot(
                "tokenizer.json SHA-256 \(digest) does not match pinned \(expectedTokenizerSHA256)")
        }
        let file = try JSONDecoder().decode(TokenizerFile.self, from: data)
        // Foundation normalizes some Unicode-equivalent JSON object keys while decoding.
        // The immutable file hash is the authoritative full-vocabulary identity check.
        guard file.model.type == "BPE", file.model.vocab.count > 260_000,
            file.model.vocab["<bos>"] == 2,
            file.model.vocab["red"] == 1192,
            file.model.vocab["blue"] == 9503
        else {
            throw SandboxError.invalidSnapshot("unexpected pinned tokenizer model or vocabulary")
        }
        guard file.decoder.isExpectedGemma4Decoder else {
            throw SandboxError.invalidSnapshot("unexpected pinned tokenizer decoder pipeline")
        }
        self.vocabulary = file.model.vocab
        self.tokensByID = Dictionary(uniqueKeysWithValues: file.model.vocab.map { ($0.value, $0.key) })
        self.specialTokenIDs = Set(file.addedTokens.filter(\.special).map(\.id))
        self.sha256 = digest
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if text == colorPrompt { return Array(Self.promptTokens.dropFirst(addSpecialTokens ? 0 : 1)) }
        return vocabulary[text].map { [$0] } ?? [3]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var result = ""
        var byteBuffer: [UInt8] = []
        func flushBytes() {
            if !byteBuffer.isEmpty {
                result += String(decoding: byteBuffer, as: UTF8.self)
                byteBuffer.removeAll(keepingCapacity: true)
            }
        }
        for id in tokenIds {
            if skipSpecialTokens, specialTokenIDs.contains(id) { continue }
            guard let token = tokensByID[id] else { continue }
            if token.hasPrefix("<0x"), token.hasSuffix(">"), token.count == 6,
                let byte = UInt8(token.dropFirst(3).dropLast(), radix: 16)
            {
                byteBuffer.append(byte)
            } else {
                flushBytes()
                result += token.replacingOccurrences(of: "▁", with: " ")
            }
        }
        flushBytes()
        return result
    }

    func convertTokenToId(_ token: String) -> Int? { vocabulary[token] }
    func convertIdToToken(_ id: Int) -> String? { tokensByID[id] }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        guard tools == nil, additionalContext == nil, messages.count == 1 else {
            throw SandboxError.invalidOption("sandbox accepts exactly one image prompt")
        }
        let printable = String(describing: messages[0])
        guard printable.contains(colorPrompt), printable.contains("image") else {
            throw SandboxError.invalidOption("prompt differs from the pinned color-classification prompt")
        }
        return Self.promptTokens
    }

    static let promptTokens = [
        2, 105, 2364, 107, 258_880, 7925, 607, 7121, 886, 67505, 3658, 236_787,
        2604, 653, 3730, 236_761, 2900, 2258, 563, 672, 2471, 236_881, 106, 107,
        105, 4368, 107,
    ]
}

private struct TokenizerFile: Decodable {
    struct Model: Decodable { let type: String; let vocab: [String: Int] }
    struct AddedToken: Decodable {
        let id: Int
        let special: Bool
    }
    struct Decoder: Decodable {
        struct Entry: Decodable {
            let type: String
            let content: String?
            struct Pattern: Decodable { let string: String; enum CodingKeys: String, CodingKey { case string = "String" } }
            let pattern: Pattern?
        }
        let type: String
        let decoders: [Entry]
        var isExpectedGemma4Decoder: Bool {
            type == "Sequence" && decoders.map(\.type) == ["Replace", "ByteFallback", "Fuse"]
                && decoders.first?.pattern?.string == "▁" && decoders.first?.content == " "
        }
    }
    let model: Model
    let addedTokens: [AddedToken]
    let decoder: Decoder
    enum CodingKeys: String, CodingKey { case model, decoder; case addedTokens = "added_tokens" }
}
