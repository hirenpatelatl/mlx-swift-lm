import CryptoKit
import Foundation

func prepareMetalLibrary(override: URL?) throws -> String {
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    let destination = executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
    let candidates = [
        override,
        URL(fileURLWithPath:
            "/Applications/SyncNotesMac.app/Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"),
        URL(fileURLWithPath:
            "/tmp/DerivedData-SyncNotes-Gemma4-Mac/Build/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"),
        URL(fileURLWithPath:
            "/Users/hirenpatel/Developer/Local/tmp/phase-c-vision-harness/default.metallib"),
    ].compactMap { $0 }

    guard let source = try candidates.first(where: { candidate in
        guard FileManager.default.fileExists(atPath: candidate.path) else { return false }
        return try sha256(candidate) == expectedMetalLibrarySHA256
    }) else {
        throw SandboxError.invalidSnapshot(
            "no matching MLX metallib found; pass --metallib with SHA-256 \(expectedMetalLibrarySHA256)")
    }

    if FileManager.default.fileExists(atPath: destination.path)
        || (try? destination.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true
    {
        if try sha256(destination) == expectedMetalLibrarySHA256 {
            return expectedMetalLibrarySHA256
        }
        try FileManager.default.removeItem(at: destination)
    }
    try FileManager.default.createSymbolicLink(
        at: destination, withDestinationURL: source.resolvingSymlinksInPath())
    return expectedMetalLibrarySHA256
}

private func sha256(_ url: URL) throws -> String {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
