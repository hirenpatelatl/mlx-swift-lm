import Darwin
import Foundation

@main
struct Gemma4MacSandbox {
    static func main() async {
        do {
            try await run()
        } catch {
            fputs("\(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }

    private static func run() async throws {
        var arguments = Array(CommandLine.arguments.dropFirst())
        guard let command = arguments.first else { throw SandboxError.usage }
        arguments.removeFirst()
        let options = Options(arguments)
        switch command {
        case "self-test":
            let metalHash = try prepareMetalLibrary(override: options.metalLibraryURL)
            try runSelfTest(modelDirectory: options.modelDirectory, metalLibrarySHA256: metalHash)
        case "run":
            let metalHash = try prepareMetalLibrary(override: options.metalLibraryURL)
            try await runOne(
                mode: options.required("mode"),
                color: options.required("color"),
                modelDirectory: options.modelDirectory,
                output: options.outputURL,
                cacheRows: options.cacheRows,
                metalLibrarySHA256: metalHash)
        case "compare":
            try compareReceipts(at: options.requiredURL("receipts"))
        default:
            throw SandboxError.usage
        }
    }
}
