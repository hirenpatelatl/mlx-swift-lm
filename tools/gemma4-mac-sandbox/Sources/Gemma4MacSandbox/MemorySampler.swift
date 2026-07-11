import Darwin
import Foundation

struct MemorySample: Codable {
    let physicalFootprint: UInt64
    let residentSetSize: UInt64

    static func current() -> MemorySample {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return MemorySample(physicalFootprint: 0, residentSetSize: 0)
        }
        return MemorySample(
            physicalFootprint: UInt64(info.phys_footprint),
            residentSetSize: UInt64(info.resident_size))
    }
}

final class PeakMemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private let timer: DispatchSourceTimer
    private var peak: MemorySample

    init(intervalMilliseconds: Int = 5) {
        peak = .current()
        timer = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
        timer.schedule(
            deadline: .now(), repeating: .milliseconds(intervalMilliseconds),
            leeway: .milliseconds(1))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
    }

    func stop() -> MemorySample {
        sample()
        timer.cancel()
        return lock.withLock { peak }
    }

    private func sample() {
        let current = MemorySample.current()
        lock.withLock {
            peak = MemorySample(
                physicalFootprint: max(peak.physicalFootprint, current.physicalFootprint),
                residentSetSize: max(peak.residentSetSize, current.residentSetSize))
        }
    }
}

func thermalStateDescription() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: "nominal"
    case .fair: "fair"
    case .serious: "serious"
    case .critical: "critical"
    @unknown default: "unknown"
    }
}
