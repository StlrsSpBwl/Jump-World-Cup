import AppKit
import Foundation
import Vision

struct OCRLine: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
    let text: String
}

guard CommandLine.arguments.count > 1 else {
    fputs("Usage: vision_ocr IMAGE [IMAGE ...]\n", stderr)
    exit(2)
}

let encoder = JSONEncoder()

for imagePath in CommandLine.arguments.dropFirst() {
    guard
        let image = NSImage(contentsOfFile: imagePath),
        let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
    else {
        fputs("Could not read image: \(imagePath)\n", stderr)
        continue
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["en-US"]

    do {
        try VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([request])
        let lines = (request.results ?? []).compactMap { observation -> OCRLine? in
            guard let candidate = observation.topCandidates(1).first else {
                return nil
            }
            let box = observation.boundingBox
            return OCRLine(
                x: box.origin.x,
                y: box.origin.y,
                width: box.width,
                height: box.height,
                text: candidate.string
            )
        }.sorted {
            if abs($0.y - $1.y) > 0.008 {
                return $0.y > $1.y
            }
            return $0.x < $1.x
        }

        let payload = try encoder.encode(lines)
        print(String(data: payload, encoding: .utf8) ?? "[]")
    } catch {
        fputs("Vision OCR failed for \(imagePath): \(error)\n", stderr)
        print("[]")
    }
}
