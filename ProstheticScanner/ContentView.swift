import SwiftUI

struct ContentView: View {
    @StateObject private var scanningManager = ScanningManager.shared
    @State private var currentScreen: Screen = .guide

    enum Screen {
        case guide, scanning, processing, meshView
    }

    var body: some View {
        switch currentScreen {
        case .guide:
            Button(action: {
                scanningManager.startScanning()
                currentScreen = .scanning
            }) {
                Text("Start Scanning")
            }

        case .scanning:
            if scanningManager.isScanning {
                Button(action: {
                    scanningManager.stopScanning()
                    currentScreen = .processing
                    MeshProcessor.shared.processScanData(scanningManager.scanData) { result in
                        switch result {
                        case .success(let meshData):
                            scanningManager.meshData = meshData
                            currentScreen = .meshView
                        case .failure(let error):
                            print("Processing failed: \(error)")
                            currentScreen = .guide // Fallback on error
                        }
                    }
                }) {
                    Text("Complete Scan")
                }
            }

        case .processing:
            ProcessingView(onProcessingComplete: {
                currentScreen = .meshView
            })

        case .meshView:
            if let meshData = scanningManager.meshData {
                MeshDisplayView(meshData: meshData)
            }
        }

        // Reset button available in all screens
        Button(action: {
            scanningManager.reset()
            MeshProcessor.shared.reset()
            currentScreen = .guide
        }) {
            Text("Reset")
        }
    }
}