import SwiftUI

struct ContentView: View {
    @StateObject private var scanningManager = ScanningManager.shared
    @StateObject private var meshProcessor = MeshProcessor.shared
    @State private var currentScreen: Screen = .guide

    enum Screen {
        case guide, scanning, processing, meshView
    }

    var body: some View {
        VStack {
            // Status area
            Text(scanningManager.statusMessage)
                .padding()
            
            // Main content area
            switch currentScreen {
            case .guide:
                Button(action: {
                    scanningManager.startScanning()
                    currentScreen = .scanning
                }) {
                    Text("Start Scanning")
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }

            case .scanning:
                if scanningManager.isScanning {
                    VStack {
                        Text("Points captured: \(scanningManager.pointCount)")
                        
                        ProgressView(value: scanningManager.progress)
                            .padding()
                        
                        Button(action: {
                            scanningManager.stopScanning()
                            currentScreen = .processing
                            
                            // Process the scan data
                            meshProcessor.processScanData(scanningManager.scanData) { result in
                                switch result {
                                case .success(let meshData):
                                    DispatchQueue.main.async {
                                        scanningManager.meshData = meshData
                                        currentScreen = .meshView
                                    }
                                case .failure(let error):
                                    print("Processing failed: \(error)")
                                    currentScreen = .guide // Fallback on error
                                }
                            }
                        }) {
                            Text("Complete Scan")
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                } else {
                    Text("Scanning stopped")
                }

            case .processing:
                VStack {
                    Text("Processing scan...")
                    ProgressView(value: meshProcessor.processingProgress)
                    Text(meshProcessor.processingMessage)
                }

            case .meshView:
                if let meshData = scanningManager.meshData {
                    MeshDisplayView(meshData: meshData)
                } else {
                    Text("No mesh data available")
                }
            }

            // Reset button available in all screens
            Button(action: {
                scanningManager.reset()
                currentScreen = .guide
            }) {
                Text("Reset")
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()
        }
    }
}