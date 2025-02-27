import SwiftUI
import RealityKit

struct ContentView: View {
    @StateObject private var scanningManager = ScanningManager()
    @StateObject private var meshProcessor = MeshProcessor.shared
    @State private var showGuide = true
    @State private var currentScreen: ScanScreen = .guide
    
    enum ScanScreen {
        case guide
        case scanning
        case processing
        case meshView
    }
    
    var body: some View {
        ZStack {
            switch currentScreen {
            case .guide:
                GuideView(showGuide: $showGuide)
                    .onChange(of: showGuide) { _, _ in
                        if !showGuide {
                            currentScreen = .scanning
                        }
                    }
                
            case .scanning:
                ZStack {
                    // AR View is the background
                    ARViewContainer(scanningManager: scanningManager)
                        .edgesIgnoringSafeArea(.all)
                    
                    // Controls overlay
                    VStack {
                        Text(scanningManager.statusMessage)
                            .font(.headline)
                            .padding()
                            .background(.ultraThinMaterial)
                            .cornerRadius(10)
                        
                        Spacer()
                        
                        VStack {
                            Button("Start Scanning") {
                                scanningManager.startScanning()
                            }
                            .padding()
                            .background(.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                            
                            if scanningManager.state == .scanning {
                                ProgressView(value: scanningManager.progress)
                                    .progressViewStyle(LinearProgressViewStyle())
                                    .padding()
                                
                                Button("Complete Scan") {
                                    completeScan()
                                }
                                .padding()
                                .background(.green)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                            }
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .cornerRadius(10)
                    }
                    .padding()
                }
                
            case .processing:
                ProcessingView(meshProcessor: meshProcessor) { meshData in
                    currentScreen = .meshView
                }
                
            case .meshView:
                MeshVisualizationView(
                    meshData: meshProcessor.meshData,
                    onExport: exportMesh,
                    onNewScan: resetScan
                )
            }
        }
    }
    
    // Function to complete the scan and move to the processing screen
    private func completeScan() {
        scanningManager.stopScanning()  // Stops the scanning process in ScanningManager
        
        // Prepare scan data for processing
        let scanData = ScanData(
            points: scanningManager.points,
            normals: scanningManager.normals,
            confidences: scanningManager.confidences,
            colors: scanningManager.colors
        )
        
        currentScreen = .processing
        processScannedData(scanData)
    }
    
    // Process the scanned data by passing it to MeshProcessor for mesh generation
    private func processScannedData(_ scanData: ScanData) {
        meshProcessor.processScanData(scanData) { result in
            switch result {
            case .success:
                currentScreen = .meshView
            case .failure(let error):
                print("Mesh generation failed: \(error)")
            }
        }
    }
    
    // Function to handle exporting the mesh data
    private func exportMesh() {
        print("Mesh export function called.")
    }
    
    // Function to reset both scanning and mesh processing states
    private func resetScan() {
        scanningManager.reset()
        meshProcessor.reset()
        currentScreen = .scanning
    }
}

// Add ARViewContainer right after ContentView
struct ARViewContainer: UIViewRepresentable {
    var scanningManager: ScanningManager
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        scanningManager.setup(arView: arView)
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {}
}

// Preview provider if needed
#Preview {
    ContentView()
}
