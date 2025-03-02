import SwiftUI
import RealityKit

struct ContentView: View {
    @StateObject private var scanningManager = ScanningManager.shared
    @State private var currentScreen: Screen = .guide
    @State private var showErrorAlert: Bool = false
    @State private var errorMessage: String = ""

    enum Screen {
        case guide, scanning, processing, meshView
    }

    var body: some View {
        ZStack {
            // Background color
            Color(.systemBackground)
                .edgesIgnoringSafeArea(.all)
            
            // Main content
            VStack {
                // Header
                Text("Prosthetic Scanner")
                    .font(.headline)
                    .padding()
                
                // Main content area
                mainContent
                
                // Bottom controls always visible
                bottomControls
            }
            .padding()
        }
        .alert("Error", isPresented: $showErrorAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage)
        }
    }
    
    // Main content based on current screen
    private var mainContent: some View {
        Group {
            switch currentScreen {
            case .guide:
                GuideView(showGuide: .constant(true))
                    .transition(.opacity)
                    .onAppear {
                        // Reset scanning state when returning to guide
                        scanningManager.reset()
                    }
                
            case .scanning:
                scanningView
                    .transition(.opacity)
                
            case .processing:
                processingView
                    .transition(.opacity)
                
            case .meshView:
                meshVisualizationView
                    .transition(.opacity)
            }
        }
        .animation(.default, value: currentScreen)
    }
    
    // View for active scanning
    private var scanningView: some View {
        VStack {
            // AR scanning view would be here
            ZStack {
                // This is a placeholder - in a real app, you'd use ARViewContainer
                Rectangle()
                    .fill(Color.black.opacity(0.8))
                    .aspectRatio(4/3, contentMode: .fit)
                    .cornerRadius(12)
                
                VStack {
                    Text("Move device around the object")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Text("\(scanningManager.pointCount) points captured")
                        .foregroundColor(.white)
                    
                    ProgressView(value: scanningManager.progress)
                        .progressViewStyle(LinearProgressViewStyle())
                        .padding()
                }
            }
            
            Text(scanningManager.statusMessage)
                .padding()
            
            if scanningManager.isScanning {
                Button(action: {
                    scanningManager.stopScanning()
                    
                    // Check if we have scan data
                    if let scanData = scanningManager.scanData {
                        currentScreen = .processing
                        processData(scanData)
                    } else {
                        showError("Not enough points captured. Try scanning again.")
                        currentScreen = .guide
                    }
                }) {
                    Text("Complete Scan")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                .padding(.horizontal)
            } else {
                Button(action: {
                    scanningManager.startScanning()
                }) {
                    Text("Start Scanning")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(10)
                }
                .padding(.horizontal)
            }
        }
    }
    
    // View for processing screen
    private var processingView: some View {
        ProcessingView(onProcessingComplete: {
            if scanningManager.meshData != nil {
                currentScreen = .meshView
            } else {
                showError("Processing failed. Please try again.")
                currentScreen = .guide
            }
        })
    }
    
    // View for mesh visualization
    private var meshVisualizationView: some View {
        Group {
            if let meshData = scanningManager.meshData {
                MeshVisualizationView(
                    meshData: meshData,
                    onExport: exportMesh,
                    onNewScan: { currentScreen = .guide }
                )
            } else {
                VStack {
                    Text("No mesh data available")
                        .font(.headline)
                    
                    Button("Return to Guide") {
                        currentScreen = .guide
                    }
                    .padding()
                }
            }
        }
    }
    
    // Bottom controls
    private var bottomControls: some View {
        HStack {
            Button(action: {
                // Reset everything and go back to guide
                scanningManager.reset()
                MeshProcessor.shared.reset()
                currentScreen = .guide
            }) {
                HStack {
                    Image(systemName: "arrow.counterclockwise")
                    Text("Reset")
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color(.systemGray5))
                .cornerRadius(8)
            }
            
            Spacer()
            
            // Status indicator
            Text(statusText)
                .font(.caption)
                .foregroundColor(.secondary)
                .animation(.none, value: currentScreen)
                
            Spacer()
            
            // Help button
            Button(action: {
                currentScreen = .guide
            }) {
                Image(systemName: "questionmark.circle")
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray5))
                    .cornerRadius(8)
            }
        }
        .padding(.top)
    }
    
    // Helper computed property for status text
    private var statusText: String {
        switch currentScreen {
        case .guide:
            return "Ready to scan"
        case .scanning:
            return "Scanning in progress"
        case .processing:
            return "Processing scan data"
        case .meshView:
            return "Viewing 3D model"
        }
    }
    
    // MARK: - Helper Methods
    
    // Process scan data
    private func processData(_ scanData: ScanData) {
        MeshProcessor.shared.processScanData(scanData) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let meshData):
                    scanningManager.meshData = meshData
                    currentScreen = .meshView
                case .failure(let error):
                    showError("Processing failed: \(error.localizedDescription)")
                    currentScreen = .guide
                }
            }
        }
    }
    
    // Export mesh
    private func exportMesh() {
        // Implementation would depend on how you want to export
        // (e.g., share sheet, save to documents, etc.)
        print("Export mesh functionality would be implemented here")
    }
    
    // Show error alert
    private func showError(_ message: String) {
        errorMessage = message
        showErrorAlert = true
    }
}

// MARK: - Preview Provider
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
