import SwiftUI
import ARKit
import RealityKit

struct ContentView: View {
    @StateObject private var scanningManager = ScanningManager()
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
                    
                    // Overlay with scanning status and controls
                    VStack {
                        // Scanning status indicator at top
                        HStack {
                            Text("Point Count: \(scanningManager.pointCount)")
                                .font(.caption)
                            Spacer()
                            if let confidence = scanningManager.averageConfidence {
                                Text("Quality: \(Int(confidence * 100))%")
                                    .font(.caption)
                            }
                        }
                        .padding(8)
                        .background(.ultraThinMaterial)
                        .cornerRadius(8)
                        .padding(.horizontal)
                        .padding(.top, 8)
                        
                        // Scanning guidance in the middle
                        Text(scanningManager.statusMessage)
                            .font(.headline)
                            .padding()
                            .background(.ultraThinMaterial)
                            .cornerRadius(10)
                            .padding()
                        
                        Spacer()
                        
                        // Controls at bottom
                        VStack(spacing: 15) {
                            // Progress bar
                            if scanningManager.isScanning {
                                ProgressView(value: scanningManager.progress)
                                    .progressViewStyle(LinearProgressViewStyle())
                                    .padding(.horizontal)
                            }
                            
                            // Buttons
                            HStack(spacing: 20) {
                                if scanningManager.isScanning {
                                    Button(action: {
                                        scanningManager.stopScanning()
                                        currentScreen = .processing
                                    }) {
                                        Text("Complete Scan")
                                            .padding()
                                            .background(Color.green)
                                            .foregroundColor(.white)
                                            .cornerRadius(10)
                                    }
                                } else {
                                    Button(action: {
                                        scanningManager.startScanning()
                                    }) {
                                        Text("Start Scanning")
                                            .padding()
                                            .background(Color.blue)
                                            .foregroundColor(.white)
                                            .cornerRadius(10)
                                    }
                                }
                                
                                Button(action: {
                                    scanningManager.reset()
                                    showGuide = true
                                    currentScreen = .guide
                                }) {
                                    Text("Reset")
                                        .padding()
                                        .background(Color.red)
                                        .foregroundColor(.white)
                                        .cornerRadius(10)
                                }
                            }
                            .padding(.bottom)
                        }
                        .background(.ultraThinMaterial)
                        .cornerRadius(15)
                        .padding()
                    }
                }
                
            case .processing:
                ZStack {
                    Color.black.edgesIgnoringSafeArea(.all)
                    
                    VStack {
                        Text("Processing Scan...")
                            .font(.title)
                            .foregroundColor(.white)
                        
                        ProgressView(value: scanningManager.progress)
                            .progressViewStyle(LinearProgressViewStyle())
                            .padding()
                            .frame(width: 250)
                        
                        Text(scanningManager.statusMessage)
                            .foregroundColor(.white)
                            .padding()
                    }
                }
                .onAppear {
                    // After processing, transition to mesh view
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        processMeshGeneration()
                    }
                }
                
            case .meshView:
                MeshDisplayView(scanningManager: scanningManager, onExport: {
                    // Handle export
                    print("Exporting mesh...")
                }, onReset: {
                    scanningManager.reset()
                    showGuide = true
                    currentScreen = .guide
                })
            }
        }
    }
    
    private func processMeshGeneration() {
        // Processing already happens in stopScanning() but we need to handle the view transition
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            currentScreen = .meshView
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    var scanningManager: ScanningManager
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        scanningManager.setup(arView: arView)
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {}
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
