import SwiftUI
import ARKit
import RealityKit

struct ContentView: View {
    @StateObject private var scanningManager = ScanningManager()
    
    var body: some View {
        ZStack {
            ARViewContainer(scanningManager: scanningManager)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Spacer()
                
                // Status message
                Text(scanningManager.statusMessage)
                    .padding()
                    .background(Color.black.opacity(0.6))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .padding(.horizontal)
                
                // Progress indicator
                if scanningManager.state == .scanning || scanningManager.state == .processing {
                    ProgressView(value: scanningManager.progress)
                        .padding()
                        .frame(maxWidth: 300)
                }
                
                // Control buttons
                HStack(spacing: 20) {
                    switch scanningManager.state {
                    case .ready:
                        Button(action: {
                            scanningManager.startScanning()
                        }) {
                            Text("Start Scanning")
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                        
                    case .scanning:
                        Button(action: {
                            scanningManager.stopScanning()
                        }) {
                            Text("Stop Scanning")
                                .padding()
                                .background(Color.red)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                        
                    case .completed:
                        HStack {
                            Button(action: {
                                // Export mesh functionality here
                            }) {
                                Text("Export")
                                    .padding()
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            
                            Button(action: {
                                scanningManager.reset()
                            }) {
                                Text("New Scan")
                                    .padding()
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                        
                    case .processing:
                        Text("Processing...")
                            .padding()
                            .background(Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                        
                    case .failed(let error):
                        VStack {
                            Text("Error: \(error.localizedDescription)")
                                .foregroundColor(.red)
                                .padding()
                            
                            Button(action: {
                                scanningManager.reset()
                            }) {
                                Text("Try Again")
                                    .padding()
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                    }
                }
                .padding()
            }
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
