import SwiftUI
import SceneKit

struct MeshDisplayView: View {
    var scanningManager: ScanningManager
    var onExport: () -> Void
    var onReset: () -> Void
    
    @State private var rotationAngle: CGFloat = 0
    
    var body: some View {
        ZStack {
            // 3D mesh display using SceneKit
            SceneView(scene: createScene(), options: [.allowsCameraControl, .autoenablesDefaultLighting])
                .edgesIgnoringSafeArea(.all)
            
            // Mesh statistics overlay
            VStack {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Scan Statistics:")
                        .font(.headline)
                    Text("Points: \(scanningManager.pointCount)")
                    Text("Triangles: \(scanningManager.triangleCount)")
                    if let confidence = scanningManager.averageConfidence {
                        Text("Avg. Confidence: \(Int(confidence * 100))%")
                    }
                    if let density = scanningManager.pointDensity {
                        Text("Point Density: \(Int(density)) pts/cm²")
                    }
                }
                .padding()
                .background(.ultraThinMaterial)
                .cornerRadius(10)
                .padding()
                
                Spacer()
                
                // Export and reset buttons
                HStack(spacing: 20) {
                    Button(action: onExport) {
                        Text("Export OBJ")
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    
                    Button(action: onReset) {
                        Text("New Scan")
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                }
                .padding()
            }
        }
    }
    
    private func createScene() -> SCNScene {
        let scene = SCNScene()
        
        // Convert MDLMesh to SCNNode if available
        if let mesh = scanningManager.scannedMesh {
            let scnNode = SCNNode(mdlObject: mesh)
            
            // Adjust material to make it visible
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.white
            material.lightingModel = .physicallyBased
            scnNode.geometry?.firstMaterial = material
            
            // Center the mesh
            scnNode.position = SCNVector3(0, 0, 0)
            
            // Add to scene
            scene.rootNode.addChildNode(scnNode)
        }
        
        return scene
    }
} 