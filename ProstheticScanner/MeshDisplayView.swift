import SwiftUI
import SceneKit

struct MeshDisplayView: View {
    var scanningManager: ScanningManager
    var onExport: () -> Void
    var onReset: () -> Void
    
    @State private var displayMode: DisplayMode = .wireframe
    @State private var showMeasurements = false
    @State private var rotationDegrees: Double = 0
    
    enum DisplayMode: String, CaseIterable {
        case wireframe = "Wireframe"
        case solid = "Solid"
        case points = "Points"
        case textured = "Textured"
    }
    
    var body: some View {
        ZStack {
            // 3D mesh display using SceneKit
            SceneView(scene: createScene(), options: [.allowsCameraControl, .autoenablesDefaultLighting])
                .edgesIgnoringSafeArea(.all)
            
            // Controls overlay
            VStack {
                // Top controls - statistics and view modes
                HStack {
                    // Mesh statistics in a capsule
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Points: \(scanningManager.pointCount)")
                                .font(.caption)
                            Spacer()
                            Text("Triangles: \(scanningManager.triangleCount)")
                                .font(.caption)
                        }
                        
                        if let confidence = scanningManager.averageConfidence {
                            Text("Quality: \(Int(confidence * 100))%")
                                .font(.caption)
                        }
                    }
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(10)
                    
                    Spacer()
                    
                    // Display mode picker
                    Picker("Display", selection: $displayMode) {
                        ForEach(DisplayMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .frame(maxWidth: 200)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                }
                .padding()
                
                Spacer()
                
                // Bottom controls - actions and measurements
                VStack(spacing: 12) {
                    // Measurement toggle
                    Toggle("Show Measurements", isOn: $showMeasurements)
                        .padding(.horizontal)
                    
                    // Export and reset buttons
                    HStack(spacing: 20) {
                        Button(action: onExport) {
                            Label("Export", systemImage: "square.and.arrow.up")
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                        
                        Button(action: onReset) {
                            Label("New Scan", systemImage: "camera")
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                }
                .padding()
                .background(.ultraThinMaterial)
                .cornerRadius(15)
                .padding()
            }
        }
        .onChange(of: displayMode) { _, newMode in
            updateDisplayMode(newMode)
        }
    }
    
    private func createScene() -> SCNScene {
        let scene = SCNScene()
        
        if let mesh = scanningManager.scannedMesh {
            do {
                // Create an MDL asset from the mesh
                let asset = MDLAsset()
                asset.add(mesh)
                
                // Convert to SCNScene
                guard let tempScene = SCNScene(mdlAsset: asset) else {
                    print("Failed to create scene from asset")
                    createFallbackSphere(in: scene)
                    return scene
                }
                
                // Extract the node
                if let meshNode = tempScene.rootNode.childNodes.first {
                    // Apply material based on display mode
                    applyMaterial(to: meshNode, mode: displayMode)
                    
                    // Add to our scene
                    scene.rootNode.addChildNode(meshNode)
                    
                    // Add measurement indicators if needed
                    if showMeasurements {
                        addMeasurementsToScene(scene, for: mesh)
                    }
                }
            } catch {
                print("Failed to convert mesh: \(error)")
                // Fall back to a simple sphere
                createFallbackSphere(in: scene)
            }
        } else {
            // No mesh available, create fallback sphere
            createFallbackSphere(in: scene)
        }
        
        return scene
    }
    
    private func createFallbackSphere(in scene: SCNScene) {
        let sphereGeometry = SCNSphere(radius: 0.1)
        let sphereNode = SCNNode(geometry: sphereGeometry)
        
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.blue
        sphereGeometry.firstMaterial = material
        
        scene.rootNode.addChildNode(sphereNode)
    }
    
    private func applyMaterial(to node: SCNNode, mode: DisplayMode) {
        guard let geometry = node.geometry else { return }
        
        switch mode {
        case .wireframe:
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.clear
            material.emission.contents = UIColor.red
            material.fillMode = .lines
            material.isDoubleSided = true
            geometry.firstMaterial = material
            
        case .solid:
            let material = SCNMaterial()
            material.diffuse.contents = UIColor(red: 0.2, green: 0.6, blue: 1.0, alpha: 1.0)
            material.lightingModel = .physicallyBased
            geometry.firstMaterial = material
            
        case .points:
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.white
            material.emission.contents = UIColor.yellow
            material.fillMode = .lines
            material.isDoubleSided = true
            geometry.firstMaterial = material
            
        case .textured:
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.white
            material.normal.contents = UIColor.blue.withAlphaComponent(0.5)
            material.metalness.contents = 0.5
            material.roughness.contents = 0.5
            geometry.firstMaterial = material
        }
    }
    
    private func updateDisplayMode(_ newMode: DisplayMode) {
        // We don't need to do anything here since the scene will be recreated
        // when SwiftUI refreshes the view after the displayMode changes
        // The onChange handler already triggers a view refresh
    }
    
    private func addMeasurementsToScene(_ scene: SCNScene, for mesh: MDLMesh) {
        // Get the bounding box
        let boundingBox = mesh.boundingBox
        let minBounds = boundingBox.minBounds
        let maxBounds = boundingBox.maxBounds
        
        // Calculate dimensions
        let width = maxBounds.x - minBounds.x
        let height = maxBounds.y - minBounds.y
        let depth = maxBounds.z - minBounds.z
        
        // Add dimension labels
        addDimensionLabel(to: scene, text: "W: \(String(format: "%.2f", width))m", 
                         at: SCNVector3(minBounds.x + width/2, minBounds.y, minBounds.z))
        
        addDimensionLabel(to: scene, text: "H: \(String(format: "%.2f", height))m", 
                         at: SCNVector3(minBounds.x, minBounds.y + height/2, minBounds.z))
        
        addDimensionLabel(to: scene, text: "D: \(String(format: "%.2f", depth))m", 
                         at: SCNVector3(minBounds.x, minBounds.y, minBounds.z + depth/2))
    }
    
    private func addDimensionLabel(to scene: SCNScene, text: String, at position: SCNVector3) {
        let textGeometry = SCNText(string: text, extrusionDepth: 0.01)
        textGeometry.font = UIFont.systemFont(ofSize: 0.05)
        
        let textNode = SCNNode(geometry: textGeometry)
        textNode.position = position
        textNode.scale = SCNVector3(0.01, 0.01, 0.01)
        
        // Always face the camera
        textNode.constraints = [SCNBillboardConstraint()]
        
        scene.rootNode.addChildNode(textNode)
    }
} 