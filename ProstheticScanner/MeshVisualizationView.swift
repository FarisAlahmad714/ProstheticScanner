import SwiftUI
import MetalKit
import SceneKit

struct MeshVisualizationView: View {
    let meshData: MeshData
    let onExport: () -> Void
    let onNewScan: () -> Void
    
    // View state
    @State private var isShowingControls = true
    @State private var rotation: Float = 0
    @State private var scale: Float = 1.0
    @State private var showExportSheet = false
    @State private var showingError = false
    @State private var errorMessage = ""
    
    // Camera controls
    @State private var cameraDistance: Float = 2.0
    @State private var cameraAngleX: Float = 0.0
    @State private var cameraAngleY: Float = 0.0
    
    // Toggle for display mode
    @State private var displayMode: DisplayMode = .shaded
    
    enum DisplayMode: String, CaseIterable, Identifiable {
        case shaded = "Shaded"
        case wireframe = "Wireframe"
        case points = "Points"
        
        var id: String { self.rawValue }
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // SceneKit view for mesh rendering
                SceneKitMeshView(
                    meshData: meshData,
                    rotation: $rotation,
                    scale: $scale,
                    cameraDistance: $cameraDistance,
                    cameraAngleX: $cameraAngleX,
                    cameraAngleY: $cameraAngleY,
                    displayMode: $displayMode
                )
                .edgesIgnoringSafeArea(.all)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            // Update camera angles based on drag
                            cameraAngleX += Float(value.translation.height) * 0.01
                            cameraAngleY += Float(value.translation.width) * 0.01
                        }
                )
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            scale = Float(value) * scale
                        }
                        .onEnded { value in
                            scale = min(max(0.5, scale), 5.0) // Limit scale range
                        }
                )
                
                if isShowingControls {
                    // Controls overlay
                    VStack {
                        // Top bar with stats and controls
                        HStack {
                            // Mesh stats
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Vertices: \(meshData.vertices.count)")
                                Text("Triangles: \(meshData.triangles.count / 3)")
                            }
                            .font(.caption)
                            .padding(8)
                            .background(Color.black.opacity(0.7))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                            
                            Spacer()
                            
                            // View controls
                            HStack(spacing: 12) {
                                Button(action: resetView) {
                                    Image(systemName: "arrow.counterclockwise")
                                        .foregroundColor(.white)
                                }
                                
                                Button(action: { scale *= 1.2 }) {
                                    Image(systemName: "plus.magnifyingglass")
                                        .foregroundColor(.white)
                                }
                                
                                Button(action: { scale *= 0.8 }) {
                                    Image(systemName: "minus.magnifyingglass")
                                        .foregroundColor(.white)
                                }
                                
                                Menu {
                                    Picker("Display Mode", selection: $displayMode) {
                                        ForEach(DisplayMode.allCases) { mode in
                                            Text(mode.rawValue).tag(mode)
                                        }
                                    }
                                } label: {
                                    Image(systemName: "eye")
                                        .foregroundColor(.white)
                                }
                            }
                            .padding(8)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(8)
                        }
                        .padding()
                        
                        Spacer()
                        
                        // Bottom action buttons
                        HStack {
                            Button(action: onNewScan) {
                                Label("New Scan", systemImage: "camera")
                                    .padding()
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            
                            Spacer()
                            
                            Button(action: {
                                showExportSheet = true
                            }) {
                                Label("Export", systemImage: "square.and.arrow.up")
                                    .padding()
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                        .padding()
                    }
                }
            }
            .onTapGesture {
                withAnimation {
                    isShowingControls.toggle()
                }
            }
            .sheet(isPresented: $showExportSheet) {
                ExportOptionsSheet(onExport: onExport)
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func resetView() {
        withAnimation(.spring()) {
            rotation = 0
            scale = 1.0
            cameraDistance = 2.0
            cameraAngleX = 0.0
            cameraAngleY = 0.0
        }
    }
}

// SceneKit renderer for the mesh
struct SceneKitMeshView: UIViewRepresentable {
    let meshData: MeshData
    @Binding var rotation: Float
    @Binding var scale: Float
    @Binding var cameraDistance: Float
    @Binding var cameraAngleX: Float
    @Binding var cameraAngleY: Float
    @Binding var displayMode: MeshVisualizationView.DisplayMode
    
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.backgroundColor = .black
        scnView.autoenablesDefaultLighting = true
        scnView.allowsCameraControl = false
        scnView.scene = createScene()
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        // Update camera position
        if let cameraNode = uiView.scene?.rootNode.childNode(withName: "camera", recursively: true) {
            updateCamera(cameraNode)
        }
        
        // Update mesh material based on display mode
        if let meshNode = uiView.scene?.rootNode.childNode(withName: "mesh", recursively: true),
           let geometry = meshNode.geometry {
            updateMaterialForDisplayMode(geometry)
        }
    }
    
    private func createScene() -> SCNScene {
        let scene = SCNScene()
        
        // Create mesh geometry from mesh data
        let vertices = meshData.vertices
        let normals = meshData.normals
        let indices = meshData.triangles
        
        let verticesSource = SCNGeometrySource(vertices: vertices.map { SCNVector3($0.x, $0.y, $0.z) })
        
        let normalsSource: SCNGeometrySource
        if normals.count == vertices.count {
            normalsSource = SCNGeometrySource(normals: normals.map { SCNVector3($0.x, $0.y, $0.z) })
        } else {
            // Generate default normals if they don't exist
            normalsSource = SCNGeometrySource(normals: [SCNVector3](repeating: SCNVector3(0, 1, 0), count: vertices.count))
        }
        
        let indicesData = Data(bytes: indices, count: indices.count * MemoryLayout<UInt32>.stride)
        let indexElement = SCNGeometryElement(data: indicesData,
                                            primitiveType: .triangles,
                                            primitiveCount: indices.count / 3,
                                            bytesPerIndex: MemoryLayout<UInt32>.stride)
        
        let geometry = SCNGeometry(sources: [verticesSource, normalsSource], elements: [indexElement])
        
        // Apply default material
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.blue
        material.specular.contents = UIColor.white
        material.shininess = 0.5
        geometry.firstMaterial = material
        
        // Create node for the mesh
        let meshNode = SCNNode(geometry: geometry)
        meshNode.name = "mesh"
        
        // Center the mesh based on its bounds
        let (min, max) = geometry.boundingBox
        let center = SCNVector3((min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2)
        meshNode.position = SCNVector3(-center.x, -center.y, -center.z)
        
        // Add mesh to the scene
        scene.rootNode.addChildNode(meshNode)
        
        // Create and position the camera
        let cameraNode = SCNNode()
        cameraNode.name = "camera"
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 0, cameraDistance)
        
        // Add camera to the scene
        scene.rootNode.addChildNode(cameraNode)
        
        return scene
    }
    
    private func updateCamera(_ cameraNode: SCNNode) {
        // Calculate camera position based on distance and angles
        let x = cameraDistance * sin(cameraAngleY) * cos(cameraAngleX)
        let y = cameraDistance * sin(cameraAngleX)
        let z = cameraDistance * cos(cameraAngleY) * cos(cameraAngleX)
        
        // Apply rotation and scale
        cameraNode.position = SCNVector3(x, y, z)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        
        // Update mesh node scale if needed
        if let meshNode = cameraNode.parent?.childNode(withName: "mesh", recursively: true) {
            meshNode.scale = SCNVector3(scale, scale, scale)
        }
    }
    
    private func updateMaterialForDisplayMode(_ geometry: SCNGeometry) {
        guard let material = geometry.firstMaterial else { return }
        
        switch displayMode {
        case .shaded:
            material.diffuse.contents = UIColor.blue
            material.specular.contents = UIColor.white
            material.lightingModel = .blinn
            material.fillMode = .fill
            
        case .wireframe:
            material.diffuse.contents = UIColor.white
            material.lightingModel = .constant
            material.fillMode = .lines
            
        case .points:
            material.diffuse.contents = UIColor.yellow
            material.lightingModel = .constant
            material.fillMode = .fill
            
            // Change element primitive type to points
            if let element = geometry.elements.first {
                element.primitiveType = .points
            }
        }
    }
}

// Export options sheet
struct ExportOptionsSheet: View {
    let onExport: () -> Void
    @Environment(\.dismiss) var dismiss
    @State private var selectedFormat: ExportFormat = .obj
    @State private var includeTextures: Bool = true
    @State private var optimizeMesh: Bool = true
    
    enum ExportFormat: String, CaseIterable, Identifiable {
        case obj = "OBJ File"
        case stl = "STL File"
        case ply = "PLY File"
        
        var id: String { self.rawValue }
        
        var fileExtension: String {
            switch self {
            case .obj: return "obj"
            case .stl: return "stl"
            case .ply: return "ply"
            }
        }
        
        var icon: String {
            switch self {
            case .obj: return "doc.text"
            case .stl: return "cube"
            case .ply: return "square.3.stack.3d"
            }
        }
    }
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Export Format")) {
                    ForEach(ExportFormat.allCases) { format in
                        Button(action: {
                            selectedFormat = format
                            onExport()
                            dismiss()
                        }) {
                            Label {
                                Text(format.rawValue)
                            } icon: {
                                Image(systemName: format.icon)
                            }
                        }
                    }
                }
                
                Section(header: Text("Options")) {
                    Toggle("Include Textures", isOn: $includeTextures)
                    Toggle("Optimize Mesh", isOn: $optimizeMesh)
                    
                    HStack {
                        Text("Estimated File Size")
                        Spacer()
                        Text("1.2 MB") // Placeholder - would be calculated in real app
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Export Mesh")
            .navigationBarItems(trailing: Button("Cancel") {
                dismiss()
            })
        }
    }
}
