<<<<<<< HEAD
//
//  MeshVisualizationView.swift
//  ProstheticScanner
//
//  Created by Faris Alahmad on 11/9/24.
//


import SwiftUI
import MetalKit
import SceneKit

struct MeshVisualizationView: View {
    let meshData: MeshData?
    let onExport: () -> Void
    let onNewScan: () -> Void
    
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
    
    var body: some View {
        ZStack {
            // Metal view for mesh rendering
            MetalMeshRenderer(
                meshData: meshData,
                rotation: $rotation,
                scale: $scale,
                cameraDistance: $cameraDistance,
                cameraAngleX: $cameraAngleX,
                cameraAngleY: $cameraAngleY
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
                        scale = Float(value) * scale
                    }
            )
            
            if isShowingControls {
                // Controls overlay
                VStack {
                    // Top bar with stats
                    HStack {
                        VStack(alignment: .leading) {
                            if let meshData = meshData {
                                Text("Vertices: \(meshData.vertices.count)")
                                Text("Triangles: \(meshData.triangles.count / 3)")
                            }
                        }
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(10)
                        
                        Spacer()
                        
                        // View controls
                        HStack(spacing: 20) {
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
                        }
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(10)
                    }
                    .padding()
                    
                    Spacer()
                    
                    // Bottom controls
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

// Metal renderer for the mesh
struct MetalMeshRenderer: UIViewRepresentable {
    let meshData: MeshData?
    @Binding var rotation: Float
    @Binding var scale: Float
    @Binding var cameraDistance: Float
    @Binding var cameraAngleX: Float
    @Binding var cameraAngleY: Float
    
    class Coordinator {
        var metalView: MTKView?
        var renderer: MeshRenderer?
        
        init(meshData: MeshData?) {
            guard let device = MTLCreateSystemDefaultDevice(),
                  let meshData = meshData else { return }
            
            metalView = MTKView(frame: .zero, device: device)
            renderer = MeshRenderer(device: device, meshData: meshData)
            metalView?.delegate = renderer
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(meshData: meshData)
    }
    
    func makeUIView(context: Context) -> MTKView {
        context.coordinator.metalView ?? MTKView()
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.renderer?.update(
            rotation: rotation,
            scale: scale,
            cameraDistance: cameraDistance,
            cameraAngleX: cameraAngleX,
            cameraAngleY: cameraAngleY
        )
    }
}

// Export options sheet
struct ExportOptionsSheet: View {
    let onExport: () -> Void
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Export Format")) {
                    Button(action: {
                        onExport()
                        dismiss()
                    }) {
                        Label("OBJ File", systemImage: "doc")
                    }
                    
                    Button(action: {
                        onExport()
                        dismiss()
                    }) {
                        Label("STL File", systemImage: "doc")
                    }
                }
                
                Section(header: Text("Options")) {
                    Toggle("Include Textures", isOn: .constant(true))
                    Toggle("Optimize Mesh", isOn: .constant(true))
                }
            }
            .navigationTitle("Export Mesh")
            .navigationBarItems(trailing: Button("Cancel") {
                dismiss()
            })
        }
    }
}

// Mesh renderer class (Metal implementation)
class MeshRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineState: MTLRenderPipelineState?
    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?
    
    init(device: MTLDevice, meshData: MeshData) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        super.init()
        setupPipeline()
        setupBuffers(with: meshData)
    }
    
    private func setupPipeline() {
        // Implement Metal pipeline setup
    }
    
    private func setupBuffers(with meshData: MeshData) {
        // Implement buffer setup with mesh data
    }
    
    func update(rotation: Float, scale: Float, cameraDistance: Float, cameraAngleX: Float, cameraAngleY: Float) {
        // Update rendering parameters
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize
    }
    
    func draw(in view: MTKView) {
        // Implement mesh drawing
    }
}
=======
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
                SimpleMeshView(
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
                ExportOptionsSheet(meshData: meshData, onExport: onExport)
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

// Simplified SceneKit renderer for the mesh
struct SimpleMeshView: UIViewRepresentable {
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
        
        // Update mesh node
        if let meshNode = uiView.scene?.rootNode.childNode(withName: "mesh", recursively: true) {
            // Update scale
            meshNode.scale = SCNVector3(scale, scale, scale)
            
            // Update material based on display mode
            if let geometry = meshNode.geometry,
               let material = geometry.firstMaterial {
                updateMaterial(material)
            }
        }
    }
    
    private func createScene() -> SCNScene {
        let scene = SCNScene()
        
        // Create geometry sources
        let vertices = meshData.vertices.map { SCNVector3($0.x, $0.y, $0.z) }
        let vertexSource = SCNGeometrySource(vertices: vertices)
        
        // Use normals if available, or create default ones
        let normals: [SCNVector3]
        if meshData.normals.count == meshData.vertices.count {
            normals = meshData.normals.map { SCNVector3($0.x, $0.y, $0.z) }
        } else {
            normals = Array(repeating: SCNVector3(0, 1, 0), count: meshData.vertices.count)
        }
        let normalSource = SCNGeometrySource(normals: normals)
        
        // Create geometry element
        let indices = meshData.triangles
        let data = Data(bytes: indices, count: indices.count * MemoryLayout<UInt32>.size)
        let element = SCNGeometryElement(
            data: data,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )
        
        // Create geometry
        let geometry = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])
        
        // Create material
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.blue
        material.specular.contents = UIColor.white
        material.shininess = 0.5
        geometry.firstMaterial = material
        
        // Create mesh node
        let meshNode = SCNNode(geometry: geometry)
        meshNode.name = "mesh"
        
        // Center the mesh
        let boundingBox = calculateBoundingBox(vertices: meshData.vertices)
        let center = SCNVector3(
            (boundingBox.min.x + boundingBox.max.x) / 2,
            (boundingBox.min.y + boundingBox.max.y) / 2,
            (boundingBox.min.z + boundingBox.max.z) / 2
        )
        meshNode.position = SCNVector3(-center.x, -center.y, -center.z)
        
        scene.rootNode.addChildNode(meshNode)
        
        // Create camera
        let camera = SCNCamera()
        camera.zNear = 0.1
        camera.zFar = 100
        
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, cameraDistance)
        cameraNode.name = "camera"
        
        scene.rootNode.addChildNode(cameraNode)
        
        return scene
    }
    
    private func calculateBoundingBox(vertices: [SIMD3<Float>]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        var minBounds = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxBounds = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
        
        for vertex in vertices {
            minBounds = min(minBounds, vertex)
            maxBounds = max(maxBounds, vertex)
        }
        
        return (minBounds, maxBounds)
    }
    
    private func updateCamera(_ cameraNode: SCNNode) {
        // Calculate camera position based on spherical coordinates
        let x = cameraDistance * sin(cameraAngleY) * cos(cameraAngleX)
        let y = cameraDistance * sin(cameraAngleX)
        let z = cameraDistance * cos(cameraAngleY) * cos(cameraAngleX)
        
        // Update camera position
        cameraNode.position = SCNVector3(x, y, z)
        
        // Make camera look at the center
        cameraNode.look(at: SCNVector3(0, 0, 0))
    }
    
    private func updateMaterial(_ material: SCNMaterial) {
        switch displayMode {
        case .shaded:
            material.diffuse.contents = UIColor.blue
            material.lightingModel = .blinn
            
        case .wireframe:
            material.diffuse.contents = UIColor.white
            material.lightingModel = .constant
            material.fillMode = .lines
            
        case .points:
            material.diffuse.contents = UIColor.yellow
            material.lightingModel = .constant
        }
    }
}
>>>>>>> 57d48373c35c6526189f0514e3f11772729acb9c
