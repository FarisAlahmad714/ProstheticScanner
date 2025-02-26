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