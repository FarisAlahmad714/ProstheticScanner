import SwiftUI
import UniformTypeIdentifiers
import SceneKit

// Export options sheet with functioning download
struct ExportOptionsSheet: View {
    let meshData: MeshData
    let onExport: () -> Void
    @Environment(\.dismiss) var dismiss
    @State private var selectedFormat: ExportFormat = .obj
    @State private var includeTextures: Bool = true
    @State private var optimizeMesh: Bool = true
    @State private var isExporting = false
    @State private var exportSuccess = false
    @State private var exportError: String? = nil
    
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
        
        var utType: UTType {
            switch self {
            case .obj: return UTType.obj
            case .stl: return UTType.stl
            case .ply: return UTType.ply
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
                        }) {
                            HStack {
                                Label {
                                    Text(format.rawValue)
                                } icon: {
                                    Image(systemName: format.icon)
                                }
                                
                                Spacer()
                                
                                if selectedFormat == format {
                                    Image(systemName: "checkmark")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                        .foregroundColor(.primary)
                    }
                }
                
                Section(header: Text("Options")) {
                    Toggle("Include Textures", isOn: $includeTextures)
                    Toggle("Optimize Mesh", isOn: $optimizeMesh)
                    
                    HStack {
                        Text("Estimated File Size")
                        Spacer()
                        Text(estimatedFileSize())
                            .foregroundColor(.secondary)
                    }
                }
                
                Section {
                    Button(action: {
                        exportMesh()
                    }) {
                        HStack {
                            Spacer()
                            
                            if isExporting {
                                ProgressView()
                                    .padding(.trailing, 10)
                            }
                            
                            Text("Download \(selectedFormat.fileExtension.uppercased())")
                                .bold()
                            
                            Spacer()
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(isExporting)
                }
                .listRowBackground(Color.clear)
                
                if exportSuccess {
                    Section {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            Text("Export successful!")
                        }
                    }
                }
                
                if let error = exportError {
                    Section {
                        HStack {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                            Text(error)
                                .foregroundColor(.red)
                        }
                    }
                }
            }
            .navigationTitle("Export Mesh")
            .navigationBarItems(trailing: Button("Cancel") {
                dismiss()
            })
        }
    }
    
    // Calculate estimated file size based on mesh complexity
    private func estimatedFileSize() -> String {
        let vertexCount = meshData.vertices.count
        let triangleCount = meshData.triangles.count / 3
        
        // Rough estimation based on format
        var bytesPerVertex: Int
        var bytesPerTriangle: Int
        
        switch selectedFormat {
        case .obj:
            bytesPerVertex = 30  // ~30 bytes per vertex in OBJ (with normals)
            bytesPerTriangle = 15  // ~15 bytes per triangle reference
        case .stl:
            bytesPerVertex = 0   // STL doesn't store vertices separately
            bytesPerTriangle = 50  // ~50 bytes per triangle in STL
        case .ply:
            bytesPerVertex = 25  // ~25 bytes per vertex in PLY
            bytesPerTriangle = 10  // ~10 bytes per triangle reference
        }
        
        let estimatedBytes = vertexCount * bytesPerVertex + triangleCount * bytesPerTriangle
        
        // Convert to appropriate unit
        if estimatedBytes < 1024 {
            return "\(estimatedBytes) B"
        } else if estimatedBytes < 1024 * 1024 {
            let kb = Double(estimatedBytes) / 1024.0
            return String(format: "%.1f KB", kb)
        } else {
            let mb = Double(estimatedBytes) / (1024.0 * 1024.0)
            return String(format: "%.1f MB", mb)
        }
    }
    
    // Export the mesh to the selected format
    private func exportMesh() {
        isExporting = true
        exportSuccess = false
        exportError = nil
        
        // Create a background task to generate the file
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Generate the file data
                let fileData = try generateFileData()
                
                // Update UI on main thread
                DispatchQueue.main.async {
                    isExporting = false
                    
                    // Present the share sheet to save the file
                    shareFile(data: fileData)
                    
                    // Mark as success
                    exportSuccess = true
                    
                    // Call original export handler
                    onExport()
                }
            } catch {
                DispatchQueue.main.async {
                    isExporting = false
                    exportError = "Export failed: \(error.localizedDescription)"
                }
            }
        }
    }
    
    // Generate file data based on selected format
    private func generateFileData() throws -> Data {
        // Create a temporary scene to export
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
        
        // Create geometry and node
        let geometry = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.blue
        geometry.firstMaterial = material
        
        let node = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(node)
        
        // Generate the file data
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("export_temp.\(selectedFormat.fileExtension)")
        
        // Export based on format
        switch selectedFormat {
        case .obj:
            try scene.write(to: url, options: nil, delegate: nil, progressHandler: nil)
        case .stl, .ply:
            // Use SCNScene's export capability
            try scene.export(to: url, options: [SCNSceneExportDestinationURL: url])
        }
        
        // Read the data from the URL
        let fileData = try Data(contentsOf: url)
        
        // Clean up the temporary file
        try? FileManager.default.removeItem(at: url)
        
        return fileData
    }
    
    // Share the file using system share sheet
    private func shareFile(data: Data) {
        let fileName = "prosthetic_scan.\(selectedFormat.fileExtension)"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        
        // Write data to the temporary file
        try? data.write(to: url)
        
        // Create a UIActivityViewController to share the file
        let activityViewController = UIActivityViewController(
            activityItems: [url],
            applicationActivities: nil
        )
        
        // Present the view controller
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootViewController = windowScene.windows.first?.rootViewController {
            rootViewController.present(activityViewController, animated: true, completion: nil)
        }
    }
}

// UTType extensions for 3D file formats
extension UTType {
    static var obj: UTType {
        UTType(filenameExtension: "obj")!
    }
    
    static var stl: UTType {
        UTType(filenameExtension: "stl")!
    }
    
    static var ply: UTType {
        UTType(filenameExtension: "ply")!
    }
}
