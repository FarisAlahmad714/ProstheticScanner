import SwiftUI
import UniformTypeIdentifiers
import SceneKit
import simd

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
            case .obj: return .obj
            case .stl: return .stl
            case .ply: return .ply
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
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let fileData = try generateFileData()
                
                DispatchQueue.main.async {
                    isExporting = false
                    shareFile(data: fileData)
                    exportSuccess = true
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
        let scene = SCNScene()
        
        let vertices = meshData.vertices.map { SCNVector3($0.x, $0.y, $0.z) }
        let vertexSource = SCNGeometrySource(vertices: vertices)
        
        let normals: [SCNVector3]
        if meshData.normals.count == meshData.vertices.count {
            normals = meshData.normals.map { SCNVector3($0.x, $0.y, $0.z) }
        } else {
            normals = Array(repeating: SCNVector3(0, 1, 0), count: meshData.vertices.count)
        }
        let normalSource = SCNGeometrySource(normals: normals)
        
        let indices = meshData.triangles
        let data = Data(bytes: indices, count: indices.count * MemoryLayout<UInt32>.size)
        let element = SCNGeometryElement(
            data: data,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )
        
        let geometry = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.blue
        geometry.firstMaterial = material
        
        let node = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(node)
        
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("export_temp.\(selectedFormat.fileExtension)")
        
        switch selectedFormat {
        case .obj:
            try scene.write(to: url, options: nil, delegate: nil, progressHandler: nil)
            
        case .stl:
            let tempDaeURL = FileManager.default.temporaryDirectory.appendingPathComponent("temp.dae")
            try scene.write(to: tempDaeURL, options: nil, delegate: nil, progressHandler: nil)
            
            let stlData = try convertToSTL(from: tempDaeURL)
            try stlData.write(to: url)
            try? FileManager.default.removeItem(at: tempDaeURL)
            
        case .ply:
            let tempDaeURL = FileManager.default.temporaryDirectory.appendingPathComponent("temp.dae")
            try scene.write(to: tempDaeURL, options: nil, delegate: nil, progressHandler: nil)
            
            let plyData = try convertToPLY(from: tempDaeURL)
            try plyData.write(to: url)
            try? FileManager.default.removeItem(at: tempDaeURL)
        }
        
        let fileData = try Data(contentsOf: url)
        try? FileManager.default.removeItem(at: url)
        
        return fileData
    }
    
    // Convert to STL binary format
    private func convertToSTL(from daeURL: URL) throws -> Data {
        var stlData = Data()
        
        // STL Binary Header (80 bytes) + triangle count (4 bytes)
        let header = "ProstheticScanner STL Export" + String(repeating: " ", count: 80 - 25)
        stlData.append(Data(header.utf8))
        
        let triangleCount = UInt32(meshData.triangles.count / 3)
        withUnsafeBytes(of: triangleCount.littleEndian) { stlData.append(contentsOf: $0) }
        
        for i in stride(from: 0, to: meshData.triangles.count, by: 3) {
            let v1 = meshData.vertices[Int(meshData.triangles[i])]
            let v2 = meshData.vertices[Int(meshData.triangles[i + 1])]
            let v3 = meshData.vertices[Int(meshData.triangles[i + 2])]
            
            // Calculate normal
            let normal = normalize(cross(v2 - v1, v3 - v1))
            
            // Append normal (3 floats, 4 bytes each)
            stlData.append(floatToLittleEndian(normal.x))
            stlData.append(floatToLittleEndian(normal.y))
            stlData.append(floatToLittleEndian(normal.z))
            
            // Append vertex 1
            stlData.append(floatToLittleEndian(v1.x))
            stlData.append(floatToLittleEndian(v1.y))
            stlData.append(floatToLittleEndian(v1.z))
            
            // Append vertex 2
            stlData.append(floatToLittleEndian(v2.x))
            stlData.append(floatToLittleEndian(v2.y))
            stlData.append(floatToLittleEndian(v2.z))
            
            // Append vertex 3
            stlData.append(floatToLittleEndian(v3.x))
            stlData.append(floatToLittleEndian(v3.y))
            stlData.append(floatToLittleEndian(v3.z))
            
            // Append attribute byte count (2 bytes, typically 0)
            withUnsafeBytes(of: UInt16(0).littleEndian) { stlData.append(contentsOf: $0) }
        }
        
        return stlData
    }
    
    // Helper function to convert Float to little-endian Data
    private func floatToLittleEndian(_ value: Float) -> Data {
        var littleEndianValue = value.bitPattern.littleEndian
        return Data(bytes: &littleEndianValue, count: MemoryLayout<UInt32>.size)
    }
    
    // Convert to PLY ASCII format
    private func convertToPLY(from daeURL: URL) throws -> Data {
        var plyString = "ply\nformat ascii 1.0\n"
        plyString += "element vertex \(meshData.vertices.count)\n"
        plyString += "property float x\nproperty float y\nproperty float z\n"
        plyString += "element face \(meshData.triangles.count / 3)\n"
        plyString += "property list uchar int vertex_indices\nend_header\n"
        
        for vertex in meshData.vertices {
            plyString += "\(vertex.x) \(vertex.y) \(vertex.z)\n"
        }
        
        for i in stride(from: 0, to: meshData.triangles.count, by: 3) {
            let idx1 = meshData.triangles[i]
            let idx2 = meshData.triangles[i + 1]
            let idx3 = meshData.triangles[i + 2]
            plyString += "3 \(idx1) \(idx2) \(idx3)\n"
        }
        
        guard let data = plyString.data(using: .utf8) else {
            throw NSError(domain: "ExportError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to encode PLY data"])
        }
        return data
    }
    
    // Share the file using system share sheet
    private func shareFile(data: Data) {
        let fileName = "prosthetic_scan.\(selectedFormat.fileExtension)"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        
        try? data.write(to: url)
        
        let activityViewController = UIActivityViewController(
            activityItems: [url],
            applicationActivities: nil
        )
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootViewController = windowScene.windows.first?.rootViewController {
            rootViewController.present(activityViewController, animated: true, completion: nil)
        }
    }
}

// UTType extensions for 3D file formats
extension UTType {
    static var obj: UTType {
        UTType(filenameExtension: "obj") ?? .plainText
    }
    
    static var stl: UTType {
        UTType(filenameExtension: "stl") ?? .plainText
    }
    
    static var ply: UTType {
        UTType(filenameExtension: "ply") ?? .plainText
    }
}
