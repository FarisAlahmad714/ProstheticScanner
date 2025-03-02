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
    
    private func estimatedFileSize() -> String {
        let vertexCount = meshData.vertices.count
        let triangleCount = meshData.triangles.count / 3
        
        var bytesPerVertex: Int
        var bytesPerTriangle: Int
        
        switch selectedFormat {
        case .obj:
            bytesPerVertex = 30
            bytesPerTriangle = 15
        case .stl:
            bytesPerVertex = 0
            bytesPerTriangle = 50
        case .ply:
            bytesPerVertex = 25
            bytesPerTriangle = 10
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
    
    private func exportMesh() {
        isExporting = true
        exportSuccess = false
        exportError = nil
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let fileData = try generateFileData()
                
                DispatchQueue.main.async {
                    self.shareFile(data: fileData)
                }
            } catch {
                DispatchQueue.main.async {
                    self.isExporting = false
                    self.exportError = "Export failed: \(error.localizedDescription)"
                }
            }
        }
    }
    
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
        
        let tempDaeURL = FileManager.default.temporaryDirectory.appendingPathComponent("temp.dae")
        try scene.write(to: tempDaeURL, options: nil, delegate: nil, progressHandler: nil)
        
        switch selectedFormat {
        case .obj:
            let objData = try convertToOBJ(from: tempDaeURL)
            try objData.write(to: url)
        case .stl:
            let stlData = try convertToSTL(from: tempDaeURL)
            try stlData.write(to: url)
        case .ply:
            let plyData = try convertToPLY(from: tempDaeURL)
            try plyData.write(to: url)
        }
        
        try? FileManager.default.removeItem(at: tempDaeURL)
        
        let fileData = try Data(contentsOf: url)
        try? FileManager.default.removeItem(at: url)
        
        return fileData
    }
    
    private func convertToOBJ(from daeURL: URL) throws -> Data {
        var objString = "# OBJ file generated by ProstheticScanner\n"
        
        for vertex in meshData.vertices {
            objString += "v \(vertex.x) \(vertex.y) \(vertex.z)\n"
        }
        
        for normal in meshData.normals {
            objString += "vn \(normal.x) \(normal.y) \(normal.z)\n"
        }
        
        for i in stride(from: 0, to: meshData.triangles.count, by: 3) {
            let idx1 = meshData.triangles[i] + 1
            let idx2 = meshData.triangles[i + 1] + 1
            let idx3 = meshData.triangles[i + 2] + 1
            objString += "f \(idx1)//\(idx1) \(idx2)//\(idx2) \(idx3)//\(idx3)\n"
        }
        
        guard let data = objString.data(using: .utf8) else {
            throw NSError(domain: "ExportError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to encode OBJ data"])
        }
        return data
    }
    
    private func convertToSTL(from daeURL: URL) throws -> Data {
        var stlData = Data()
        
        let header = "ProstheticScanner STL Export" + String(repeating: " ", count: 80 - 25)
        stlData.append(Data(header.utf8))
        
        let triangleCount = UInt32(meshData.triangles.count / 3)
        withUnsafeBytes(of: triangleCount.littleEndian) { stlData.append(contentsOf: $0) }
        
        for i in stride(from: 0, to: meshData.triangles.count, by: 3) {
            let v1 = meshData.vertices[Int(meshData.triangles[i])]
            let v2 = meshData.vertices[Int(meshData.triangles[i + 1])]
            let v3 = meshData.vertices[Int(meshData.triangles[i + 2])]
            
            let normal = normalize(cross(v2 - v1, v3 - v1))
            
            stlData.append(floatToLittleEndian(normal.x))
            stlData.append(floatToLittleEndian(normal.y))
            stlData.append(floatToLittleEndian(normal.z))
            
            stlData.append(floatToLittleEndian(v1.x))
            stlData.append(floatToLittleEndian(v1.y))
            stlData.append(floatToLittleEndian(v1.z))
            
            stlData.append(floatToLittleEndian(v2.x))
            stlData.append(floatToLittleEndian(v2.y))
            stlData.append(floatToLittleEndian(v2.z))
            
            stlData.append(floatToLittleEndian(v3.x))
            stlData.append(floatToLittleEndian(v3.y))
            stlData.append(floatToLittleEndian(v3.z))
            
            withUnsafeBytes(of: UInt16(0).littleEndian) { stlData.append(contentsOf: $0) }
        }
        
        return stlData
    }
    
    private func floatToLittleEndian(_ value: Float) -> Data {
        var littleEndianValue = value.bitPattern.littleEndian
        return Data(bytes: &littleEndianValue, count: MemoryLayout<UInt32>.size)
    }
    
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
    
    private func shareFile(data: Data) {
        let fileName = "prosthetic_scan.\(selectedFormat.fileExtension)"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        
        do {
            try data.write(to: url, options: .atomic)
            print("File written to: \(url.path)")
        } catch {
            print("Failed to write file: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.exportError = "Failed to save file: \(error.localizedDescription)"
                self.exportSuccess = false
            }
            return
        }
        
        let activityViewController = UIActivityViewController(
            activityItems: [url],
            applicationActivities: nil
        )
        
        activityViewController.completionWithItemsHandler = { activityType, completed, returnedItems, error in
            if let error = error {
                print("Share sheet error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.exportError = "Share sheet error: \(error.localizedDescription)"
                    self.exportSuccess = false
                }
                return
            }
            
            if completed {
                print("Share sheet completed with activity: \(String(describing: activityType))")
                DispatchQueue.main.async {
                    self.exportSuccess = true
                }
            } else {
                print("Share sheet dismissed without completion")
                DispatchQueue.main.async {
                    self.exportSuccess = false
                    self.exportError = "Export cancelled"
                }
            }
            
            try? FileManager.default.removeItem(at: url)
        }
        
        DispatchQueue.main.async {
            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
               let rootViewController = windowScene.windows.first?.rootViewController {
                print("Presenting share sheet from root view controller")
                if rootViewController.presentedViewController == nil {
                    rootViewController.present(activityViewController, animated: true, completion: nil)
                } else {
                    print("Root view controller is already presenting another view controller")
                    rootViewController.dismiss(animated: true) {
                        rootViewController.present(activityViewController, animated: true, completion: nil)
                    }
                }
            } else {
                print("Failed to find root view controller for presentation")
                self.saveToDocumentsDirectory(data: data, fileName: fileName)
            }
        }
    }
    
    private func saveToDocumentsDirectory(data: Data, fileName: String) {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent(fileName)
        
        do {
            try data.write(to: fileURL, options: .atomic)
            print("File saved to Documents directory: \(fileURL.path)")
            DispatchQueue.main.async {
                self.exportError = "Share sheet unavailable. File saved to Documents directory: \(fileURL.path)"
                self.exportSuccess = false
            }
        } catch {
            print("Failed to save file to Documents directory: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.exportError = "Failed to save file: \(error.localizedDescription)"
                self.exportSuccess = false
            }
        }
    }
}

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
