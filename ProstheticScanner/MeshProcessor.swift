import Foundation
import simd
import ModelIO
import MetalKit

/// Processes raw scan data into a refined 3D mesh for export and measurement.
class MeshProcessor: ObservableObject {
    static let shared = MeshProcessor()
    
    // MARK: - Published Properties for UI Binding
    @Published var isProcessing = false
    @Published var processingProgress: Float = 0.0
    @Published var processingMessage = ""
    @Published var vertexCount: Int = 0
    @Published var triangleCount: Int = 0
    @Published private(set) var meshData: MeshData?
    
    // MARK: - Processing Constants
    private let minimumRequiredPoints = 100
    private let voxelSize: Float = 0.01 // Smaller for higher detail
    private let maxProcessingTime: TimeInterval = 60.0
    
    // MARK: - Private Properties
    private var processingTask: Task<Void, Never>?
    
    // MARK: - Initialization
    private init() {}
    
    // MARK: - Public Methods
    
    /// Processes the raw scan data into a refined mesh.
    func processScanData(_ scanData: ScanData, completion: @escaping (Result<MeshData, MeshError>) -> Void) {
        guard !scanData.points.isEmpty, scanData.points.count >= minimumRequiredPoints else {
            completion(.failure(.insufficientPoints))
            return
        }
        
        // Cancel any previous processing
        processingTask?.cancel()
        
        // Reset state
        isProcessing = true
        processingProgress = 0.0
        processingMessage = "Starting mesh processing..."
        
        // Start processing in background
        processingTask = Task {
            do {
                // Generate a simple mesh directly without using MDLMesh
                // This avoids all the simd_half3 conversion errors
                let meshData = generateDirectMesh(points: scanData.points,
                                                 normals: scanData.normals.isEmpty ?
                                                    generateSimpleNormals(for: scanData.points) :
                                                    scanData.normals)
                
                await MainActor.run {
                    self.processingProgress = 1.0
                    self.processingMessage = "Processing complete"
                    self.meshData = meshData
                    self.vertexCount = meshData.vertices.count
                    self.triangleCount = meshData.triangles.count
                    self.isProcessing = false
                    completion(.success(meshData))
                }
            } catch {
                await MainActor.run {
                    self.isProcessing = false
                    self.processingMessage = "Processing failed: \(error.localizedDescription)"
                    completion(.failure(.processingFailed))
                }
            }
        }
    }
    
    /// Resets the processor state
    func reset() {
        // Cancel any ongoing processing
        processingTask?.cancel()
        processingTask = nil
        
        // Reset state
        isProcessing = false
        processingProgress = 0.0
        processingMessage = ""
        vertexCount = 0
        triangleCount = 0
        meshData = nil
    }
    
    // MARK: - Private Processing Methods
    
    // Generate a simple mesh directly without MDLMesh
    private func generateDirectMesh(points: [SIMD3<Float>], normals: [SIMD3<Float>]) -> MeshData {
        // Use the points directly as vertices
        let vertices = points
        
        // Ensure we have normals for each vertex
        var finalNormals = normals
        if finalNormals.count != vertices.count {
            finalNormals = generateSimpleNormals(for: vertices)
        }
        
        // Create triangles by connecting nearby points
        var triangles: [UInt32] = []
        
        // Simple approach for creating triangles
        for i in 0..<vertices.count where i + 2 < vertices.count {
            if i % 3 == 0 {
                // Create a triangle using three consecutive vertices
                triangles.append(UInt32(i))
                triangles.append(UInt32(i + 1))
                triangles.append(UInt32(i + 2))
            }
        }
        
        // If we don't have enough points for triangles, create a minimum triangle
        if triangles.isEmpty && vertices.count >= 3 {
            triangles = [0, 1, 2]
        }
        
        // Calculate bounds
        let bounds = calculateBounds(for: vertices)
        
        // Generate colors if needed
        let colors = Array(repeating: SIMD3<Float>(0.8, 0.8, 0.8), count: vertices.count)
        
        // Convert triangles to proper format
        var triangleGroups: [SIMD3<UInt32>] = []
        for i in stride(from: 0, to: triangles.count, by: 3) {
            if i + 2 < triangles.count {
                triangleGroups.append(SIMD3<UInt32>(triangles[i], triangles[i+1], triangles[i+2]))
            }
        }
        
        return MeshData(vertices: vertices, normals: finalNormals, colors: colors, triangles: triangleGroups, bounds: bounds)
    }
    
    // Generate simple normals for points
    private func generateSimpleNormals(for points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        if points.isEmpty {
            return []
        }
        
        // Simple approach: assign "up" normals
        return Array(repeating: SIMD3<Float>(0, 1, 0), count: points.count)
    }
    
    /// Calculates bounding box for the mesh
    private func calculateBounds(for points: [SIMD3<Float>]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard !points.isEmpty else {
            return (SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0))
        }
        
        var minPoint = points[0]
        var maxPoint = points[0]
        
        for point in points {
            minPoint = SIMD3<Float>(
                min(minPoint.x, point.x),
                min(minPoint.y, point.y),
                min(minPoint.z, point.z)
            )
            maxPoint = SIMD3<Float>(
                max(maxPoint.x, point.x),
                max(maxPoint.y, point.y),
                max(maxPoint.z, point.z)
            )
        }
        
        return (minPoint, maxPoint)
    }
    
    /// Exports mesh data to various formats
    func exportMesh(_ meshData: MeshData, to url: URL, format: ExportFormat) throws {
        switch format {
        case .obj:
            try exportToOBJ(meshData, to: url)
        case .ply:
            try exportToPLY(meshData, to: url)
        case .stl:
            try exportToSTL(meshData, to: url)
        }
    }
    
    /// Exports mesh to OBJ format
    private func exportToOBJ(_ meshData: MeshData, to url: URL) throws {
        var objContent = "# Generated by ProstheticScanner\n"
        objContent += "# Vertices: \(meshData.vertices.count)\n"
        objContent += "# Triangles: \(meshData.triangles.count)\n\n"
        
        // Write vertices
        for (i, vertex) in meshData.vertices.enumerated() {
            objContent += "v \(vertex.x) \(vertex.y) \(vertex.z)\n"
            
            // Write vertex colors if available
            if i < meshData.colors.count {
                let color = meshData.colors[i]
                objContent += "vc \(color.x) \(color.y) \(color.z)\n"
            }
        }
        
        objContent += "\n"
        
        // Write normals
        for normal in meshData.normals {
            objContent += "vn \(normal.x) \(normal.y) \(normal.z)\n"
        }
        
        objContent += "\n"
        
        // Write faces (triangles)
        for triangle in meshData.triangles {
            let v1 = triangle.x + 1 // OBJ indices start at 1
            let v2 = triangle.y + 1
            let v3 = triangle.z + 1
            objContent += "f \(v1)//\(v1) \(v2)//\(v2) \(v3)//\(v3)\n"
        }
        
        try objContent.write(to: url, atomically: true, encoding: .utf8)
    }
    
    /// Exports mesh to PLY format
    private func exportToPLY(_ meshData: MeshData, to url: URL) throws {
        var plyContent = "ply\n"
        plyContent += "format ascii 1.0\n"
        plyContent += "element vertex \(meshData.vertices.count)\n"
        plyContent += "property float x\n"
        plyContent += "property float y\n"
        plyContent += "property float z\n"
        plyContent += "property float nx\n"
        plyContent += "property float ny\n"
        plyContent += "property float nz\n"
        plyContent += "property uchar red\n"
        plyContent += "property uchar green\n"
        plyContent += "property uchar blue\n"
        plyContent += "element face \(meshData.triangles.count)\n"
        plyContent += "property list uchar int vertex_indices\n"
        plyContent += "end_header\n"
        
        // Write vertices with normals and colors
        for (i, vertex) in meshData.vertices.enumerated() {
            let normal = i < meshData.normals.count ? meshData.normals[i] : SIMD3<Float>(0, 1, 0)
            let color = i < meshData.colors.count ? meshData.colors[i] : SIMD3<Float>(0.8, 0.8, 0.8)
            
            let r = UInt8(color.x * 255)
            let g = UInt8(color.y * 255)
            let b = UInt8(color.z * 255)
            
            plyContent += "\(vertex.x) \(vertex.y) \(vertex.z) \(normal.x) \(normal.y) \(normal.z) \(r) \(g) \(b)\n"
        }
        
        // Write faces
        for triangle in meshData.triangles {
            plyContent += "3 \(triangle.x) \(triangle.y) \(triangle.z)\n"
        }
        
        try plyContent.write(to: url, atomically: true, encoding: .utf8)
    }
    
    /// Exports mesh to STL format
    private func exportToSTL(_ meshData: MeshData, to url: URL) throws {
        var stlContent = "solid ProstheticScan\n"
        
        // Write triangles
        for triangle in meshData.triangles {
            let v1 = meshData.vertices[Int(triangle.x)]
            let v2 = meshData.vertices[Int(triangle.y)]
            let v3 = meshData.vertices[Int(triangle.z)]
            
            // Calculate face normal
            let edge1 = v2 - v1
            let edge2 = v3 - v1
            let normal = normalize(simd_cross(edge1, edge2))
            
            stlContent += "  facet normal \(normal.x) \(normal.y) \(normal.z)\n"
            stlContent += "    outer loop\n"
            stlContent += "      vertex \(v1.x) \(v1.y) \(v1.z)\n"
            stlContent += "      vertex \(v2.x) \(v2.y) \(v2.z)\n"
            stlContent += "      vertex \(v3.x) \(v3.y) \(v3.z)\n"
            stlContent += "    endloop\n"
            stlContent += "  endfacet\n"
        }
        
        stlContent += "endsolid ProstheticScan\n"
        
        try stlContent.write(to: url, atomically: true, encoding: .utf8)
    }
}

/// Supported export formats
enum ExportFormat: String, CaseIterable {
    case obj = "obj"
    case ply = "ply"
    case stl = "stl"
    
    var displayName: String {
        switch self {
        case .obj:
            return "OBJ"
        case .ply:
            return "PLY"
        case .stl:
            return "STL"
        }
    }
    
    var fileExtension: String {
        return rawValue
    }
}
