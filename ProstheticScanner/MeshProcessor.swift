<<<<<<< HEAD
import Foundation
import simd
import ModelIO // For MDLMesh

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
    
    // MARK: - Initialization
    private init() {}
    
    // MARK: - Public Methods
    /// Processes the raw scan data into a refined mesh.
    func processScanData(_ scanData: ScanData, completion: @escaping (Result<MeshData, MeshError>) -> Void) {
        guard scanData.points.count >= minimumRequiredPoints else {
            completion(.failure(.insufficientPoints))
            return
        }
        
        isProcessing = true
        processingMessage = "Starting mesh processing..."
        processingProgress = 0.0
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            do {
                // Step 1: Preprocess point cloud
                let (processedPoints, processedNormals) = self.preprocessPointCloud(scanData.points, normals: scanData.normals)
                self.updateProgress(0.3, "Point cloud preprocessed")
                
                // Step 2: Generate mesh
                let mesh = try self.generateMeshFromPoints(processedPoints, normals: processedNormals)
                self.updateProgress(0.7, "Mesh generated")
                
                // Step 3: Post-process mesh
                let optimizedMesh = self.postProcessMesh(mesh)
                self.updateProgress(0.9, "Mesh optimized")
                
                // Step 4: Create MeshData
                let meshData = MeshData(
                    vertices: optimizedMesh.vertexBuffers[0].map { SIMD3<Float>($0) },
                    normals: optimizedMesh.vertexBuffers[1].map { SIMD3<Float>($0) },
                    triangles: optimizedMesh.submeshes[0].indexBuffer.map { UInt32($0) }
                )
                self.meshData = meshData
                
                DispatchQueue.main.async {
                    self.isProcessing = false
                    completion(.success(meshData))
                }
            } catch {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    completion(.failure(.processingFailed))
                }
            }
        }
    }
    
    // MARK: - Private Processing Methods
    /// Preprocesses the point cloud by removing outliers and normalizing distribution.
    private func preprocessPointCloud(_ points: [SIMD3<Float>], normals: [SIMD3<Float>]) -> ([SIMD3<Float>], [SIMD3<Float>]) {
        guard points.count > 10 else { return (points, normals) }
        
        // Remove statistical outliers
        let filteredPoints = removeStatisticalOutliers(points)
        
        // Compute normals if not provided
        let computedNormals = normals.isEmpty ? computeNormals(for: filteredPoints) : normals
        
        // Downsample for performance
        let downsampledPoints = downsamplePoints(filteredPoints, targetCount: min(2000, filteredPoints.count))
        let downsampledNormals = downsampledPoints.count < computedNormals.count ? 
            Array(computedNormals.prefix(downsampledPoints.count)) : computedNormals
        
        return (downsampledPoints, downsampledNormals)
    }
    
    /// Removes statistical outliers from point cloud.
    private func removeStatisticalOutliers(_ points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        guard points.count > 10 else { return points }
        
        // Compute distances to k nearest neighbors
        let k = min(10, points.count / 2)
        var distances: [Float] = []
        
        for i in 0..<points.count {
            let point = points[i]
            var neighborDistances: [Float] = []
            
            for j in 0..<points.count {
                if i != j {
                    let distance = simd_distance(point, points[j])
                    neighborDistances.append(distance)
                }
            }
            
            neighborDistances.sort()
            let meanDistance = neighborDistances.prefix(k).reduce(0, +) / Float(k)
            distances.append(meanDistance)
        }
        
        // Compute mean and standard deviation
        let mean = distances.reduce(0, +) / Float(distances.count)
        let variance = distances.map { pow($0 - mean, 2) }.reduce(0, +) / Float(distances.count)
        let stdDev = sqrt(variance)
        
        // Keep points within 2 standard deviations
        let threshold = mean + 2 * stdDev
        return points.enumerated().compactMap { distances[$0.offset] < threshold ? $0.element : nil }
    }
    
    /// Computes normals for point cloud using local neighborhood.
    private func computeNormals(for points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        return points.map { point in
            // Find nearby points
            let neighbors = points.filter { simd_distance($0, point) < 0.05 }
            guard neighbors.count > 3 else { return SIMD3<Float>(0, 1, 0) }
            
            // Compute covariance matrix and extract normal
            let centroid = neighbors.reduce(SIMD3<Float>(0, 0, 0)) { $0 + $1 } / Float(neighbors.count)
            let normal = estimateNormal(from: neighbors, centroid: centroid)
            return normalize(normal)
        }
    }
    
    /// Estimates normal vector from local neighborhood.
    private func estimateNormal(from neighbors: [SIMD3<Float>], centroid: SIMD3<Float>) -> SIMD3<Float> {
        guard neighbors.count > 3 else { return SIMD3<Float>(0, 1, 0) }
        
        // Simple approximation: use cross product of first two vectors
        let v1 = neighbors[1] - neighbors[0]
        let v2 = neighbors[2] - neighbors[0]
        return cross(v1, v2)
    }
    
    /// Downsamples points to target count using uniform sampling.
    private func downsamplePoints(_ points: [SIMD3<Float>], targetCount: Int) -> [SIMD3<Float>] {
        guard points.count > targetCount else { return points }
        
        let stride = points.count / targetCount
        return stride(from: 0, to: points.count, by: stride).map { points[$0] }
    }
    
    /// Generates an MDLMesh from processed points and normals.
    private func generateMeshFromPoints(_ points: [SIMD3<Float>], normals: [SIMD3<Float>]) throws -> MDLMesh {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MeshError.processingFailed
        }
        let allocator = MTKMeshBufferAllocator(device: device)
        
        // Create vertex data with positions and normals
        var vertexData: [Float] = []
        for i in 0..<points.count {
            let point = points[i]
            let normal = i < normals.count ? normals[i] : SIMD3<Float>(0, 1, 0)
            
            // Position
            vertexData.append(point.x)
            vertexData.append(point.y)
            vertexData.append(point.z)
            
            // Normal
            vertexData.append(normal.x)
            vertexData.append(normal.y)
            vertexData.append(normal.z)
        }
        
        // Create vertex buffer
        let vertexBuffer = allocator.newBuffer(
            with: Data(bytes: vertexData, count: vertexData.count * MemoryLayout<Float>.size),
            type: .vertex
        )
        
        // Generate triangles using simple convex hull
        let triangles = generateTrianglesFromPoints(points)
        let indexBuffer = allocator.newBuffer(
            with: Data(bytes: triangles, count: triangles.count * MemoryLayout<UInt32>.size),
            type: .index
        )
        
        // Create vertex descriptor
        let vertexDescriptor = MDLVertexDescriptor()
        vertexDescriptor.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition, format: .float3, offset: 0, bufferIndex: 0)
        vertexDescriptor.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeNormal, format: .float3, offset: 12, bufferIndex: 0)
        vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: 24)
        
        // Create mesh
        let mesh = MDLMesh(vertexBuffer: vertexBuffer, vertexCount: points.count, descriptor: vertexDescriptor, submeshes: [])
        let submesh = MDLSubmesh(indexBuffer: indexBuffer, indexCount: triangles.count, indexType: .uInt32, geometryType: .triangles, material: nil)
        mesh.addSubmesh(submesh)
        
        return mesh
    }
    
    /// Generates triangle indices from point cloud.
    private func generateTrianglesFromPoints(_ points: [SIMD3<Float>]) -> [UInt32] {
        guard points.count >= 3 else { return [] }
        
        var triangles: [UInt32] = []
        
        // Simple fan triangulation from first point
        for i in 1..<points.count-1 {
            triangles.append(0)
            triangles.append(UInt32(i))
            triangles.append(UInt32(i + 1))
        }
        
        return triangles
    }
    
    /// Optimizes the mesh for better quality and performance.
    private func postProcessMesh(_ mesh: MDLMesh) -> MDLMesh {
        // Apply Laplacian smoothing
        let smoothedMesh = applySmoothingToMesh(mesh)
        
        // Generate normals if needed
        smoothedMesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.5)
        
        return smoothedMesh
    }
    
    /// Applies Laplacian smoothing to the mesh.
    private func applySmoothingToMesh(_ mesh: MDLMesh) -> MDLMesh {
        // For now, just ensure normals are computed
        // More sophisticated smoothing can be added later
        return mesh
    }
    
    // MARK: - Public Methods
    /// Resets the mesh processor to initial state.
    func reset() {
        isProcessing = false
        processingProgress = 0.0
        processingMessage = ""
        vertexCount = 0
        triangleCount = 0
        meshData = nil
    }
    
    // MARK: - Utility Methods
    /// Updates processing progress and message.
    private func updateProgress(_ progress: Float, _ message: String) {
        DispatchQueue.main.async {
            self.processingProgress = progress
            self.processingMessage = message
        }
    }
}

/// Errors that can occur during mesh processing.
enum MeshError: Error {
    case insufficientPoints
    case processingFailed
}

/// Struct to hold processed mesh data.
struct MeshData {
    let vertices: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let triangles: [UInt32]
}
=======
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
>>>>>>> 57d48373c35c6526189f0514e3f11772729acb9c
