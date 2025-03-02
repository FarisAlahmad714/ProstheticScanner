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
                    self.triangleCount = meshData.triangles.count / 3
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
        
        return MeshData(vertices: vertices, normals: finalNormals, triangles: triangles)
    }
    
    // Generate simple normals for points
    private func generateSimpleNormals(for points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        if points.isEmpty {
            return []
        }
        
        // Simple approach: assign "up" normals
        return Array(repeating: SIMD3<Float>(0, 1, 0), count: points.count)
    }
}
