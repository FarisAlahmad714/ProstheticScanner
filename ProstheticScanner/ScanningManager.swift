import RealityKit
import ARKit
import Combine

/// Manages the AR scanning process, including real-time mesh visualization and depth data capture.
class ScanningManager: NSObject, ObservableObject, ARSessionDelegate {
    // Singleton instance for global access
    static let shared = ScanningManager()
    
    // MARK: - Published Properties for UI Binding
    @Published var pointCount: Int = 0
    @Published var progress: Float = 0.0
    @Published var statusMessage: String = "Ready to scan"
    @Published var isScanning: Bool = false
    @Published var scannedMesh: MDLMesh?
    @Published var meshData: MeshData?
    
    // Computed property for scan data
    var scanData: ScanData {
        return ScanData(points: points, normals: normals, confidences: confidences, colors: colors)
    }
    
    // MARK: - Internal Data Storage
    private(set) var points: [SIMD3<Float>] = []       // Captured 3D points
    private(set) var normals: [SIMD3<Float>] = []      // Surface normals
    private(set) var confidences: [Float] = []         // Confidence values for points
    private(set) var colors: [SIMD3<Float>] = []       // Point colors
    private(set) var triangles: [UInt32] = []          // Mesh triangle indices
    
    // MARK: - Private Properties
    private var arView: ARView?                        // Reference to the AR view
    private var session: ARSession { arView?.session ?? ARSession() }
    private var meshAnchors: [ARMeshAnchor] = []       // Tracked mesh anchors
    private var meshEntities: [ModelEntity] = []       // Rendered mesh entities
    
    // MARK: - Configuration Constants
    private let minimumRequiredPoints = 100            // Minimum points for a valid mesh
    private let maxPoints = 5000                       // Maximum points to prevent overload
    
    // MARK: - Initialization
    override init() {
        super.init()
    }
    
    // MARK: - Setup
    /// Configures the ScanningManager with an ARView and initializes the AR session.
    func setup(arView: ARView) {
        self.arView = arView
        setupARSession()
    }
    
    // MARK: - Scanning Controls
    /// Starts the scanning process, resetting state and running the AR session.
    func startScanning() {
        guard let arView = arView else {
            statusMessage = "Error: ARView not initialized"
            return
        }
        resetScanningState()
        isScanning = true
        statusMessage = "Move around the object to scan all surfaces..."
        arView.session.run(arView.session.configuration!, options: [])
    }
    
    /// Stops scanning and initiates mesh processing.
    func stopScanning() {
        isScanning = false
        statusMessage = "Processing captured data..."
        processMeshGeneration()
    }
    
    /// Resets the scanning state to initial conditions.
    func reset() {
        resetScanningState()
        setupARSession()
        pointCount = 0
        progress = 0.0
        statusMessage = "Ready to scan"
        isScanning = false
        scannedMesh = nil
        clearData()
    }
    
    // MARK: - ARSessionDelegate Methods
    /// Updates scanning data with each new AR frame.
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if isScanning {
            captureDepthPoints(frame: frame)
            updateProgress()
        }
    }
    
    /// Adds new mesh anchors to the scene for real-time visualization.
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        guard let arView = arView else { return }
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                addMeshAnchorToScene(meshAnchor)
            }
        }
    }
    
    /// Updates existing mesh anchors as they change.
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let arView = arView else { return }
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                updateMeshAnchorInScene(meshAnchor)
            }
        }
    }
    
    // MARK: - Private Helper Methods
    /// Configures the AR session with scene reconstruction enabled.
    private func setupARSession() {
        guard let arView = arView else { return }
        let configuration = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
            configuration.planeDetection = [.horizontal, .vertical] // Enhance tracking
        }
        configuration.frameSemantics.insert(.sceneDepth) // Enable depth data
        arView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        arView.session.delegate = self
    }
    
    /// Resets internal scanning state while preserving configuration.
    private func resetScanningState() {
        points = []
        meshAnchors = []
        meshEntities.forEach { $0.removeFromParent() }
        meshEntities = []
    }
    
    /// Clears all captured data.
    private func clearData() {
        points.removeAll()
        normals.removeAll()
        confidences.removeAll()
        colors.removeAll()
        triangles.removeAll()
    }
    
    /// Captures depth points from the AR frame.
    private func captureDepthPoints(frame: ARFrame) {
        guard let sceneDepth = frame.sceneDepth else { return }
        let depthMap = sceneDepth.depthMap
        let transform = frame.camera.transform
        
        // Convert depth map to point cloud (simplified example)
        let newPoints = extractPoints(from: depthMap, transform: transform)
        points.append(contentsOf: newPoints)
        pointCount = points.count
    }
    
    /// Extracts 3D points from a depth map with camera transform.
    private func extractPoints(from depthMap: CVPixelBuffer, transform: matrix_float4x4) -> [SIMD3<Float>] {
        var extractedPoints: [SIMD3<Float>] = []
        
        CVPixelBufferLockBaseAddress(depthMap, CVPixelBufferLockFlags.readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, CVPixelBufferLockFlags.readOnly) }
        
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return [] }
        let depthData = baseAddress.assumingMemoryBound(to: Float32.self)
        
        // Camera intrinsics (approximate values for iPhone)
        let fx: Float = 570.0 // Focal length X
        let fy: Float = 570.0 // Focal length Y
        let cx: Float = Float(width) / 2.0 // Principal point X
        let cy: Float = Float(height) / 2.0 // Principal point Y
        
        // Sample every nth pixel to avoid too many points
        let stride = max(1, min(width, height) / 100)
        
        for y in stride(from: 0, to: height, by: stride) {
            for x in stride(from: 0, to: width, by: stride) {
                let depthIndex = y * (bytesPerRow / MemoryLayout<Float32>.size) + x
                let depth = depthData[depthIndex]
                
                // Skip invalid depth values
                guard depth > 0.01 && depth < 10.0 else { continue }
                
                // Convert pixel coordinates to camera space
                let cameraX = (Float(x) - cx) * depth / fx
                let cameraY = (Float(y) - cy) * depth / fy
                let cameraZ = -depth // Negative because camera looks down -Z axis
                
                // Transform to world space
                let cameraPoint = SIMD4<Float>(cameraX, cameraY, cameraZ, 1.0)
                let worldPoint = transform * cameraPoint
                
                extractedPoints.append(SIMD3<Float>(worldPoint.x, worldPoint.y, worldPoint.z))
            }
        }
        
        return extractedPoints
    }
    
    /// Updates scanning progress and status.
    private func updateProgress() {
        progress = min(Float(pointCount) / Float(maxPoints), 1.0)
        statusMessage = "Scanning: \(pointCount) points captured"
    }
    
    /// Adds a mesh anchor to the scene as a visual entity.
    private func addMeshAnchorToScene(_ meshAnchor: ARMeshAnchor) {
        guard let arView = arView else { return }
        let meshGeometry = meshAnchor.geometry
        do {
            let meshResource = try MeshResource.generate(from: [meshGeometry])
            let material = SimpleMaterial(color: .blue.withAlphaComponent(0.5), isMetallic: false)
            let meshEntity = ModelEntity(mesh: meshResource, materials: [material])
            let anchorEntity = AnchorEntity(anchor: meshAnchor)
            anchorEntity.addChild(meshEntity)
            arView.scene.addAnchor(anchorEntity)
            meshEntities.append(meshEntity)
            meshAnchors.append(meshAnchor)
        } catch {
            print("Failed to generate mesh resource: \(error)")
        }
    }
    
    /// Updates an existing mesh entity with new geometry.
    private func updateMeshAnchorInScene(_ meshAnchor: ARMeshAnchor) {
        guard let arView = arView,
              let anchorEntity = arView.scene.anchors.first(where: { $0.anchor == meshAnchor }),
              let meshEntity = anchorEntity.children.first as? ModelEntity else { return }
        do {
            let meshGeometry = meshAnchor.geometry
            let meshResource = try MeshResource.generate(from: [meshGeometry])
            meshEntity.model?.mesh = meshResource
        } catch {
            print("Failed to update mesh resource: \(error)")
        }
    }
    
    /// Processes captured points into a final mesh.
    private func processMeshGeneration() {
        if points.count >= minimumRequiredPoints {
            let mesh = generateMeshFromPoints(points)
            scannedMesh = mesh
            statusMessage = "Scan completed!"
        } else {
            statusMessage = "Not enough points captured (\(pointCount)/\(minimumRequiredPoints)). Please try again."
        }
    }
    
    /// Generates an MDLMesh from captured points using Delaunay triangulation.
    private func generateMeshFromPoints(_ points: [SIMD3<Float>]) -> MDLMesh {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not available")
        }
        let allocator = MTKMeshBufferAllocator(device: device)
        
        // Simple convex hull approximation for now
        let hull = computeConvexHull(points)
        
        // Create vertices buffer
        let vertexData = hull.flatMap { [$0.x, $0.y, $0.z] }
        let vertexBuffer = allocator.newBuffer(with: Data(bytes: vertexData, count: vertexData.count * MemoryLayout<Float>.size), type: .vertex)
        
        // Create triangles using fan triangulation from first vertex
        var triangles: [UInt32] = []
        for i in 1..<hull.count-1 {
            triangles.append(0)
            triangles.append(UInt32(i))
            triangles.append(UInt32(i + 1))
        }
        
        let indexBuffer = allocator.newBuffer(with: Data(bytes: triangles, count: triangles.count * MemoryLayout<UInt32>.size), type: .index)
        
        // Create mesh
        let mesh = MDLMesh(vertexBuffer: vertexBuffer, vertexCount: hull.count, descriptor: MDLVertexDescriptor.defaultLayout, submeshes: [])
        let submesh = MDLSubmesh(indexBuffer: indexBuffer, indexCount: triangles.count, indexType: .uInt32, geometryType: .triangles, material: nil)
        mesh.addSubmesh(submesh)
        
        return mesh
    }
    
    /// Computes a simple convex hull approximation.
    private func computeConvexHull(_ points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        guard points.count > 3 else { return points }
        
        // Find extreme points
        let minX = points.min { $0.x < $1.x }!
        let maxX = points.max { $0.x < $1.x }!
        let minY = points.min { $0.y < $1.y }!
        let maxY = points.max { $0.y < $1.y }!
        let minZ = points.min { $0.z < $1.z }!
        let maxZ = points.max { $0.z < $1.z }!
        
        // Return unique extreme points
        let extremePoints = [minX, maxX, minY, maxY, minZ, maxZ]
        return Array(Set(extremePoints.map { $0 }))
    }
}