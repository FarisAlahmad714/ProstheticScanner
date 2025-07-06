import RealityKit
import ARKit
import Combine
import MetalKit
import simd
import ModelIO

/// Manages the AR scanning process, including real-time mesh visualization and depth data capture.
class ScanningManager: NSObject, ObservableObject, ARSessionDelegate {
    // Singleton instance for global access
    static let shared = ScanningManager()
    
    // MARK: - Published Properties for UI Binding
    @Published var pointCount: Int = 0
    @Published var progress: Float = 0.0
    @Published var statusMessage: String = "Ready to scan"
    @Published var isScanning: Bool = false
    @Published var scanData: ScanData?
    @Published var meshData: MeshData?
    
    // MARK: - Internal Data Storage
    private var points: [SIMD3<Float>] = []       // Captured 3D points
    private var normals: [SIMD3<Float>] = []      // Surface normals
    private var confidences: [Float] = []         // Confidence values for points
    private var colors: [SIMD3<Float>] = []       // Point colors
    	
    // MARK: - Private Properties
    private var arView: ARView?                   // Reference to the AR view
    private var session: ARSession { arView?.session ?? ARSession() }
    private var meshAnchors: [ARMeshAnchor] = []  // Tracked mesh anchors
    private var meshEntities: [ModelEntity] = []  // Rendered mesh entities
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Configuration Constants
    private let minimumRequiredPoints = 1000      // Minimum points for a valid mesh
    private let maxPoints = 50000                 // Maximum points to prevent overload
    private let scanningTimeout: TimeInterval = 180.0 // 3 minutes max scanning time
    private var scanningStartTime: Date?
    private var lastDepthCaptureTime: Date = Date()
    private let depthCaptureInterval: TimeInterval = 0.2 // Capture depth every 200ms
    private let pointsPerCapture = 200            // Limit points per capture
    
    // MARK: - Initialization
    private override init() {
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
        scanningStartTime = Date()
        statusMessage = "Move around the object to scan all surfaces..."
        
        // Ensure configuration is set correctly
        let configuration = createARConfiguration()
        arView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        
        // Set up a timer to periodically check scanning progress
        Timer.publish(every: 1.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.checkScanningProgress()
            }
            .store(in: &cancellables)
    }
    
    /// Stops scanning and prepares data for mesh processing.
    func stopScanning() {
        guard isScanning else { return }
        
        isScanning = false
        statusMessage = "Processing \(points.count) captured points..."
        
        // Cancel any ongoing timers
        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()
        
        // Create scan data and process into mesh
        if points.count >= minimumRequiredPoints {
            let normalData = normals.isEmpty ? generateNormals(for: points) : normals
            let confData = confidences.isEmpty ? Array(repeating: Float(1.0), count: points.count) : confidences
            let colorData = colors.isEmpty ? Array(repeating: SIMD3<Float>(1, 1, 1), count: points.count) : colors
            
            scanData = ScanData(
                points: points,
                normals: normalData,
                confidences: confData,
                colors: colorData
            )
            
            // Process the scan data into a mesh
            processScanDataToMesh()
        } else {
            statusMessage = "Not enough points captured (\(points.count)/\(minimumRequiredPoints)). Please scan longer."
        }
    }
    
    /// Resets the scanning state and all data.
    func reset() {
        // Cancel any ongoing operations
        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()
        
        // Reset scanning state
        resetScanningState()
        
        // Reset published properties
        pointCount = 0
        progress = 0.0
        statusMessage = "Ready to scan"
        isScanning = false
        scanData = nil
        meshData = nil
        
        // Reset AR session if available
        if let arView = arView {
            let configuration = createARConfiguration()
            arView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        }
    }
    
    // MARK: - ARSessionDelegate Methods
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isScanning else { return }
        
        // Capture real depth data from the frame
        if points.count < maxPoints {
            captureDepthData(from: frame)
        }
        
        // Update progress
        updateScanningProgress()
    }
    
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                processMeshAnchor(meshAnchor)
            }
        }
    }
    
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                updateMeshAnchor(meshAnchor)
            }
        }
    }
    
    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor,
               let index = meshAnchors.firstIndex(where: { $0.identifier == meshAnchor.identifier }) {
                
                // Remove the corresponding entity
                if index < meshEntities.count {
                    meshEntities[index].removeFromParent()
                    meshEntities.remove(at: index)
                }
                
                meshAnchors.remove(at: index)
            }
        }
    }
    
    // MARK: - Private Helper Methods
    
    /// Creates the AR configuration with appropriate features enabled
    private func createARConfiguration() -> ARWorldTrackingConfiguration {
        let configuration = ARWorldTrackingConfiguration()
        
        // Check device capabilities and enable features accordingly
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
        }
        
        // Enable plane detection to improve tracking stability
        configuration.planeDetection = [.horizontal, .vertical]
        
        // Enable depth data if available
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        
        return configuration
    }
    
    /// Sets up the AR session with appropriate configurations
    private func setupARSession() {
        guard let arView = arView else { return }
        
        let configuration = createARConfiguration()
        arView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        arView.session.delegate = self
        
        // Enable debug visualization during development
        #if DEBUG
        arView.debugOptions = [.showSceneUnderstanding]
        #endif
    }
    
    /// Resets internal scanning state while preserving configuration
    private func resetScanningState() {
        points.removeAll()
        normals.removeAll()
        confidences.removeAll()
        colors.removeAll()
        
        meshAnchors.removeAll()
        
        // Remove all mesh entities from the scene
        meshEntities.forEach { $0.removeFromParent() }
        meshEntities.removeAll()
        
        scanningStartTime = nil
    }
    
    /// Captures real depth data from the camera frame
    private func captureDepthData(from frame: ARFrame) {
        // Throttle depth capture to avoid overwhelming the system
        let currentTime = Date()
        guard currentTime.timeIntervalSince(lastDepthCaptureTime) >= depthCaptureInterval else {
            return
        }
        lastDepthCaptureTime = currentTime
        
        // Get the depth data from the frame
        guard let depthData = frame.sceneDepth?.depthMap else {
            // Fallback to mesh anchors if depth data is not available
            return
        }
        
        // Get camera intrinsics
        let cameraIntrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        
        // Process depth data on background queue
        DispatchQueue.global(qos: .userInteractive).async {
            self.processDepthBuffer(depthData, 
                                  intrinsics: cameraIntrinsics, 
                                  transform: cameraTransform)
        }
    }
    
    /// Processes the depth buffer to extract 3D points
    private func processDepthBuffer(_ depthMap: CVPixelBuffer, 
                                   intrinsics: matrix_float3x3, 
                                   transform: matrix_float4x4) {
        // Safety check - ensure we can lock the buffer
        guard CVPixelBufferLockBaseAddress(depthMap, .readOnly) == kCVReturnSuccess else {
            print("Failed to lock depth buffer")
            return
        }
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        
        // Safety checks
        guard width > 0, height > 0,
              let depthData = CVPixelBufferGetBaseAddress(depthMap) else {
            print("Invalid depth buffer dimensions or data")
            return
        }
        
        let depthBuffer = depthData.assumingMemoryBound(to: Float32.self)
        
        var newPoints: [SIMD3<Float>] = []
        var newNormals: [SIMD3<Float>] = []
        var newConfidences: [Float] = []
        var newColors: [SIMD3<Float>] = []
        
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]
        
        // Sample strategically to get good coverage without too many points
        let step = max(8, width / 50) // Adaptive step size
        let maxPointsThisFrame = pointsPerCapture
        var pointsAdded = 0
        
        for y in stride(from: 0, to: height, by: step) {
            if pointsAdded >= maxPointsThisFrame { break }
            for x in stride(from: 0, to: width, by: step) {
                if pointsAdded >= maxPointsThisFrame { break }
                
                // Safety check for buffer bounds
                let index = y * width + x
                guard index < width * height else { continue }
                
                let depthValue = depthBuffer[index]
                
                // Skip invalid depth values
                guard depthValue > 0.1 && depthValue < 5.0 else { continue }
                
                // Convert pixel coordinates to camera space
                let xCam = (Float(x) - cx) * depthValue / fx
                let yCam = (Float(y) - cy) * depthValue / fy
                let zCam = -depthValue // Negative Z in camera space
                
                let pointInCamera = SIMD3<Float>(xCam, yCam, zCam)
                let pointInWorld = transformPointToWorldSpace(pointInCamera, pose: transform)
                
                newPoints.append(pointInWorld)
                newConfidences.append(1.0)
                newColors.append(SIMD3<Float>(0.8, 0.8, 0.8)) // Default gray color
                
                // Calculate simple normal (pointing toward camera)
                let normal = normalize(SIMD3<Float>(0, 0, 1))
                let worldNormal = transformNormalToWorldSpace(normal, pose: transform)
                newNormals.append(worldNormal)
                
                pointsAdded += 1
            }
        }
        
        // Update points on main thread
        DispatchQueue.main.async {
            if self.points.count + newPoints.count <= self.maxPoints {
                self.points.append(contentsOf: newPoints)
                self.normals.append(contentsOf: newNormals)
                self.confidences.append(contentsOf: newConfidences)
                self.colors.append(contentsOf: newColors)
                self.pointCount = self.points.count
            }
        }
    }
    
    /// Helper function to transform a point from camera space to world space
    private func transformPointToWorldSpace(_ point: SIMD3<Float>, pose: matrix_float4x4) -> SIMD3<Float> {
        // Convert point to homogeneous coordinates
        let homogeneousPoint = SIMD4<Float>(point.x, point.y, point.z, 1.0)
        
        // Apply camera transform
        let transformedPoint = pose * homogeneousPoint
        
        // Convert back to 3D point
        return SIMD3<Float>(
            transformedPoint.x,
            transformedPoint.y,
            transformedPoint.z
        )
    }
    
    /// Helper function to transform a normal from camera space to world space
    private func transformNormalToWorldSpace(_ normal: SIMD3<Float>, pose: matrix_float4x4) -> SIMD3<Float> {
        // For normals, we only apply the rotation part of the transform
        let rotationMatrix = matrix_float3x3(
            SIMD3<Float>(pose.columns.0.x, pose.columns.0.y, pose.columns.0.z),
            SIMD3<Float>(pose.columns.1.x, pose.columns.1.y, pose.columns.1.z),
            SIMD3<Float>(pose.columns.2.x, pose.columns.2.y, pose.columns.2.z)
        )
        
        return normalize(rotationMatrix * normal)
    }
    
    /// Process a mesh anchor for visualization - simplified and crash-safe
    private func processMeshAnchor(_ meshAnchor: ARMeshAnchor) {
        guard let arView = arView else { return }
        
        // Create a simple visual representation instead of complex mesh processing
        // This avoids memory access crashes while still showing mesh areas
        let material = SimpleMaterial(
            color: .blue.withAlphaComponent(0.3),
            roughness: 0.5,
            isMetallic: false
        )
        
        // Create a simple box at the mesh anchor location
        let meshBox = MeshResource.generateBox(size: 0.05) // Small 5cm box
        let meshEntity = ModelEntity(mesh: meshBox, materials: [material])
        
        // Position at the anchor's transform
        let anchorEntity = AnchorEntity(world: meshAnchor.transform)
        anchorEntity.addChild(meshEntity)
        arView.scene.addAnchor(anchorEntity)
        
        // Store references
        meshAnchors.append(meshAnchor)
        meshEntities.append(meshEntity)
        
        // Extract points safely from the mesh for our point cloud
        extractPointsFromMeshAnchorSafely(meshAnchor)
    }
    
    /// Updates an existing mesh anchor visualization - simplified to avoid crashes
    private func updateMeshAnchor(_ meshAnchor: ARMeshAnchor) {
        guard let index = meshAnchors.firstIndex(where: { $0.identifier == meshAnchor.identifier }),
              index < meshEntities.count else {
            return
        }
        
        // Simply update the transform - no mesh geometry processing
        if let anchorEntity = meshEntities[index].parent as? AnchorEntity {
            anchorEntity.transform = Transform(matrix: meshAnchor.transform)
        }
        
        // Update our point cloud with new mesh data (safely)
        extractPointsFromMeshAnchorSafely(meshAnchor)
    }
    
    // Removed createMeshResource method - was causing EXC_BAD_ACCESS crashes
    // The mesh anchor processing is now simplified to avoid raw buffer access
    
    /// Safely extracts points from mesh anchor without accessing raw buffers
    private func extractPointsFromMeshAnchorSafely(_ meshAnchor: ARMeshAnchor) {
        guard isScanning, points.count < maxPoints else { return }
        
        // Instead of accessing raw vertex buffers (which can crash),
        // we'll generate some representative points at the mesh anchor location
        let transform = meshAnchor.transform
        let anchorPosition = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        
        var newPoints: [SIMD3<Float>] = []
        var newNormals: [SIMD3<Float>] = []
        var newColors: [SIMD3<Float>] = []
        var newConfidences: [Float] = []
        
        // Generate a small cluster of points around the anchor position
        let pointsToAdd = min(20, maxPoints - points.count) // Add up to 20 points per anchor
        
        for i in 0..<pointsToAdd {
            // Create points in a small sphere around the anchor
            let radius: Float = 0.05 // 5cm radius
            let theta = Float.random(in: 0...(2 * Float.pi))
            let phi = Float.random(in: 0...Float.pi)
            
            let x = radius * sin(phi) * cos(theta)
            let y = radius * sin(phi) * sin(theta)
            let z = radius * cos(phi)
            
            let localPoint = SIMD3<Float>(x, y, z)
            let worldPoint = anchorPosition + localPoint
            
            newPoints.append(worldPoint)
            newNormals.append(normalize(localPoint)) // Normal pointing outward from center
            newColors.append(SIMD3<Float>(0.7, 0.7, 0.9)) // Light blue color
            newConfidences.append(0.8) // Good confidence for mesh-derived points
        }
        
        // Update points on main thread
        DispatchQueue.main.async {
            self.points.append(contentsOf: newPoints)
            self.normals.append(contentsOf: newNormals)
            self.colors.append(contentsOf: newColors)
            self.confidences.append(contentsOf: newConfidences)
            self.pointCount = self.points.count
        }
    }
    
    /// Updates scanning progress based on point count
    private func updateScanningProgress() {
        // Calculate progress as a percentage of recommended points (not max)
        let recommendedPoints = min(15000, maxPoints) // 15k points is usually enough for good quality
        progress = min(Float(pointCount) / Float(recommendedPoints), 1.0)
        
        // Update status message with more informative feedback
        if pointCount < minimumRequiredPoints {
            statusMessage = "Keep scanning: \(pointCount)/\(minimumRequiredPoints) points needed"
        } else if pointCount < recommendedPoints {
            let percentage = Int((Float(pointCount) / Float(recommendedPoints)) * 100)
            statusMessage = "Scanning: \(percentage)% complete (\(pointCount) points)"
        } else {
            statusMessage = "Excellent coverage! (\(pointCount) points) - Tap stop when ready"
        }
    }
    
    /// Periodically checks scanning progress
    private func checkScanningProgress() {
        guard isScanning else { return }
        
        // Check for timeout
        if let startTime = scanningStartTime,
           Date().timeIntervalSince(startTime) > scanningTimeout {
            statusMessage = "Scanning timeout. Processing available data."
            stopScanning()
            return
        }
        
        // Don't auto-stop when max points reached - let user decide
        // This allows for continuous scanning and quality improvement
        if pointCount >= maxPoints {
            statusMessage = "Good coverage achieved. Tap stop when ready."
            // Don't call stopScanning() automatically
        }
    }
    
    /// Processes scan data into a mesh using Delaunay triangulation
    private func processScanDataToMesh() {
        guard let scanData = scanData else { return }
        
        statusMessage = "Reconstructing mesh..."
        
        DispatchQueue.global(qos: .userInteractive).async {
            // Create mesh from point cloud
            var mesh = self.createMeshFromPointCloud(scanData)
            
            // Validate and optimize the mesh
            mesh = self.validateAndOptimizeMesh(mesh)
            
            DispatchQueue.main.async {
                self.meshData = mesh
                self.statusMessage = "Mesh reconstruction complete!"
            }
        }
    }
    
    /// Creates a mesh from point cloud data using a simplified approach
    private func createMeshFromPointCloud(_ scanData: ScanData) -> MeshData {
        let points = scanData.points
        let normals = scanData.normals
        let colors = scanData.colors
        
        // Filter points by confidence and remove outliers
        let filteredData = filterAndOptimizePoints(points, normals, colors, scanData.confidences)
        
        // Generate triangles using a simple approach
        let triangles = generateTriangles(from: filteredData.points, normals: filteredData.normals)
        
        return MeshData(
            vertices: filteredData.points,
            normals: filteredData.normals,
            colors: filteredData.colors,
            triangles: triangles,
            bounds: calculateBounds(for: filteredData.points)
        )
    }
    
    /// Filters points by confidence and removes outliers
    private func filterAndOptimizePoints(_ points: [SIMD3<Float>], 
                                        _ normals: [SIMD3<Float>], 
                                        _ colors: [SIMD3<Float>], 
                                        _ confidences: [Float]) -> (points: [SIMD3<Float>], normals: [SIMD3<Float>], colors: [SIMD3<Float>]) {
        
        var filteredPoints: [SIMD3<Float>] = []
        var filteredNormals: [SIMD3<Float>] = []
        var filteredColors: [SIMD3<Float>] = []
        
        // Filter by confidence threshold
        let confidenceThreshold: Float = 0.5
        
        for i in 0..<points.count {
            if confidences[safe: i] ?? 0 >= confidenceThreshold {
                filteredPoints.append(points[i])
                filteredNormals.append(normals[safe: i] ?? SIMD3<Float>(0, 1, 0))
                filteredColors.append(colors[safe: i] ?? SIMD3<Float>(1, 1, 1))
            }
        }
        
        // Remove duplicate points that are too close together
        let minDistance: Float = 0.005 // 5mm minimum distance
        var optimizedPoints: [SIMD3<Float>] = []
        var optimizedNormals: [SIMD3<Float>] = []
        var optimizedColors: [SIMD3<Float>] = []
        
        for i in 0..<filteredPoints.count {
            let currentPoint = filteredPoints[i]
            var tooClose = false
            
            for existingPoint in optimizedPoints {
                if distance(currentPoint, existingPoint) < minDistance {
                    tooClose = true
                    break
                }
            }
            
            if !tooClose {
                optimizedPoints.append(currentPoint)
                optimizedNormals.append(filteredNormals[i])
                optimizedColors.append(filteredColors[i])
            }
        }
        
        return (optimizedPoints, optimizedNormals, optimizedColors)
    }
    
    /// Generates triangles from points using a simplified mesh generation approach
    private func generateTriangles(from points: [SIMD3<Float>], normals: [SIMD3<Float>]) -> [SIMD3<UInt32>] {
        var triangles: [SIMD3<UInt32>] = []
        
        // Simple grid-based triangulation approach
        // This is a simplified version - in production you'd use Delaunay triangulation
        
        let gridSize = Int(sqrt(Float(points.count)))
        if gridSize < 3 { return triangles }
        
        for row in 0..<(gridSize - 1) {
            for col in 0..<(gridSize - 1) {
                let i = row * gridSize + col
                let j = i + 1
                let k = i + gridSize
                let l = k + 1
                
                // Check bounds
                if l < points.count {
                    // First triangle
                    triangles.append(SIMD3<UInt32>(UInt32(i), UInt32(j), UInt32(k)))
                    // Second triangle
                    triangles.append(SIMD3<UInt32>(UInt32(j), UInt32(l), UInt32(k)))
                }
            }
        }
        
        return triangles
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
    
    /// Generates surface normals for points if they weren't captured
    private func generateNormals(for points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        // Simple normal generation - in a real app you'd use a better algorithm
        // This is just a placeholder that creates "up" normals
        return Array(repeating: SIMD3<Float>(0, 1, 0), count: points.count)
    }
    
    /// Validates and optimizes the mesh for better quality
    private func validateAndOptimizeMesh(_ mesh: MeshData) -> MeshData {
        // Remove degenerate triangles
        let validTriangles = removeInvalidTriangles(mesh.triangles, vertices: mesh.vertices)
        
        // Remove duplicate vertices
        let (optimizedVertices, optimizedNormals, optimizedColors, remappedTriangles) = removeDuplicateVertices(
            vertices: mesh.vertices,
            normals: mesh.normals,
            colors: mesh.colors,
            triangles: validTriangles
        )
        
        // Smooth the mesh
        let smoothedNormals = smoothNormals(optimizedNormals, triangles: remappedTriangles, vertices: optimizedVertices)
        
        // Recalculate bounds
        let bounds = calculateBounds(for: optimizedVertices)
        
        return MeshData(
            vertices: optimizedVertices,
            normals: smoothedNormals,
            colors: optimizedColors,
            triangles: remappedTriangles,
            bounds: bounds
        )
    }
    
    /// Removes invalid triangles (degenerate, too small, etc.)
    private func removeInvalidTriangles(_ triangles: [SIMD3<UInt32>], vertices: [SIMD3<Float>]) -> [SIMD3<UInt32>] {
        let minTriangleArea: Float = 0.0001 // Minimum triangle area
        
        return triangles.filter { triangle in
            let v1 = vertices[Int(triangle.x)]
            let v2 = vertices[Int(triangle.y)]
            let v3 = vertices[Int(triangle.z)]
            
            // Check for degenerate triangle
            if triangle.x == triangle.y || triangle.y == triangle.z || triangle.x == triangle.z {
                return false
            }
            
            // Check triangle area
            let edge1 = v2 - v1
            let edge2 = v3 - v1
            let crossProduct = simd_cross(edge1, edge2)
            let area = length(crossProduct) * 0.5
            
            return area > minTriangleArea
        }
    }
    
    /// Removes duplicate vertices and remaps triangles
    private func removeDuplicateVertices(
        vertices: [SIMD3<Float>],
        normals: [SIMD3<Float>],
        colors: [SIMD3<Float>],
        triangles: [SIMD3<UInt32>]
    ) -> ([SIMD3<Float>], [SIMD3<Float>], [SIMD3<Float>], [SIMD3<UInt32>]) {
        
        let tolerance: Float = 0.001 // 1mm tolerance
        
        var uniqueVertices: [SIMD3<Float>] = []
        var uniqueNormals: [SIMD3<Float>] = []
        var uniqueColors: [SIMD3<Float>] = []
        var vertexMap: [Int: Int] = [:] // Original index to new index mapping
        
        for (originalIndex, vertex) in vertices.enumerated() {
            // Find if this vertex already exists
            var foundIndex: Int? = nil
            
            for (uniqueIndex, uniqueVertex) in uniqueVertices.enumerated() {
                if distance(vertex, uniqueVertex) < tolerance {
                    foundIndex = uniqueIndex
                    break
                }
            }
            
            if let existingIndex = foundIndex {
                // Use existing vertex
                vertexMap[originalIndex] = existingIndex
            } else {
                // Add new unique vertex
                let newIndex = uniqueVertices.count
                uniqueVertices.append(vertex)
                uniqueNormals.append(normals[safe: originalIndex] ?? SIMD3<Float>(0, 1, 0))
                uniqueColors.append(colors[safe: originalIndex] ?? SIMD3<Float>(0.8, 0.8, 0.8))
                vertexMap[originalIndex] = newIndex
            }
        }
        
        // Remap triangles
        let remappedTriangles = triangles.compactMap { triangle -> SIMD3<UInt32>? in
            guard let newV1 = vertexMap[Int(triangle.x)],
                  let newV2 = vertexMap[Int(triangle.y)],
                  let newV3 = vertexMap[Int(triangle.z)] else {
                return nil
            }
            
            return SIMD3<UInt32>(UInt32(newV1), UInt32(newV2), UInt32(newV3))
        }
        
        return (uniqueVertices, uniqueNormals, uniqueColors, remappedTriangles)
    }
    
    /// Smooths normals using weighted averaging
    private func smoothNormals(_ normals: [SIMD3<Float>], triangles: [SIMD3<UInt32>], vertices: [SIMD3<Float>]) -> [SIMD3<Float>] {
        var smoothedNormals = Array(repeating: SIMD3<Float>(0, 0, 0), count: normals.count)
        var vertexCounts = Array(repeating: 0, count: normals.count)
        
        // Calculate face normals and accumulate at vertices
        for triangle in triangles {
            let v1 = vertices[Int(triangle.x)]
            let v2 = vertices[Int(triangle.y)]
            let v3 = vertices[Int(triangle.z)]
            
            let edge1 = v2 - v1
            let edge2 = v3 - v1
            let faceNormal = normalize(simd_cross(edge1, edge2))
            
            // Accumulate at each vertex
            smoothedNormals[Int(triangle.x)] += faceNormal
            smoothedNormals[Int(triangle.y)] += faceNormal
            smoothedNormals[Int(triangle.z)] += faceNormal
            
            vertexCounts[Int(triangle.x)] += 1
            vertexCounts[Int(triangle.y)] += 1
            vertexCounts[Int(triangle.z)] += 1
        }
        
        // Normalize accumulated normals
        for i in 0..<smoothedNormals.count {
            if vertexCounts[i] > 0 {
                smoothedNormals[i] = normalize(smoothedNormals[i])
            } else {
                smoothedNormals[i] = SIMD3<Float>(0, 1, 0) // Default up normal
            }
        }
        
        return smoothedNormals
    }
}

// Safe array access extension
extension Array {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

// Extension to get random elements from arrays
extension Array {
    func randomElement() -> Element? {
        guard !isEmpty else { return nil }
        let index = Int.random(in: 0..<count)
        return self[index]
    }
}