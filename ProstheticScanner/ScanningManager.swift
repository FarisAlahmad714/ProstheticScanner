import RealityKit
import ARKit
import Metal
import MetalKit
import Accelerate
import CoreImage
import simd
import Combine

enum ScanningState: Equatable {
    case ready
    case scanning
    case processing
    case completed
    case failed(Error)
    
    // Custom Equatable implementation to handle Error cases
    static func == (lhs: ScanningState, rhs: ScanningState) -> Bool {
        switch (lhs, rhs) {
        case (.ready, .ready),
             (.scanning, .scanning),
             (.processing, .processing),
             (.completed, .completed):
            return true
        case let (.failed(lhsError), .failed(rhsError)):
            return lhsError.localizedDescription == rhsError.localizedDescription
        default:
            return false
        }
    }
}

enum ScanningError: Error {
    case insufficientPoints
    case processingTimeout
    case meshGenerationFailed
    case sessionInterrupted
    case unknown
}

class ScanningManager: NSObject, ObservableObject, ARSessionDelegate, MTKViewDelegate {
    static let shared = ScanningManager()
    
    // MARK: - Published Properties
    @Published var state: ScanningState = .ready
    @Published var progress: Float = 0.0
    @Published var statusMessage: String = "Ready to scan"
    @Published var scannedMesh: MDLMesh?
    @Published var isScanning = false
    @Published var isProcessing = false
    @Published var scanningMessage = "Position device and tap 'Start Scanning'"
    @Published var scanProgress: Float = 0.0
    @Published var pointCount: Int = 0
    @Published var triangleCount: Int = 0
    @Published var averageConfidence: Float? = nil
    @Published var pointDensity: Float? = nil
    @Published var fileSize: UInt64? = nil
    @Published var exportedFileURL: URL? = nil
    @Published var isExporting: Bool = false
    
    // MARK: - Internal Properties (for accessibility in other classes)
    var points: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var confidences: [Float] = []
    var colors: [SIMD3<Float>] = []
    var triangles: [UInt32] = []
    
    // MARK: - Private Properties
    private var arView: ARView?
    private var scanTimer: Timer?
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var metalLibrary: MTLLibrary!
    private var pipelineState: MTLRenderPipelineState!
    private var vertexBuffer: MTLBuffer!
    private var normalBuffer: MTLBuffer!
    private var indexBuffer: MTLBuffer!
    private var mtkView: MTKView!
    private var meshRenderer: MeshRenderer?
    private var currentMeshVertices: [SIMD3<Float>] = []
    private var currentMeshNormals: [SIMD3<Float>] = []
    private var session: ARSession { arView?.session ?? ARSession() }
    private var capturedPoints: [SIMD3<Float>] = []
    private var capturedNormals: [SIMD3<Float>] = []
    private var pointConfidences: [Float] = []
    private var octree: Octree?
    private var processingQueue = DispatchQueue(label: "com.prostheticscanner.processing", qos: .userInitiated)
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Scanning Parameters
    private let minimumConfidence: Float = 0.7
    private let maxPoints = 5000
    private let scanFrequency: TimeInterval = 1.0 / 5.0
    private let strideAmount = 10
    private let maxScanDistance: Float = 1.0
    private let normalCalculationRadius: Int = 3
    private let minimumRequiredPoints = 100
    private var lastTrackingState: ARCamera.TrackingState = .notAvailable
    private var isTrackingQualityAcceptable = false
    private var lastCameraTransform: simd_float4x4?
    private var lastCaptureTime: TimeInterval = 0
    private let minimumCaptureInterval: TimeInterval = 0.1
    private let voxelSize: Float = 0.01
    private let confidenceThreshold: Float = 0.5
    private let distanceThreshold: Float = 0.01  // 1cm
    private let captureFrequency = 10  // Frames
    private let processingTimeout: TimeInterval = 60.0  // 1 minute
    private var frameCount = 0
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        setupMetal()
    }
    
    // MARK: - Public Methods
    
    func setup(arView: ARView) {
        self.arView = arView
        setupARSession()
    }
    
    func startScanning() {
        guard state == .ready else { return }
        
        resetScanningState()
        state = .scanning
        isScanning = true
        statusMessage = "Move around the object to scan all surfaces..."
        showScanningGuidance()  // Add visual guidance
    }
    
    func stopScanning() {
        guard state == .scanning else { return }
        
        state = .processing
        statusMessage = "Processing captured data..."
        
        processMeshGeneration()
    }
    
    func reset() {
        resetScanningState()
        setupARSession()
        state = .ready
        statusMessage = "Ready to scan"
        progress = 0.0
        isScanning = false
        isProcessing = false
        scanningMessage = "Position device and tap 'Start Scanning'"
        scanProgress = 0.0
        pointCount = 0
        triangleCount = 0
        averageConfidence = nil
        pointDensity = nil
        fileSize = nil
        exportedFileURL = nil
        isExporting = false
        clearData()  // Clears the accumulated points, normals, colors, etc.
        print("ScanningManager has been reset.")
    }
    
    // MARK: - Metal Setup
    private func setupMetal() {
        // Device setup
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        print("Metal device name: \(device.name)")
        
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.commandQueue = queue
        print("Successfully created command queue")
        
        do {
            // Try to load custom shader first
            if let libraryPath = Bundle.main.path(forResource: "scan", ofType: "metallib") {
                let libraryURL = URL(fileURLWithPath: libraryPath)
                do {
                    metalLibrary = try device.makeLibrary(URL: libraryURL)
                    print("Successfully loaded Metal library from scan.metallib")
                } catch {
                    print("Error loading scan.metallib: \(error)")
                    metalLibrary = try device.makeDefaultLibrary()
                    print("Using default Metal library")
                }
            } else {
                print("scan.metallib not found, using default library")
                metalLibrary = try device.makeDefaultLibrary()
                print("Using default Metal library")
            }
            
            // Get shader functions
            guard let vertexFunction = metalLibrary.makeFunction(name: "vertex_main"),
                  let fragmentFunction = metalLibrary.makeFunction(name: "fragment_main") else {
                throw NSError(domain: "MetalSetup", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not find Metal shader functions"])
            }
            print("Successfully loaded shader functions")
            
            // Setup vertex descriptor to match VertexIn struct in metal shader
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float4
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            
            vertexDescriptor.attributes[1].format = .float4
            vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD4<Float>>.stride
            vertexDescriptor.attributes[1].bufferIndex = 0
            
            vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD4<Float>>.stride * 2
            vertexDescriptor.layouts[0].stepFunction = .perVertex
            print("Vertex descriptor configured")
            
            // Create pipeline state
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.vertexDescriptor = vertexDescriptor
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            
            // Try to create pipeline state
            do {
                pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
                print("Successfully created pipeline state")
            } catch {
                throw NSError(domain: "MetalSetup", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create pipeline state: \(error.localizedDescription)"])
            }
            
        } catch {
            print("Failed to complete Metal setup: \(error)")
            fatalError("Metal setup failed")
        }
    }
    
    private func setupMetalView() {
        guard let arView = self.arView else { return }
        
        let mtkView = MTKView(frame: arView.bounds, device: device)
        mtkView.delegate = self
        mtkView.device = device
        mtkView.enableSetNeedsDisplay = true
        mtkView.isPaused = true
        mtkView.contentScaleFactor = UIScreen.main.scale
        self.mtkView = mtkView
        print("Metal view setup complete.")
    }
    
    // MARK: - ARKit Setup
    private func setupARView() {
        guard let arView = arView else { return }
        
        // Set up debug visualization
        arView.debugOptions = [
            .showSceneUnderstanding,
            .showWorldOrigin,
            .showFeaturePoints
        ]
        
        // Add wireframe visualization using RealityKit's built-in materials
        let _ = SimpleMaterial(
            color: .white,
            roughness: 1.0,
            isMetallic: false
        )
        
        // Configure debug rendering options
        arView.renderOptions = [
            .disablePersonOcclusion,
            .disableDepthOfField,
            .disableMotionBlur
        ]
        
        // Enable mesh visualization
        arView.environment.sceneUnderstanding.options = [
            .occlusion,
            .physics,
            .receivesLighting
        ]
        
        // Call setupARSession after view setup
        setupARSession()
    }

    private func setupARSession() {
        guard let arView = arView else { return }
        
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        
        // Enable scene mesh
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
        }
        
        // Configure plane detection
        configuration.planeDetection = [.horizontal, .vertical]
        
        arView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        arView.session.delegate = self
    }
    
    // MARK: - Private Methods
    
    private func resetScanningState() {
        capturedPoints = []
        capturedNormals = []
        pointConfidences = []
        octree = nil
        scannedMesh = nil
        frameCount = 0
    }
    
    private func processMeshGeneration() {
        progress = 0.1
        
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            autoreleasepool {
                // Check if we have enough points
                if self.capturedPoints.count < self.minimumRequiredPoints {
                    DispatchQueue.main.async {
                        self.state = .failed(ScanningError.insufficientPoints)
                        self.statusMessage = "Not enough points captured. Please try again."
                    }
                    return
                }
                
                DispatchQueue.main.async {
                    self.progress = 0.3
                    self.statusMessage = "Building 3D surface..."
                }
                
                // Generate mesh using the enhanced method
                do {
                    DispatchQueue.main.async {
                        self.progress = 0.5
                        self.statusMessage = "Generating high-quality mesh..."
                    }
                    
                    let mesh = try self.generateHighQualityMesh()
                    
                    DispatchQueue.main.async {
                        self.scannedMesh = mesh
                        self.state = .completed
                        self.progress = 1.0
                        self.statusMessage = "Scan completed!"
                        self.isProcessing = false
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.state = .failed(error)
                        self.statusMessage = "Mesh generation failed: \(error.localizedDescription)"
                        self.isProcessing = false
                    }
                }
            }
        }
    }
    
    // Add a simple mesh generator for development
    private func generateHighQualityMesh() throws -> MDLMesh {
        guard capturedPoints.count >= minimumRequiredPoints else {
            throw ScanningError.insufficientPoints
        }
        
        let allocator = MTKMeshBufferAllocator(device: device)
        
        // Start with a voxel grid to downsample and regularize points
        let voxelizedPoints = voxelizePointCloud(capturedPoints, normals: capturedNormals, voxelSize: voxelSize)
        
        // Enough points to generate a real mesh?
        if voxelizedPoints.points.count > 100 {
            // Create MDLVertexDescriptor
            let vertexDescriptor = MDLVertexDescriptor()
            vertexDescriptor.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition,
                                                              format: .float3,
                                                              offset: 0,
                                                              bufferIndex: 0)
            vertexDescriptor.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeNormal,
                                                              format: .float3,
                                                              offset: 0,
                                                              bufferIndex: 1)
            vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
            vertexDescriptor.layouts[1] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
            
            // Create vertex buffer
            let positionBuffer = allocator.newBuffer(MemoryLayout<SIMD3<Float>>.stride * voxelizedPoints.points.count,
                                                          type: .vertex)
            
            // Copy points into buffer
            let positionPtr = positionBuffer.map().bytes.bindMemory(to: SIMD3<Float>.self,
                                                                        capacity: voxelizedPoints.points.count)
            for i in 0..<voxelizedPoints.points.count {
                positionPtr[i] = voxelizedPoints.points[i]
            }
            
            // Create normal buffer
            let normalBuffer = allocator.newBuffer(MemoryLayout<SIMD3<Float>>.stride * voxelizedPoints.normals.count,
                                                        type: .vertex)
            
            // Copy normals into buffer
            let normalPtr = normalBuffer.map().bytes.bindMemory(to: SIMD3<Float>.self,
                                                                     capacity: voxelizedPoints.normals.count)
            for i in 0..<voxelizedPoints.normals.count {
                normalPtr[i] = voxelizedPoints.normals[i]
            }
            
            // Generate triangles using a surface reconstruction algorithm
            // For this example, we'll use a simplified triangulation
            let triangleIndices = triangulatePoints(voxelizedPoints.points)
            
            // Create index buffer
            let indexBuffer = allocator.newBuffer(MemoryLayout<UInt32>.stride * triangleIndices.count,
                                                       type: .index)
            
            // Copy indices into buffer
            let indexPtr = indexBuffer.map().bytes.bindMemory(to: UInt32.self,
                                                                  capacity: triangleIndices.count)
            for i in 0..<triangleIndices.count {
                indexPtr[i] = triangleIndices[i]
            }
            
            // Create submesh
            let submesh = MDLSubmesh(indexBuffer: indexBuffer,
                                    indexCount: triangleIndices.count,
                                    indexType: .uint32,
                                    geometryType: .triangles,
                                    material: nil)
            
            // Create the mesh
            let mesh = MDLMesh(vertexBuffers: [positionBuffer, normalBuffer],
                             vertexCount: voxelizedPoints.points.count,
                             descriptor: vertexDescriptor,
                             submeshes: [submesh])
            
            // Update triangle count for UI
            DispatchQueue.main.async {
                self.triangleCount = triangleIndices.count / 3
            }
            
            return mesh
        } else {
            // Fallback to simple sphere if not enough points
            return MDLMesh(sphereWithExtent: [0.1, 0.1, 0.1],
                          segments: [20, 20],
                          inwardNormals: false,
                          geometryType: .triangles,
                          allocator: allocator)
        }
    }
    
    // Helper functions for mesh generation
    private func voxelizePointCloud(_ points: [SIMD3<Float>], normals: [SIMD3<Float>], voxelSize: Float) -> (points: [SIMD3<Float>], normals: [SIMD3<Float>]) {
        guard !points.isEmpty, points.count == normals.count else {
            return ([], [])
        }
        
        // Dictionary to group points by voxel
        var voxelDict: [SIMD3<Int>: (point: SIMD3<Float>, normal: SIMD3<Float>, count: Int)] = [:]
        
        // Place each point in a voxel
        for i in 0..<points.count {
            let point = points[i]
            let normal = normals[i]
            
            // Calculate voxel coordinates
            let voxelX = Int(floor(point.x / voxelSize))
            let voxelY = Int(floor(point.y / voxelSize))
            let voxelZ = Int(floor(point.z / voxelSize))
            let voxelCoord = SIMD3<Int>(voxelX, voxelY, voxelZ)
            
            // Add to or update voxel
            if let existing = voxelDict[voxelCoord] {
                let newPoint = existing.point + point
                let newNormal = existing.normal + normal
                let newCount = existing.count + 1
                voxelDict[voxelCoord] = (newPoint, newNormal, newCount)
            } else {
                voxelDict[voxelCoord] = (point, normal, 1)
            }
        }
        
        // Extract averaged points and normals
        var resultPoints: [SIMD3<Float>] = []
        var resultNormals: [SIMD3<Float>] = []
        
        for (_, value) in voxelDict {
            let avgPoint = value.point / Float(value.count)
            let avgNormal = normalize(value.normal)  // Average then normalize
            
            resultPoints.append(avgPoint)
            resultNormals.append(avgNormal)
        }
        
        return (resultPoints, resultNormals)
    }
    
    private func triangulatePoints(_ points: [SIMD3<Float>]) -> [UInt32] {
        // This is a simplified triangulation - not suitable for concave shapes
        // A real app would use Delaunay triangulation or other advanced algorithms
        
        var triangles: [UInt32] = []
        
        // Need at least 3 points for a triangle
        if points.count < 3 {
            return triangles
        }
        
        // Very basic triangulation - fan triangulation from first point
        // This works for simple convex shapes only
        for i in 1..<(points.count-1) {
            triangles.append(0)                // Center point
            triangles.append(UInt32(i))        // Current point
            triangles.append(UInt32(i + 1))    // Next point
        }
        
        return triangles
    }
    
    // MARK: - Point Cloud Processing
    
    private func captureDepthPoints(frame: ARFrame) {
        guard let depthMap = frame.sceneDepth?.depthMap,
              let confidenceMap = frame.sceneDepth?.confidenceMap else { return }
        
        // Process at reduced frequency to improve performance
        frameCount += 1
        if frameCount % captureFrequency != 0 { return }
        
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
        
        defer {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly)
        }
        
        let depthPointer = unsafeBitCast(CVPixelBufferGetBaseAddress(depthMap), to: UnsafeMutablePointer<Float>.self)
        let confidencePointer = unsafeBitCast(CVPixelBufferGetBaseAddress(confidenceMap), to: UnsafeMutablePointer<UInt8>.self)
        
        // Use adaptive sampling - more points in areas with high detail
        let colorImage = frame.capturedImage
        let adaptiveSampling = calculateAdaptiveSampling(from: colorImage, width: width, height: height)
        
        for y in stride(from: 0, to: height, by: adaptiveSampling) {
            for x in stride(from: 0, to: width, by: adaptiveSampling) {
                let index = y * width + x
                
                // Get depth and confidence
                let depth = depthPointer[index]
                let confidence = Float(confidencePointer[index]) / 255.0
                
                // Skip invalid points
                if depth <= 0 || depth > maxScanDistance || confidence < confidenceThreshold {
                    continue
                }
                
                // Convert pixel to 3D point
                let pointInCamera = self.unprojectPoint(x: Int(x), y: Int(y), depth: depth, 
                                                         intrinsics: frame.camera.intrinsics, 
                                                         viewMatrix: frame.camera.viewMatrix(for: .portrait))
                
                // Add to captured points
                if !self.isTooClose(point: pointInCamera) {
                    self.capturedPoints.append(pointInCamera)
                    
                    // Calculate normal based on neighboring points
                    let normal = self.calculateRobustNormal(at: Int(x), y: Int(y), 
                                                              depthMap: depthPointer, 
                                                              width: width, height: height)
                    self.capturedNormals.append(normal)
                    
                    // Store confidence
                    self.pointConfidences.append(confidence)
                    
                    // Get color from RGB camera
                    if let color = self.getColorFromImage(colorImage, at: CGPoint(x: x, y: y)) {
                        self.colors.append(color)
                    } else {
                        self.colors.append(SIMD3<Float>(1, 1, 1)) // White default
                    }
                }
                
                // Limit points to prevent performance issues
                if self.capturedPoints.count >= self.maxPoints {
                    break
                }
            }
        }
        
        // Update UI with new point count
        DispatchQueue.main.async {
            self.pointCount = self.capturedPoints.count
            
            // Calculate average confidence
            if !self.pointConfidences.isEmpty {
                self.averageConfidence = self.pointConfidences.reduce(0, +) / Float(self.pointConfidences.count)
            }
            
            // Show progress based on point count
            self.progress = min(0.8, Float(self.capturedPoints.count) / Float(self.maxPoints))
        }
    }
    
    private func calculateAdaptiveSampling(from image: CVPixelBuffer, width: Int, height: Int) -> Int {
        // Calculate sampling based on image features
        // More points in high-detail areas, fewer in flat areas
        // This is a simplified version - you could make this more sophisticated
        
        // Default sampling stride
        let defaultStride = 8
        
        // For now, we'll return a fixed value, but this could be improved
        // with edge detection algorithms to use lower stride in high-detail areas
        return defaultStride
    }
    
    private func calculateRobustNormal(at x: Int, y: Int, depthMap: UnsafeMutablePointer<Float>, 
                                     width: Int, height: Int) -> SIMD3<Float> {
        // Calculate surface normal using neighboring points
        // This is a simplified version - a real implementation would use more neighbors
        // and handle edge cases better
        
        let radius = normalCalculationRadius
        var neighbors: [SIMD3<Float>] = []
        
        // Collect neighboring points
        for dy in -radius...radius {
            for dx in -radius...radius {
                let nx = x + dx
                let ny = y + dy
                
                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    let index = ny * width + nx
                    let depth = depthMap[index]
                    
                    if depth > 0 {
                        // Convert to 3D point
                        let point = SIMD3<Float>(Float(nx), Float(ny), depth)
                        neighbors.append(point)
                    }
                }
            }
        }
        
        // Need at least 3 points to compute normal
        if neighbors.count < 3 {
            return SIMD3<Float>(0, 0, 1) // Default normal pointing toward camera
        }
        
        // Compute centroid
        let centroid = neighbors.reduce(SIMD3<Float>(0, 0, 0), +) / Float(neighbors.count)
        
        // Compute covariance matrix
        var covariance = simd_float3x3(0)
        
        for point in neighbors {
            let diff = point - centroid
            
            // Outer product
            covariance.columns.0 += diff * diff.x
            covariance.columns.1 += diff * diff.y
            covariance.columns.2 += diff * diff.z
        }
        
        // Get eigenvector with smallest eigenvalue (normal direction)
        // This is simplified - normally you'd compute eigenvalues/eigenvectors
        // For now, we'll use cross product of two directions as approximation
        
        let dir1 = neighbors[1] - neighbors[0]
        let dir2 = neighbors[2] - neighbors[0]
        let normal = normalize(cross(dir1, dir2))
        
        return normal
    }
    
    private func isTooClose(point: SIMD3<Float>) -> Bool {
        // Check if this point is too close to existing points (for deduplication)
        // Simple approach: check against recent points
        let checkCount = min(capturedPoints.count, 100)
        let startIndex = max(0, capturedPoints.count - checkCount)
        
        for i in startIndex..<capturedPoints.count {
            let distance = length(capturedPoints[i] - point)
            if distance < voxelSize {
                return true
            }
        }
        
        return false
    }
    
    private func getColorFromImage(_ image: CVPixelBuffer, at point: CGPoint) -> SIMD3<Float>? {
        // Extract RGB color from camera image
        // This is a simplified version - you'd need proper pixel format handling
        
        let width = CVPixelBufferGetWidth(image)
        let height = CVPixelBufferGetHeight(image)
        
        // Convert point to image coordinates
        // Note: camera image might be in a different orientation/resolution than depth
        let x = Int(point.x * Float(width) / Float(CVPixelBufferGetWidth(image)))
        let y = Int(point.y * Float(height) / Float(CVPixelBufferGetHeight(image)))
        
        // Bounds check
        if x < 0 || x >= width || y < 0 || y >= height {
            return nil
        }
        
        // Simple placeholder - return a gradient based on position
        // In a real implementation, you'd extract the actual RGB values
        return SIMD3<Float>(Float(x) / Float(width),
                           Float(y) / Float(height),
                           0.5)
    }
    
    // MARK: - Octree Implementation
    
    private func buildOctree() {
        // Create octree with the bounding box of all points
        self.octree = Octree(points: self.capturedPoints, normals: self.capturedNormals, confidences: self.pointConfidences)
    }
    
    private func computeDensityField() -> [Float] {
        // Placeholder for density field computation
        return []
    }
    
    private func generateMeshWithMarchingCubes(densityField: [Float]) throws -> MDLMesh {
        // Add error conditions
        if densityField.isEmpty {
            throw ScanningError.meshGenerationFailed
        }
        
        // Marching cubes implementation could have other failure conditions
        // For example, if the density field is too small:
        if densityField.count < 10 {
            throw ScanningError.insufficientPoints
        }
        
        // If everything is valid, return the mesh
        let allocator = MTKMeshBufferAllocator(device: MTLCreateSystemDefaultDevice()!)
        return MDLMesh(sphereWithExtent: SIMD3<Float>(0.1, 0.1, 0.1), 
                       segments: SIMD2<UInt32>(20, 20), 
                       inwardNormals: false, 
                       geometryType: .triangles, 
                       allocator: allocator)
    }
    
    private func postProcessMesh(_ mesh: MDLMesh) -> MDLMesh {
        // Placeholder for mesh post-processing
        // In a real app, this would smooth and optimize the mesh
        return mesh
    }
    
    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if state == .scanning {
            captureDepthPoints(frame: frame)
            
            DispatchQueue.main.async {
                self.progress = min(0.8, Float(self.capturedPoints.count) / Float(self.maxPoints))
                if self.capturedPoints.count % 1000 == 0 {
                    self.statusMessage = "Captured \(self.capturedPoints.count) points"
                }
            }
        }
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async {
            self.state = .failed(error)
            self.statusMessage = "AR session failed: \(error.localizedDescription)"
        }
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async {
            if self.state == .scanning {
                self.state = .failed(ScanningError.sessionInterrupted)
                self.statusMessage = "Scanning interrupted"
            }
        }
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        DispatchQueue.main.async {
            self.statusMessage = "Session interruption ended. You can restart scanning."
        }
    }
    
    // MARK: - Private Methods
    private func addMeshToARView() {
        guard let arView = arView else {
            print("ARView is not set")
            return
        }
        
        var meshDescriptor = MeshDescriptor()
        meshDescriptor.positions = MeshBuffers.Positions(points)
        meshDescriptor.normals = MeshBuffers.Normals(normals)
        meshDescriptor.primitives = .triangles(triangles)
        
        guard let mesh = try? MeshResource.generate(from: [meshDescriptor]) else {
            print("Failed to generate mesh resource")
            return
        }
        
        let material = SimpleMaterial(color: .blue, isMetallic: false)
        let modelEntity = ModelEntity(mesh: mesh, materials: [material])
        
        let anchorEntity = AnchorEntity(world: SIMD3<Float>(0, 0, 0))
        anchorEntity.addChild(modelEntity)
        arView.scene.addAnchor(anchorEntity)
        print("Mesh added to ARView.")
    }
    
    private func updateMeshDisplay(_ meshData: MeshData) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.points = meshData.vertices
            self.normals = meshData.normals
            self.triangles = meshData.triangles
            self.prepareBuffers()
            self.addMeshToARView()  // Function to add the mesh to the AR scene
            
            self.isProcessing = false
            print("Mesh display updated in ARView.")
        }
    }
    
    private func handleTrackingStateUpdate(_ state: ARCamera.TrackingState) {
        lastTrackingState = state
        
        switch state {
        case .normal:
            isTrackingQualityAcceptable = true
            scanningMessage = "Tracking quality good - proceed with scanning"
            if isScanning { configureARView() }
            
        case .limited(let reason):
            isTrackingQualityAcceptable = false
            switch reason {
            case .initializing:
                scanningMessage = "Initializing AR session - hold steady"
            case .excessiveMotion:
                scanningMessage = "Moving too fast - slow down"
            case .insufficientFeatures:
                scanningMessage = "Not enough visual features - try a more textured area"
            case .relocalizing:
                scanningMessage = "Relocalizing - please wait"
            @unknown default:
                scanningMessage = "Limited tracking quality"
            }
            
        case .notAvailable:
            isTrackingQualityAcceptable = false
            scanningMessage = "Tracking not available"
            print("Camera tracking not available. Ensure camera permissions are granted in device settings.")
            
        @unknown default:
            isTrackingQualityAcceptable = false
            scanningMessage = "Unknown tracking state"
        }
    }

    private func shouldProcessPoint(at depth: Float, confidence: Float) -> Bool {
        // Skip points that are too far or too close
        guard depth > 0.3 && depth < maxScanDistance else { return false }
        
        // Higher confidence threshold for distant points
        let requiredConfidence = minimumConfidence * (1.0 + depth / maxScanDistance)
        guard confidence >= requiredConfidence else { return false }
        
        // Adaptive sampling based on depth
        let randomThreshold = 0.2 + (depth / maxScanDistance) * 0.6
        return Float.random(in: 0...1) < randomThreshold
    }
    
    private func processDepthFrame(
        depthData: Data,
        confidenceData: Data,
        width: Int,
        height: Int,
        bytesPerRow: Int,
        frame: ARFrame
    ) {
        // Bind the raw data pointers to the correct types
        depthData.withUnsafeBytes { rawDepthPointer in
            confidenceData.withUnsafeBytes { rawConfidencePointer in
                guard let depthPointer = rawDepthPointer.bindMemory(to: Float32.self).baseAddress,
                      let confidencePointer = rawConfidencePointer.bindMemory(to: UInt8.self).baseAddress else {
                    print("Failed to bind memory for depth or confidence data.")
                    return
                }
                
                // Retrieve the color image as a CVPixelBuffer
                let colorBuffer = frame.capturedImage
                
                // Initialize arrays to store results
                var newPoints: [SIMD3<Float>] = []
                var newNormals: [SIMD3<Float>] = []
                var newConfidences: [Float] = []
                var newColors: [SIMD3<Float>] = []
                
                // Iterate through each pixel in the depth image
                let totalPixels = width * height
                for y in 0..<height {
                    for x in 0..<width {
                        let index = y * width + x
                        guard index < totalPixels else {
                            print("Index \(index) out of bounds")
                            continue
                        }
                        
                        // Get depth and confidence values
                        let depth = depthPointer[index]
                        let confidence = Float(confidencePointer[index]) / Float(ARConfidenceLevel.high.rawValue)
                        
                        // Only process points with valid depth and confidence
                        guard shouldProcessPoint(at: depth, confidence: confidence) else { continue }
                        
                        // Calculate 3D point
                        let point3D = calculate3DPoint(x: x, y: y, depth: depth)
                        newPoints.append(point3D)
                        
                        // Calculate normal and color
                        let normal = calculateNormal(at: point3D, frame: frame)
                        newNormals.append(normal)
                        
                        let color = getColorForPoint(at: x, y: y, from: colorBuffer)
                        newColors.append(color)
                        
                        newConfidences.append(confidence)
                    }
                }
                
                // Update arrays on the main thread
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.points = newPoints
                    self.normals = newNormals
                    self.confidences = newConfidences
                    self.colors = newColors
                    self.pointCount = newPoints.count
                }
            }
        }
    }
    
    // Helper function to calculate 3D point from pixel coordinates and depth
    private func calculate3DPoint(x: Int, y: Int, depth: Float) -> SIMD3<Float> {
        // Replace with actual calculation based on ARKit's camera intrinsics
        return SIMD3<Float>(Float(x), Float(y), depth)
    }
    
    // Helper function to retrieve color for a point from the color buffer
    private func getColorForPoint(at x: Int, y: Int, from colorBuffer: CVPixelBuffer) -> SIMD3<Float> {
        // Placeholder for color retrieval logic; convert RGB values to SIMD3<Float> format
        return SIMD3<Float>(1.0, 1.0, 1.0) // Example: white color
    }
    
    private func configureARView() {
        guard let arView = arView else { return }
        
        // Set AR debug options to visualize feature points and world origin.
        arView.debugOptions = [
            .showFeaturePoints,
            .showAnchorOrigins,
            .showWorldOrigin
        ]
        
        // Additional rendering options for optimal performance
        arView.renderOptions = [
            .disableMotionBlur,
            .disableDepthOfField,
            .disablePersonOcclusion
        ]
        
        arView.environment.sceneUnderstanding.options = [.occlusion, .physics]
    }
    
    private func processFrameIfNeeded(_ frame: ARFrame) {
        guard let sceneDepth = frame.smoothedSceneDepth,
              let confidenceMapBuffer = frame.smoothedSceneDepth?.confidenceMap else { return }
        
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastCaptureTime >= minimumCaptureInterval else { return }
        
        lastCaptureTime = currentTime
        
        // Convert depth and confidence data for processing
        guard let depthData = extractData(from: sceneDepth.depthMap),
              let confidenceData = extractData(from: confidenceMapBuffer) else {
            print("Failed to extract data from depth or confidence map.")
            return
        }
        
        // Get width, height, and bytesPerRow for the depth data
        let width = CVPixelBufferGetWidth(sceneDepth.depthMap)
        let height = CVPixelBufferGetHeight(sceneDepth.depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(sceneDepth.depthMap)
        
        // Adjust this call if processDepthFrame needs more arguments
        processDepthFrame(depthData: depthData, confidenceData: confidenceData, width: width, height: height, bytesPerRow: bytesPerRow, frame: frame)
    }
    
    // Helper function to convert CVPixelBuffer to Data
    private func extractData(from pixelBuffer: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let dataSize = CVPixelBufferGetDataSize(pixelBuffer)
        let data = Data(bytes: baseAddress, count: dataSize)
        return data
    }
    
    private func updateProgress(message: String, progress: Float) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.scanProgress = progress
            self.scanningMessage = message
        }
    }
    
    private func handleScanError(_ error: MeshError) {
        DispatchQueue.main.async { [weak self] in
            self?.scanningMessage = "Error during scanning: \(error.localizedDescription)"
            self?.isProcessing = false
            self?.clearData()
        }
    }
    
    private func clearData() {
        points.removeAll(keepingCapacity: true)
        normals.removeAll(keepingCapacity: true)
        confidences.removeAll(keepingCapacity: true)
        colors.removeAll(keepingCapacity: true)
        triangles.removeAll(keepingCapacity: true)
        pointCount = 0
        triangleCount = 0
        scanProgress = 0.0
        averageConfidence = nil
        pointDensity = nil
        fileSize = nil
        exportedFileURL = nil
    }
    
    private func prepareBuffers() {
        guard let device = device else { return }
        
        vertexBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<SIMD3<Float>>.size, options: [])
        normalBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<SIMD3<Float>>.size, options: [])
        indexBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<UInt32>.size, options: [])
        print("Buffers prepared.")
    }
    
    // Renders the mesh in Metal (as part of MTKViewDelegate in ScanningManager)
    func renderMesh(in view: MTKView) {
        guard let commandQueue = commandQueue,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }
        
        let commandBuffer = commandQueue.makeCommandBuffer()
        let renderEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        
        // Add rendering commands here, e.g., setting buffers and drawing primitives
        
        renderEncoder?.endEncoding()
        commandBuffer?.present(drawable)
        commandBuffer?.commit()
    }
    
    // MARK: - MTKViewDelegate
    func draw(in view: MTKView) {
        renderMesh(in: view)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        print("Drawable size will change to: \(size)")
    }
    
    // Add these methods to provide better visual feedback during scanning

    private func showScanningGuidance() {
        guard let arView = arView else { return }
        
        // Remove the simple box indicator and replace with mesh visualization
        setupMeshVisualization()
        
        // Update status messages based on point count
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self = self, self.state == .scanning else {
                timer.invalidate()
                return
            }
            
            // Update the mesh visualization with latest captured points
            self.updateMeshVisualization()
            
            // Update status messages
            if self.pointCount < 1000 {
                self.statusMessage = "Move slowly around the object - Need more points"
            } else if self.pointCount < 3000 {
                self.statusMessage = "Getting good data - Continue scanning all sides"
            } else {
                self.statusMessage = "Excellent coverage - Complete when ready"
            }
        }
    }
    
    // Create new methods for real-time mesh visualization
    private var meshVisualization: ModelEntity?
    private var meshAnchor: AnchorEntity?

    private func setupMeshVisualization() {
        guard let arView = arView else { return }
        
        // Create an anchor at world origin
        meshAnchor = AnchorEntity(world: .zero)
        arView.scene.addAnchor(meshAnchor!)
    }

    private func updateMeshVisualization() {
        guard let meshAnchor = meshAnchor, !capturedPoints.isEmpty else { return }
        
        // Remove previous visualization if it exists
        if let oldMesh = meshVisualization {
            meshAnchor.removeChild(oldMesh)
        }
        
        // Create a mesh descriptor from captured points
        var meshDescriptor = MeshDescriptor()
        
        // Use the actual captured points
        let vertices = capturedPoints
        meshDescriptor.positions = MeshBuffers.Positions(vertices)
        
        // If we have enough points, create triangles
        if capturedPoints.count > 10 {
            // Simple triangulation for visualization - this is basic and can be improved
            var triangles: [UInt32] = []
            
            // For better visualization, create a grid-like structure
            for i in 0..<min(capturedPoints.count - 2, 2000) {
                if i % 2 == 0 && i > 0 {
                    triangles.append(UInt32(i))
                    triangles.append(UInt32(i-1))
                    triangles.append(UInt32(i-2))
                }
            }
            
            if !triangles.isEmpty {
                meshDescriptor.primitives = .triangles(triangles)
            }
        }
        
        // Create the mesh with a wireframe material
        do {
            let mesh = try MeshResource.generate(from: [meshDescriptor])
            
            // Create a material that looks like a wireframe
            var material = SimpleMaterial()
            material.color = .init(tint: .red.withAlphaComponent(0.7))
            material.metallic = 0.0
            material.roughness = 1.0
            
            // Create entity
            meshVisualization = ModelEntity(mesh: mesh, materials: [material])
            meshAnchor.addChild(meshVisualization!)
            
            // Update point count for UI
            DispatchQueue.main.async {
                self.pointCount = self.capturedPoints.count
            }
        } catch {
            print("Failed to generate preview mesh: \(error)")
        }
    }
}

// MARK: - Octree Implementation

class Octree {
    // Basic octree implementation - to be expanded
    init(points: [SIMD3<Float>], normals: [SIMD3<Float>], confidences: [Float]) {
        // Initialize octree with the point cloud data
    }
}
