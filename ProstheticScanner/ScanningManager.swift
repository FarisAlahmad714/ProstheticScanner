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
        statusMessage = "Move around the object to scan all surfaces..."
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
                if self.capturedPoints.count < 1000 {
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
                
                // Build octree
                self.buildOctree()
                
                DispatchQueue.main.async {
                    self.progress = 0.6
                    self.statusMessage = "Generating mesh..."
                }
                
                // Compute density field
                let densityField = self.computeDensityField()
                
                // Try to generate mesh
                do {
                    // Make sure this function can actually throw
                    let mesh = try self.generateMeshWithMarchingCubes(densityField: densityField)
                    
                    DispatchQueue.main.async {
                        self.progress = 0.8
                        self.statusMessage = "Post-processing mesh..."
                    }
                    
                    // Post-process mesh
                    let finalMesh = self.postProcessMesh(mesh)
                    
                    DispatchQueue.main.async {
                        self.scannedMesh = finalMesh
                        self.state = .completed
                        self.progress = 1.0
                        self.statusMessage = "Scan completed!"
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.state = .failed(error)
                        self.statusMessage = "Scanning failed: \(error.localizedDescription)"
                    }
                }
            }
        }
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
        
        // Sample every few pixels to reduce the number of points
        let sampleStep = 4
        
        for y in stride(from: 0, to: height, by: sampleStep) {
            for x in stride(from: 0, to: width, by: sampleStep) {
                let index = y * width + x
                
                let depth = depthPointer[index]
                let confidence = Float(confidencePointer[index]) / 255.0
                
                // Skip invalid depth or low confidence
                if depth.isNaN || depth <= 0 || confidence < confidenceThreshold {
                    continue
                }
                
                // Unproject point to 3D space
                let pointPos = self.unprojectPoint(x: x, y: y, depth: depth, intrinsics: frame.camera.intrinsics, viewMatrix: frame.camera.viewMatrix())
                
                // Skip points that are too far
                if self.capturedPoints.count > 0 {
                    var isTooFar = true
                    for existingPoint in self.capturedPoints.suffix(100) {
                        let distance = length(pointPos - existingPoint)
                        if distance < distanceThreshold {
                            isTooFar = false
                            break
                        }
                    }
                    
                    if isTooFar { continue }
                }
                
                // Calculate surface normal
                let normal = self.calculateNormal(at: pointPos, frame: frame)
                
                // Add the point
                self.capturedPoints.append(pointPos)
                self.capturedNormals.append(normal)
                self.pointConfidences.append(confidence)
                
                // Limit max points to prevent memory issues
                if self.capturedPoints.count >= self.maxPoints {
                    return
                }
            }
        }
    }
    
    private func unprojectPoint(x: Int, y: Int, depth: Float, intrinsics: simd_float3x3, viewMatrix: simd_float4x4) -> SIMD3<Float> {
        // Convert pixel coordinates to normalized device coordinates
        let normalizedX = (Float(x) - intrinsics[2][0]) / intrinsics[0][0]
        let normalizedY = (Float(y) - intrinsics[2][1]) / intrinsics[1][1]
        
        // Create point in camera space
        let cameraPoint = SIMD3<Float>(normalizedX * depth, normalizedY * depth, depth)
        
        // Convert to world space
        let worldPoint = viewMatrix.inverse * SIMD4<Float>(cameraPoint, 1.0)
        return SIMD3<Float>(worldPoint.x, worldPoint.y, worldPoint.z)
    }
    
    private func calculateNormal(at point: SIMD3<Float>, frame: ARFrame) -> SIMD3<Float> {
        // Simplified normal calculation - in a real app, you would use neighboring points
        // to calculate a more accurate normal
        let cameraPosition = frame.camera.transform.columns.3
        let cameraPos = SIMD3<Float>(cameraPosition.x, cameraPosition.y, cameraPosition.z)
        
        let toCamera = normalize(cameraPos - point)
        return toCamera
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
        // Placeholder for marching cubes implementation
        // In a real app, this would extract a mesh from the density field
        
        // For now, returning a simple placeholder mesh
        let allocator = MTKMeshBufferAllocator(device: MTLCreateSystemDefaultDevice()!)
        return MDLMesh(sphereWithExtent: SIMD3<Float>(0.1, 0.1, 0.1), segments: SIMD2<UInt32>(20, 20), inwardNormals: false, geometryType: .triangles, allocator: allocator)
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
}

// MARK: - Octree Implementation

class Octree {
    // Basic octree implementation - to be expanded
    init(points: [SIMD3<Float>], normals: [SIMD3<Float>], confidences: [Float]) {
        // Initialize octree with the point cloud data
    }
}
