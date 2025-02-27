import RealityKit
import ARKit
import Metal
import MetalKit
import Accelerate
import CoreImage
import simd
import Combine

/// Represents the current state of the scanning process.
enum ScanningState: Equatable {
    case ready
    case scanning
    case processing
    case completed
    case failed(Error)
    
    static func == (lhs: ScanningState, rhs: ScanningState) -> Bool {
        switch (lhs, rhs) {
        case (.ready, .ready), (.scanning, .scanning), (.processing, .processing), (.completed, .completed):
            return true
        case let (.failed(lhsError), .failed(rhsError)):
            return lhsError.localizedDescription == rhsError.localizedDescription
        default:
            return false
        }
    }
}

/// Defines possible errors during the scanning process.
enum ScanningError: Error {
    case insufficientPoints
    case processingTimeout
    case meshGenerationFailed
    case sessionInterrupted
    case unknown
}

/// Manages the AR scanning session, point cloud capture, and mesh rendering.
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
    
    // MARK: - Internal Properties
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
    private let distanceThreshold: Float = 0.01
    private let captureFrequency = 10
    private let processingTimeout: TimeInterval = 60.0
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
        setupMetalView()
    }
    
    func startScanning() {
        guard state == .ready else { return }
        
        resetScanningState()
        state = .scanning
        isScanning = true
        statusMessage = "Move around the limb to scan all surfaces..."
        showScanningGuidance()
        scanTimer = Timer.scheduledTimer(withTimeInterval: scanFrequency, repeats: true) { [weak self] _ in
            self?.updateScanningProgress()
        }
    }
    
    func stopScanning() throws {
        guard state == .scanning else { throw ScanningError.sessionInterrupted }
        
        state = .processing
        statusMessage = "Processing captured data..."
        isScanning = false
        scanTimer?.invalidate()
        
        return try processMeshGeneration()
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
        clearData()
    }
    
    // MARK: - Metal Setup
    private func setupMetal() {
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal is not supported on this device") }
        self.device = device
        guard let queue = device.makeCommandQueue() else { fatalError("Failed to create command queue") }
        self.commandQueue = queue
        
        do {
            if let libraryPath = Bundle.main.path(forResource: "scan", ofType: "metallib") {
                let libraryURL = URL(fileURLWithPath: libraryPath)
                metalLibrary = try device.makeLibrary(URL: libraryURL)
            } else {
                metalLibrary = try device.makeDefaultLibrary()
            }
            
            guard let vertexFunction = metalLibrary.makeFunction(name: "vertex_main"),
                  let fragmentFunction = metalLibrary.makeFunction(name: "fragment_main") else {
                throw NSError(domain: "MetalSetup", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not find Metal shader functions"])
            }
            
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float4
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            vertexDescriptor.attributes[1].format = .float4
            vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD4<Float>>.stride
            vertexDescriptor.attributes[1].bufferIndex = 0
            vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD4<Float>>.stride * 2
            vertexDescriptor.layouts[0].stepFunction = .perVertex
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.vertexDescriptor = vertexDescriptor
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Failed to complete Metal setup: \(error)")
        }
    }
    
    private func setupMetalView() {
        guard let arView = self.arView else { return }
        mtkView = MTKView(frame: arView.bounds, device: device)
        mtkView.delegate = self
        mtkView.device = device
        mtkView.enableSetNeedsDisplay = true
        mtkView.isPaused = true
        mtkView.contentScaleFactor = UIScreen.main.scale
        arView.addSubview(mtkView!)
    }
    
    // MARK: - ARKit Setup
    private func setupARSession() {
        guard let arView = arView else { return }
        
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
        }
        configuration.planeDetection = [.horizontal, .vertical]
        
        arView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        arView.session.delegate = self
        setupARView()
    }
    
    private func setupARView() {
        guard let arView = arView else { return }
        arView.debugOptions = [.showFeaturePoints, .showWorldOrigin, .showSceneUnderstanding]
        arView.renderOptions = [.disablePersonOcclusion, .disableDepthOfField, .disableMotionBlur]
        arView.environment.sceneUnderstanding.options = [.occlusion, .physics, .receivesLighting]
    }
    
    // MARK: - Mesh Processing
    private func processMeshGeneration() throws -> MDLMesh {
        guard capturedPoints.count >= minimumRequiredPoints else {
            throw ScanningError.insufficientPoints
        }
        
        let scanData = ScanData(points: capturedPoints, normals: capturedNormals, confidences: pointConfidences)
        let result = try MeshProcessor.shared.processScanData(scanData)
        switch result {
        case .success(let meshData):
            let mesh = try generateHighQualityMesh(from: meshData)
            DispatchQueue.main.async {
                self.scannedMesh = mesh
                self.state = .completed
                self.progress = 1.0
                self.statusMessage = "Scan completed!"
                self.isProcessing = false
                self.triangleCount = meshData.triangles.count / 3
                self.addMeshToARView()
            }
            return mesh
        case .failure(let error):
            throw error
        }
    }
    
    private func generateHighQualityMesh(from meshData: MeshData) throws -> MDLMesh {
        let allocator = MTKMeshBufferAllocator(device: device)
        
        let vertexDescriptor = MDLVertexDescriptor()
        vertexDescriptor.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition, format: .float3, offset: 0, bufferIndex: 0)
        vertexDescriptor.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeNormal, format: .float3, offset: 0, bufferIndex: 1)
        vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
        vertexDescriptor.layouts[1] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
        
        let positionBuffer = allocator.newBuffer(MemoryLayout<SIMD3<Float>>.stride * meshData.vertices.count, type: .vertex)
        let positionPtr = positionBuffer.map().bytes.bindMemory(to: SIMD3<Float>.self, capacity: meshData.vertices.count)
        for i in 0..<meshData.vertices.count { positionPtr[i] = meshData.vertices[i] }
        
        let normalBuffer = allocator.newBuffer(MemoryLayout<SIMD3<Float>>.stride * meshData.normals.count, type: .vertex)
        let normalPtr = normalBuffer.map().bytes.bindMemory(to: SIMD3<Float>.self, capacity: meshData.normals.count)
        for i in 0..<meshData.normals.count { normalPtr[i] = meshData.normals[i] }
        
        let indexBuffer = allocator.newBuffer(MemoryLayout<UInt32>.stride * meshData.triangles.count, type: .index)
        let indexPtr = indexBuffer.map().bytes.bindMemory(to: UInt32.self, capacity: meshData.triangles.count)
        for i in 0..<meshData.triangles.count { indexPtr[i] = meshData.triangles[i] }
        
        let submesh = MDLSubmesh(indexBuffer: indexBuffer, indexCount: meshData.triangles.count, indexType: .uint32, geometryType: .triangles, material: nil)
        let mesh = MDLMesh(vertexBuffers: [positionBuffer, normalBuffer], vertexCount: meshData.vertices.count, descriptor: vertexDescriptor, submeshes: [submesh])
        
        return mesh
    }
    
    // MARK: - Point Cloud Processing
    private func voxelizePointCloud(_ points: [SIMD3<Float>], normals: [SIMD3<Float>], voxelSize: Float) -> (points: [SIMD3<Float>], normals: [SIMD3<Float>]) {
        guard !points.isEmpty, points.count == normals.count else { return ([], []) }
        var voxelDict: [SIMD3<Int>: (point: SIMD3<Float>, normal: SIMD3<Float>, count: Int)] = [:]
        
        for i in 0..<points.count {
            let point = points[i]
            let normal = normals[i]
            let voxelX = Int(floor(point.x / voxelSize))
            let voxelY = Int(floor(point.y / voxelSize))
            let voxelZ = Int(floor(point.z / voxelSize))
            let voxelCoord = SIMD3<Int>(voxelX, voxelY, voxelZ)
            
            if let existing = voxelDict[voxelCoord] {
                voxelDict[voxelCoord] = (existing.point + point, existing.normal + normal, existing.count + 1)
            } else {
                voxelDict[voxelCoord] = (point, normal, 1)
            }
        }
        
        var resultPoints: [SIMD3<Float>] = []
        var resultNormals: [SIMD3<Float>] = []
        for (_, value) in voxelDict {
            resultPoints.append(value.point / Float(value.count))
            resultNormals.append(normalize(value.normal))
        }
        
        return (resultPoints, resultNormals)
    }
    
    private func triangulatePoints(_ points: [SIMD3<Float>]) -> [UInt32] {
        var triangles: [UInt32] = []
        if points.count < 3 { return triangles }
        for i in 1..<(points.count-1) {
            triangles.append(0)
            triangles.append(UInt32(i))
            triangles.append(UInt32(i + 1))
        }
        return triangles
    }
    
    private func captureDepthPoints(frame: ARFrame) throws {
        guard let depthMap = frame.smoothedSceneDepth?.depthMap,
              let confidenceMap = frame.smoothedSceneDepth?.confidenceMap else { throw ScanningError.unknown }
        
        frameCount += 1
        if frameCount % captureFrequency != 0 { return }
        
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastCaptureTime >= minimumCaptureInterval else { return }
        lastCaptureTime = currentTime
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly)
        }
        
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let depthPointer = unsafeBitCast(CVPixelBufferGetBaseAddress(depthMap), to: UnsafeMutablePointer<Float32>.self)
        let confidencePointer = unsafeBitCast(CVPixelBufferGetBaseAddress(confidenceMap), to: UnsafeMutablePointer<UInt8>.self)
        
        let adaptiveSampling = calculateAdaptiveSampling(from: frame.capturedImage, width: width, height: height)
        
        for y in stride(from: 0, to: height, by: adaptiveSampling) {
            for x in stride(from: 0, to: width, by: adaptiveSampling) {
                let index = y * width + x
                let depth = depthPointer[index]
                let confidence = Float(confidencePointer[index]) / 255.0
                
                guard depth > 0, depth <= maxScanDistance, confidence >= confidenceThreshold else { continue }
                let point = try unprojectPoint(x: x, y: y, depth: depth, frame: frame)
                
                if !isTooClose(point: point) {
                    capturedPoints.append(point)
                    capturedNormals.append(calculateRobustNormal(at: x, y: y, depthMap: depthPointer, width: width, height: height))
                    pointConfidences.append(confidence)
                    colors.append(getColorFromImage(frame.capturedImage, at: CGPoint(x: x, y: y)) ?? SIMD3<Float>(1, 1, 1))
                }
                
                if capturedPoints.count >= maxPoints { break }
            }
        }
        
        DispatchQueue.main.async {
            self.pointCount = self.capturedPoints.count
            if !self.pointConfidences.isEmpty {
                self.averageConfidence = self.pointConfidences.reduce(0, +) / Float(self.pointConfidences.count)
            }
            self.progress = min(0.8, Float(self.capturedPoints.count) / Float(self.maxPoints))
            self.updateMeshVisualization()
        }
    }
    
    private func calculateAdaptiveSampling(from image: CVPixelBuffer, width: Int, height: Int) -> Int {
        return strideAmount // Placeholder for edge detection-based sampling
    }
    
    private func calculateRobustNormal(at x: Int, y: Int, depthMap: UnsafeMutablePointer<Float32>, width: Int, height: Int) -> SIMD3<Float> {
        let radius = normalCalculationRadius
        var neighbors: [SIMD3<Float>] = []
        
        for dy in -radius...radius {
            for dx in -radius...radius {
                let nx = x + dx, ny = y + dy
                if nx >= 0, nx < width, ny >= 0, ny < height {
                    let index = ny * width + nx
                    let depth = depthMap[index]
                    if depth > 0 {
                        let point = try? unprojectPoint(x: nx, y: ny, depth: depth, frame: ARFrame()) ?? SIMD3<Float>(0, 0, 0)
                        neighbors.append(point)
                    }
                }
            }
        }
        
        if neighbors.count < 3 { return SIMD3<Float>(0, 0, 1) }
        let centroid = neighbors.reduce(.zero, +) / Float(neighbors.count)
        var covariance = simd_float3x3()
        for point in neighbors {
            let diff = point - centroid
            covariance += outer(diff, diff)
        }
        let dir1 = neighbors[1] - neighbors[0]
        let dir2 = neighbors[2] - neighbors[0]
        return normalize(cross(dir1, dir2))
    }
    
    private func isTooClose(point: SIMD3<Float>) -> Bool {
        let checkCount = min(capturedPoints.count, 100)
        let startIndex = max(0, capturedPoints.count - checkCount)
        for i in startIndex..<capturedPoints.count {
            let distance = length(capturedPoints[i] - point)
            if distance < distanceThreshold { return true }
        }
        return false
    }
    
    private func getColorFromImage(_ image: CVPixelBuffer, at point: CGPoint) -> SIMD3<Float>? {
        let width = CVPixelBufferGetWidth(image)
        let height = CVPixelBufferGetHeight(image)
        let x = Int(point.x * Float(width) / Float(CVPixelBufferGetWidth(image)))
        let y = Int(point.y * Float(height) / Float(CVPixelBufferGetHeight(image)))
        
        if x < 0 || x >= width || y < 0 || y >= height { return nil }
        // Placeholder; actual RGB extraction requires CIImage processing
        return SIMD3<Float>(Float(x) / Float(width), Float(y) / Float(height), 0.5)
    }
    
    private func unprojectPoint(x: Int, y: Int, depth: Float, frame: ARFrame) throws -> SIMD3<Float> {
        let intrinsics = frame.camera.intrinsics
        let imageResolution = frame.camera.imageResolution
        let viewMatrix = frame.camera.viewMatrix(for: .portrait)
        
        let pixelX = Float(x) - Float(imageResolution.width) * 0.5
        let pixelY = Float(y) - Float(imageResolution.height) * 0.5
        let fx = intrinsics[0, 0]
        let fy = intrinsics[1, 1]
        let cx = intrinsics[2, 0]
        let cy = intrinsics[2, 1]
        
        let normalizedX = (pixelX + cx) / fx
        let normalizedY = (pixelY + cy) / fy
        let point = SIMD3<Float>(normalizedX * depth, normalizedY * depth, depth)
        
        let transform = simd_mul(viewMatrix.inverse, simd_float4x4(translation: point))
        return SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
    }
    
    private func outer(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> simd_float3x3 {
        return simd_float3x3(
            SIMD3<Float>(a.x * b.x, a.x * b.y, a.x * b.z),
            SIMD3<Float>(a.y * b.x, a.y * b.y, a.y * b.z),
            SIMD3<Float>(a.z * b.x, a.z * b.y, a.z * b.z)
        )
    }
    
    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if state == .scanning {
            do {
                try captureDepthPoints(frame: frame)
            } catch {
                DispatchQueue.main.async {
                    self.state = .failed(error)
                    self.statusMessage = "Capture failed: \(error.localizedDescription)"
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
            self.statusMessage = "Session interruption ended. Restart scanning."
        }
    }
    
    // MARK: - MTKViewDelegate
    func draw(in view: MTKView) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
              let drawable = view.currentDrawable else { return }
        
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(normalBuffer, offset: 0, index: 1)
        renderEncoder.drawIndexedPrimitives(type: .triangle, indexCount: triangles.count, indexType: .uint32, indexBuffer: indexBuffer, indexBufferOffset: 0)
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize if needed
    }
    
    // MARK: - Visualization and Guidance
    private func showScanningGuidance() {
        guard let arView = arView else { return }
        let guidanceEntity = ModelEntity(mesh: .generateSphere(radius: 0.1), materials: [SimpleMaterial(color: .green, isMetallic: false)])
        let anchor = AnchorEntity(world: .zero)
        anchor.addChild(guidanceEntity)
        arView.scene.addAnchor(anchor)
    }
    
    private var meshVisualization: ModelEntity?
    private var meshAnchor: AnchorEntity?
    
    private func setupMeshVisualization() {
        guard let arView = arView else { return }
        meshAnchor = AnchorEntity(world: .zero)
        arView.scene.addAnchor(meshAnchor!)
    }
    
    private func updateMeshVisualization() {
        guard let meshAnchor = meshAnchor, !capturedPoints.isEmpty else { return }
        if let oldMesh = meshVisualization {
            meshAnchor.removeChild(oldMesh)
        }
        
        var meshDescriptor = MeshDescriptor()
        meshDescriptor.positions = MeshBuffers.Positions(capturedPoints)
        if capturedPoints.count > 10 {
            var triangles: [UInt32] = []
            for i in 0..<min(capturedPoints.count - 2, 2000) {
                if i % 2 == 0, i > 0 {
                    triangles.append(UInt32(i))
                    triangles.append(UInt32(i - 1))
                    triangles.append(UInt32(i - 2))
                }
            }
            if !triangles.isEmpty {
                meshDescriptor.primitives = .triangles(triangles)
            }
        }
        
        do {
            let mesh = try MeshResource.generate(from: [meshDescriptor])
            let material = SimpleMaterial(color: .red.withAlphaComponent(0.7), isMetallic: false, roughness: 1.0)
            meshVisualization = ModelEntity(mesh: mesh, materials: [material])
            meshAnchor.addChild(meshVisualization!)
            DispatchQueue.main.async {
                self.pointCount = self.capturedPoints.count
            }
        } catch {
            print("Failed to generate preview mesh: \(error)")
        }
    }
    
    private func addMeshToARView() {
        guard let arView = arView, let meshData = MeshProcessor.shared.meshData else { return }
        var meshDescriptor = MeshDescriptor()
        meshDescriptor.positions = MeshBuffers.Positions(meshData.vertices)
        meshDescriptor.normals = MeshBuffers.Normals(meshData.normals)
        meshDescriptor.primitives = .triangles(meshData.triangles)
        
        do {
            let mesh = try MeshResource.generate(from: [meshDescriptor])
            let material = SimpleMaterial(color: .blue, isMetallic: false, roughness: 0.5)
            let modelEntity = ModelEntity(mesh: mesh, materials: [material])
            let anchorEntity = AnchorEntity(world: SIMD3<Float>(0, 0, 0))
            anchorEntity.addChild(modelEntity)
            arView.scene.addAnchor(anchorEntity)
        } catch {
            print("Failed to add mesh to ARView: \(error)")
        }
    }
    
    private func updateScanningProgress() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.scanProgress = min(0.8, Float(self.capturedPoints.count) / Float(self.maxPoints))
            if self.capturedPoints.count % 1000 == 0 {
                self.statusMessage = "Captured \(self.capturedPoints.count) points"
            }
            self.updateMeshVisualization()
        }
    }
    
    private func resetScanningState() {
        capturedPoints.removeAll()
        capturedNormals.removeAll()
        pointConfidences.removeAll()
        colors.removeAll()
        frameCount = 0
        scannedMesh = nil
        if let anchor = meshAnchor {
            arView?.scene.removeAnchor(anchor)
            meshAnchor = nil
            meshVisualization = nil
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
        vertexBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<SIMD3<Float>>.size, options: .storageModeShared)
        normalBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<SIMD3<Float>>.size, options: .storageModeShared)
        indexBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<UInt32>.size, options: .storageModeShared)
    }
}