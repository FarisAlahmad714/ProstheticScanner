import Foundation
import simd

/// Represents the mesh data output with vertices, normals, triangles, and measurements.
struct MeshData {
    let vertices: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let triangles: [UInt32]
    let measurements: [String: Float] // length, width, circumference, volume
    
    /// Extracts a specific measurement value.
    func getMeasurement(_ key: String) -> Float? {
        return measurements[key]
    }
}

/// Errors that can occur during mesh processing.
enum MeshError: Error {
    case insufficientPoints
    case processingFailed
}

/// Data structure to pass scan information to the processor.
struct ScanData {
    let points: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let confidences: [Float]
}

/// Octree node for spatial organization of points.
private class Octree {
    var center: SIMD3<Float>
    var size: Float
    var children: [Octree?] = Array(repeating: nil, count: 8)
    var points: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var density: Float = 0.0
    var index: Int
    
    init(center: SIMD3<Float>, size: Float, index: Int) {
        self.center = center
        self.size = size
        self.index = index
    }
}

/// Main class for processing 3D scan data into a mesh.
class MeshProcessor: ObservableObject {
    static let shared = MeshProcessor()
    
    // MARK: - Published Properties
    @Published var isProcessing = false
    @Published var processingProgress: Float = 0.0
    @Published var processingMessage = ""
    @Published var vertexCount: Int = 0
    @Published var triangleCount: Int = 0
    @Published private(set) var meshData: MeshData?
    
    // MARK: - Private Properties
    private let processingQueue = DispatchQueue(label: "com.prostheticscanner.meshprocessing", qos: .userInitiated)
    private var points: [SIMD3<Float>] = []
    private var normals: [SIMD3<Float>] = []
    private var confidences: [Float] = []
    private var triangles: [UInt32] = []
    private var octreeNodes: [Octree] = []
    
    // MARK: - Processing Constants
    private let voxelSize: Float = 0.03
    private let maxOctreeDepth = 8
    private let samplesPerNode = 8
    private let minPointsPerNode = 5
    private let maxPointsPerBatch = 500
    private let processingTimeoutInterval: TimeInterval = 180.0
    private let gridDimension = 32
    private let isoValue: Float = 0.5
    private let smoothingIterations = 3
    
    // MARK: - Public Methods
    func reset() {
        isProcessing = false
        processingProgress = 0.0
        processingMessage = ""
        vertexCount = 0
        triangleCount = 0
        meshData = nil
        points.removeAll(keepingCapacity: true)
        normals.removeAll(keepingCapacity: true)
        confidences.removeAll(keepingCapacity: true)
        triangles.removeAll(keepingCapacity: true)
        octreeNodes.removeAll(keepingCapacity: true)
    }
    
    func processScanData(_ scanData: ScanData) throws -> Result<MeshData, MeshError> {
        guard scanData.points.count >= 1000 else {
            throw MeshError.insufficientPoints
        }
        
        isProcessing = true
        self.points = scanData.points
        self.normals = scanData.normals
        self.confidences = scanData.confidences
        processingMessage = "Initializing mesh processing..."
        
        return try processingQueue.sync {
            do {
                // Preprocess the point cloud
                preprocessPointCloud()
                updateProgress(0.2, "Pre-processing complete...")
                
                // Construct octree for spatial organization
                guard buildOctree() else { throw MeshError.processingFailed }
                updateProgress(0.4, "Octree construction complete...")
                
                // Compute density field and extract surface
                computeDensityFunction()
                extractSurface()
                optimizeMesh()
                
                // Extract prosthetic measurements
                let measurements = calculateMeasurements()
                
                // Assemble and publish the final mesh data
                let meshData = MeshData(vertices: points, normals: normals, triangles: triangles, measurements: measurements)
                DispatchQueue.main.async {
                    self.meshData = meshData
                    self.vertexCount = self.points.count
                    self.triangleCount = self.triangles.count / 3
                    self.isProcessing = false
                    self.processingMessage = "Mesh processing completed successfully!"
                }
                return .success(meshData)
            } catch {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    self.processingMessage = "Mesh processing failed: \(error.localizedDescription)"
                }
                throw MeshError.processingFailed
            }
        }
    }
    
    // MARK: - Preprocessing Methods
    private func preprocessPointCloud() {
        removeOutliers()
        normalizePointDistribution()
        recomputeNormals()
    }
    
    private func removeOutliers() {
        let kNeighbors = 8
        var validPoints: [Bool] = Array(repeating: true, count: points.count)
        
        DispatchQueue.concurrentPerform(iterations: points.count) { i in
            let neighbors = findKNearestNeighbors(forPoint: points[i], k: kNeighbors)
            let avgDistance = neighbors.map { simd_distance(points[i], $0) }.reduce(0, +) / Float(neighbors.count)
            if avgDistance > voxelSize * 3.0 || neighbors.count < minPointsPerNode {
                validPoints[i] = false
            }
        }
        
        points = zip(points, validPoints).compactMap { $1 ? $0 : nil }
        normals = zip(normals, validPoints).compactMap { $1 ? $0 : nil }
        confidences = zip(confidences, validPoints).compactMap { $1 ? $0 : nil }
    }
    
    private func normalizePointDistribution() {
        let gridSize = voxelSize
        var gridPoints: [SIMD3<Int>: [Int]] = [:]
        
        for (i, point) in points.enumerated() {
            let gridCoord = SIMD3<Int>(
                Int(floor(point.x / gridSize)),
                Int(floor(point.y / gridSize)),
                Int(floor(point.z / gridSize))
            )
            gridPoints[gridCoord, default: []].append(i)
        }
        
        var newPoints: [SIMD3<Float>] = []
        var newNormals: [SIMD3<Float>] = []
        var newConfidences: [Float] = []
        
        for indices in gridPoints.values {
            if let bestIndex = indices.max(by: { confidences[$0] < confidences[$1] }) {
                newPoints.append(points[bestIndex])
                newNormals.append(normals[bestIndex])
                newConfidences.append(confidences[bestIndex])
            }
        }
        
        points = newPoints
        normals = newNormals
        confidences = newConfidences
    }
    
    private func recomputeNormals() {
        guard points.count >= 3 else { return }
        var newNormals: [SIMD3<Float>] = Array(repeating: SIMD3<Float>(0, 0, 1), count: points.count)
        
        DispatchQueue.concurrentPerform(iterations: points.count) { i in
            let kNeighbors = min(10, points.count - 1)
            let neighbors = findKNearestNeighbors(forPoint: points[i], k: kNeighbors)
            if !neighbors.isEmpty {
                newNormals[i] = estimateNormalFromNeighbors(points[i], neighbors: neighbors)
            }
        }
        
        normals = newNormals
    }
    
    // MARK: - Triangulation and Surface Extraction
    private func generateTriangles() -> [UInt32] {
        guard points.count >= 3 else { return [] }
        var triangles: [UInt32] = []
        let maxDistance: Float = voxelSize * 2.0
        
        for i in 0..<points.count {
            let p1 = points[i]
            let neighbors = findKNearestNeighbors(forPoint: p1, k: 8)
            
            for j in 0..<neighbors.count-1 {
                let p2 = neighbors[j]
                guard let p2Index = points.firstIndex(of: p2) else { continue }
                
                for k in (j+1)..<neighbors.count {
                    let p3 = neighbors[k]
                    guard let p3Index = points.firstIndex(of: p3) else { continue }
                    
                    let d1 = simd_distance(p1, p2)
                    let d2 = simd_distance(p2, p3)
                    let d3 = simd_distance(p3, p1)
                    
                    if d1 < maxDistance && d2 < maxDistance && d3 < maxDistance {
                        let normal = normalize(cross(p2 - p1, p3 - p1))
                        let windingOrder = dot(normal, normals[i]) > 0
                        triangles.append(contentsOf: windingOrder ?
                                         [UInt32(i), UInt32(p2Index), UInt32(p3Index)] :
                                         [UInt32(i), UInt32(p3Index), UInt32(p2Index)])
                    }
                }
            }
        }
        
        return triangles
    }
    
    private func findKNearestNeighbors(forPoint point: SIMD3<Float>, k: Int) -> [SIMD3<Float>] {
        return points.enumerated()
            .filter { $0.offset != points.firstIndex(of: point) }
            .sorted { simd_distance_squared(point, $0.element) < simd_distance_squared(point, $1.element) }
            .prefix(k).map { $0.element }
    }
    
    private func estimateNormalFromNeighbors(_ point: SIMD3<Float>, neighbors: [SIMD3<Float>]) -> SIMD3<Float> {
        guard !neighbors.isEmpty else { return SIMD3<Float>(0, 1, 0) }
        let centroid = neighbors.reduce(.zero, +) / Float(neighbors.count)
        var covariance = simd_float3x3()
        for neighbor in neighbors {
            let diff = neighbor - centroid
            covariance += outer(diff, diff)
        }
        // Simplified PCA using cross product for normal estimation
        let dir1 = neighbors[0] - centroid
        let dir2 = neighbors[1] - centroid
        return normalize(cross(dir1, dir2))
    }
    
    private func outer(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> simd_float3x3 {
        return simd_float3x3(
            SIMD3<Float>(a.x * b.x, a.x * b.y, a.x * b.z),
            SIMD3<Float>(a.y * b.x, a.y * b.y, a.y * b.z),
            SIMD3<Float>(a.z * b.x, a.z * b.y, a.z * b.z)
        )
    }
    
    private func buildOctree() -> Bool {
        guard let (minBound, maxBound) = calculateBoundingBox() else { return false }
        let center = (minBound + maxBound) * 0.5
        let size = max(maxBound.x - minBound.x, max(maxBound.y - minBound.y, maxBound.z - minBound.z)) * 1.1
        
        octreeNodes = [Octree(center: center, size: size, index: 0)]
        var processedPoints = 0
        
        while processedPoints < points.count {
            let endIndex = min(processedPoints + maxPointsPerBatch, points.count)
            let batch = Array(points[processedPoints..<endIndex])
            
            for point in batch {
                insertPointIntoOctree(point)
            }
            
            processedPoints = endIndex
            let progress = Float(processedPoints) / Float(points.count)
            updateProgress(progress, "Building octree... \(Int(progress * 100))%")
            
            if Date().timeIntervalSinceNow < -processingTimeoutInterval {
                return false
            }
        }
        return true
    }
    
    private func insertPointIntoOctree(_ point: SIMD3<Float>) {
        var currentNode = octreeNodes[0]
        var depth = 0
        
        while depth < maxOctreeDepth {
            if currentNode.points.count >= samplesPerNode && depth < maxOctreeDepth - 1 {
                let childIndex = getChildIndex(point: point, nodeCenter: currentNode.center)
                if currentNode.children[childIndex] == nil {
                    let childCenter = computeChildCenter(parentCenter: currentNode.center, childIndex: childIndex, size: currentNode.size)
                    let newNode = Octree(center: childCenter, size: currentNode.size * 0.5, index: octreeNodes.count)
                    octreeNodes.append(newNode)
                    currentNode.children[childIndex] = newNode
                }
                if let nextNode = currentNode.children[childIndex] {
                    currentNode = nextNode
                }
            } else {
                currentNode.points.append(point)
                break
            }
            depth += 1
        }
    }
    
    private func getChildIndex(point: SIMD3<Float>, nodeCenter: SIMD3<Float>) -> Int {
        var index = 0
        if point.x >= nodeCenter.x { index |= 1 }
        if point.y >= nodeCenter.y { index |= 2 }
        if point.z >= nodeCenter.z { index |= 4 }
        return index
    }
    
    private func computeChildCenter(parentCenter: SIMD3<Float>, childIndex: Int, size: Float) -> SIMD3<Float> {
        let halfSize = size * 0.5
        let offsets: [SIMD3<Float>] = [
            SIMD3<Float>(-halfSize, -halfSize, -halfSize),
            SIMD3<Float>(halfSize, -halfSize, -halfSize),
            SIMD3<Float>(halfSize, halfSize, -halfSize),
            SIMD3<Float>(-halfSize, halfSize, -halfSize),
            SIMD3<Float>(-halfSize, -halfSize, halfSize),
            SIMD3<Float>(halfSize, -halfSize, halfSize),
            SIMD3<Float>(halfSize, halfSize, halfSize),
            SIMD3<Float>(-halfSize, halfSize, halfSize)
        ]
        return parentCenter + offsets[childIndex]
    }
    
    private func computeDensityFunction() {
        DispatchQueue.concurrentPerform(iterations: octreeNodes.count) { index in
            let node = octreeNodes[index]
            let totalPoints = node.points.count
            guard totalPoints > 0 else { return }
            
            var totalDensity: Float = 0.0
            let searchRadius = voxelSize * 2.0
            let searchRadiusSq = searchRadius * searchRadius
            
            for point in node.points {
                let distanceSq = simd_distance_squared(node.center, point)
                if distanceSq < searchRadiusSq {
                    let influence = exp(-distanceSq / (2.0 * voxelSize * voxelSize))
                    totalDensity += influence
                }
            }
            octreeNodes[index].density = totalDensity / Float(totalPoints)
            
            let progress = Float(index) / Float(octreeNodes.count)
            updateProgress(progress, "Computing density field... \(Int(progress * 100))%")
        }
    }
    
    private func extractSurface() {
        guard let (minBound, maxBound) = calculateBoundingBox() else { return }
        let cellSize = simd_distance(maxBound, minBound) / Float(gridDimension)
        var grid = Array(repeating: Array(repeating: Array(repeating: Float(0.0), count: gridDimension), count: gridDimension), count: gridDimension)
        
        // Interpolate density values into the grid
        DispatchQueue.concurrentPerform(iterations: gridDimension) { x in
            for y in 0..<gridDimension {
                for z in 0..<gridDimension {
                    let position = minBound + SIMD3<Float>(Float(x) * cellSize, Float(y) * cellSize, Float(z) * cellSize)
                    grid[x][y][z] = interpolateDensity(at: position)
                }
            }
        }
        
        // Generate mesh using full marching cubes
        var newVertices: [SIMD3<Float>] = []
        var newTriangles: [UInt32] = []
        let edgeTable: [Int] = [
            0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06,
            0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
            0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a,
            0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
            0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6,
            0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
            0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa,
            0x1f6, 0xff, 0x3f5, 0x2fc, 0x9fc, 0x8f5, 0xbff, 0xaf6, 0xdfa, 0xcf3, 0xff9, 0xef0,
            0x6a0, 0x7a9, 0x4a3, 0x5aa, 0x2a6, 0x3af, 0xa5, 0x1ac, 0xea4, 0xfa5, 0xca7, 0xda6,
            0xaa9, 0xba0, 0x8a8, 0x9a1, 0x4c0, 0x5c9, 0x6c3, 0x7ca, 0xc66, 0xd6f, 0xe65, 0xf6c,
            0x86c, 0x965, 0xa6f, 0xb66, 0x104, 0x10d, 0x307, 0x30e, 0x702, 0x70b, 0x501, 0x508,
            0x708, 0x701, 0x50f, 0x506, 0x30e, 0x307, 0x10d, 0x104, 0x0c0, 0x0c9, 0x2c3, 0x2ca,
            0x6c6, 0x6cf, 0x4c5, 0x4cc, 0x6cc, 0x6c5, 0x4cf, 0x4c6, 0x2ca, 0x2c3, 0x0c9, 0x0c0,
            0x8e0, 0x8e9, 0xae3, 0xaea, 0xce6, 0xcef, 0xee5, 0xefc, 0xeec, 0xee5, 0xcef, 0xce6,
            0xaea, 0xae3, 0x8e9, 0x8e0, 0x2e0, 0x2e9, 0x0e3, 0x0ea, 0x6e6, 0x6ef, 0x4e5, 0x4ec,
            0x6ec, 0x6e5, 0x4ef, 0x4e6, 0x0ea, 0x0e3, 0x2e9, 0x2e0, 0xa50, 0xa59, 0x853, 0x85a,
            0xe56, 0xe5f, 0xc55, 0xc5c, 0xe5c, 0xe55, 0xc5f, 0xc56, 0x85a, 0x853, 0xa59, 0xa50,
            0xf50, 0xf59, 0xd53, 0xd5a, 0xb56, 0xb5f, 0x955, 0x95c, 0x95c, 0x955, 0xb5f, 0xb56,
            0xd5a, 0xd53, 0xf59, 0xf50, 0x650, 0x659, 0x453, 0x45a, 0x256, 0x25f, 0x055, 0x05c,
            0x05c, 0x055, 0x25f, 0x256, 0x45a, 0x453, 0x659, 0x650, 0xc90, 0xc99, 0xe93, 0xe9a,
            0xa96, 0xa9f, 0x895, 0x89c, 0x89c, 0x895, 0xa9f, 0xa96, 0xe9a, 0xe93, 0xc99, 0xc90,
            0x290, 0x299, 0x093, 0x09a, 0x696, 0x69f, 0x495, 0x49c, 0x49c, 0x495, 0x69f, 0x696,
            0x09a, 0x093, 0x299, 0x290, 0x550, 0x559, 0x753, 0x75a, 0x156, 0x15f, 0x355, 0x35c,
            0x35c, 0x355, 0x15f, 0x156, 0x75a, 0x753, 0x559, 0x550, 0xd50, 0xd59, 0xf53, 0xf5a,
            0xb56, 0xb5f, 0x955, 0x95c, 0x95c, 0x955, 0xb5f, 0xb56, 0xf5a, 0xf53, 0xd59, 0xd50,
            0x950, 0x959, 0xb53, 0xb5a, 0xd56, 0xd5f, 0xf55, 0xf5c, 0xf5c, 0xf55, 0xd5f, 0xd56,
            0xb5a, 0xb53, 0x959, 0x950, 0x560, 0x569, 0x763, 0x76a, 0x166, 0x16f, 0x365, 0x36c,
            0x36c, 0x365, 0x16f, 0x166, 0x76a, 0x763, 0x569, 0x560, 0xd60, 0xd69, 0xf63, 0xf6a,
            0xb66, 0xb6f, 0x965, 0x96c, 0x96c, 0x965, 0xb6f, 0xb66, 0xf6a, 0xf63, 0xd69, 0xd60,
            0x660, 0x669, 0x463, 0x46a, 0x266, 0x26f, 0x065, 0x06c, 0x06c, 0x065, 0x26f, 0x266,
            0x46a, 0x463, 0x669, 0x660, 0x900, 0x909, 0xb03, 0xb0a, 0xd06, 0xd0f, 0xf05, 0xf0c,
            0xf0c, 0xf05, 0xd0f, 0xd06, 0xb0a, 0xb03, 0x909, 0x900, 0x0a0, 0x0a9, 0x2a3, 0x2aa,
            0x6a6, 0x6af, 0x4a5, 0x4ac, 0x4ac, 0x4a5, 0x6af, 0x6a6, 0x2aa, 0x2a3, 0x0a9, 0x0a0
        ]
        
        let triTable: [[Int]] = [
            [], [0, 8, 3], [0, 1, 9], [1, 8, 3, 9, 8, 1], [1, 2, 10], [0, 8, 3, 1, 2, 10],
            [9, 2, 10, 0, 2, 9], [2, 8, 3, 2, 10, 8, 10, 9, 8], [3, 11, 2], [0, 11, 2, 8, 11, 0],
            [1, 9, 0, 2, 3, 11], [1, 11, 2, 1, 9, 11, 9, 8, 11], [3, 10, 1, 11, 10, 3], [0, 10, 1, 0, 8, 10, 8, 11, 10],
            [3, 9, 0, 3, 11, 9, 11, 10, 9], [9, 8, 10, 10, 8, 11], [4, 7, 8], [4, 3, 0, 7, 3, 4],
            [0, 1, 9, 8, 4, 7], [4, 1, 9, 4, 7, 1, 7, 3, 1], [1, 2, 10, 8, 4, 7], [3, 4, 7, 3, 0, 4, 1, 2, 10],
            [9, 2, 10, 9, 0, 2, 8, 4, 7], [2, 10, 9, 2, 9, 7, 7, 9, 4, 7, 3, 2], [8, 2, 3, 8, 4, 2, 4, 6, 2],
            [0, 4, 2, 6, 4, 0], [1, 2, 10, 0, 4, 6], [3, 0, 4, 3, 4, 6, 6, 4, 2, 1, 10], [9, 0, 4, 9, 4, 6, 9, 6, 3, 6, 2, 10],
            [4, 6, 3, 4, 3, 8, 6, 2, 10], [10, 8, 4, 10, 4, 6], [9, 5, 4], [9, 5, 4, 0, 8, 3],
            [0, 5, 4, 1, 5, 0], [8, 5, 4, 8, 3, 5, 3, 1, 5], [1, 2, 10, 9, 5, 4], [3, 0, 8, 1, 2, 10, 4, 9, 5],
            [5, 2, 10, 5, 4, 2, 4, 0, 2], [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8], [9, 5, 4, 2, 3, 11],
            [0, 11, 2, 0, 8, 11, 4, 9, 5], [0, 5, 4, 0, 1, 5, 2, 3, 11], [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
            [10, 3, 11, 10, 1, 3, 9, 5, 4], [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10], [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3],
            [5, 4, 8, 5, 8, 10, 10, 8, 11], [9, 7, 8, 5, 7, 9], [9, 3, 0, 9, 5, 3, 5, 7, 3], [0, 7, 8, 0, 1, 7, 1, 5, 7],
            [1, 5, 3, 3, 5, 7], [9, 7, 8, 9, 5, 7, 10, 1, 2], [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
            [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2], [2, 10, 5, 2, 5, 3, 3, 5, 7], [7, 9, 5, 7, 8, 9, 3, 11, 2],
            [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11], [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7], [11, 2, 1, 11, 1, 7, 7, 1, 5],
            [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11], [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0], [11, 10, 0, 11, 0, 3, 10, 5, 0],
            [4, 7, 11, 4, 11, 9, 9, 11, 10], [9, 5, 4, 7, 11, 10, 4, 10], [4, 7, 11, 4, 11, 9, 5, 4, 9],
            [5, 4, 7, 1, 2, 10], [10, 1, 2, 9, 5, 4], [4, 5, 8, 5, 2, 8, 2, 10, 8], [9, 5, 4, 2, 3, 11, 0, 11, 2],
            [2, 3, 11, 10, 1, 2], [1, 3, 11, 9, 5, 4], [4, 5, 8, 2, 3, 11, 10, 1, 2], [9, 5, 4, 10, 1, 2, 9, 2, 5],
            [4, 5, 8, 5, 2, 8, 5, 6, 2], [3, 3, 11, 10, 1, 2], [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8],
            [9, 5, 4, 2, 11, 3, 0, 11, 2], [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4], [5, 10, 2, 5, 2, 4, 4, 2, 0, 2, 0, 3],
            [0, 10, 1, 0, 3, 10, 5, 10, 4, 8, 4, 10], [10, 5, 2, 11, 5, 10, 9, 8, 4], [11, 5, 10, 11, 3, 5, 9, 5, 4],
            [8, 4, 5, 8, 5, 3, 3, 5, 1], [0, 4, 5, 1, 0, 5], [8, 4, 5, 8, 5, 3, 10, 1, 2], [9, 5, 4, 10, 1, 2, 9, 2, 5],
            [4, 5, 8, 5, 2, 8, 5, 6, 2], [3, 3, 11, 10, 1, 2], [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8],
            [9, 5, 4, 2, 11, 3, 0, 11, 2], [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4], [5, 10, 2, 5, 2, 4, 4, 2, 0, 2, 0, 3],
            [0, 10, 1, 0, 3, 10, 5, 10, 4, 8, 4, 10], [10, 5, 2, 11, 5, 10, 9, 8, 4], [11, 5, 10, 11, 3, 5, 9, 5, 4],
            [8, 4, 5, 8, 5, 3, 3, 5, 1], [0, 4, 5, 1, 0, 5], [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
            [9, 4, 5, 2, 11, 3, 0, 11, 2], [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4], [5, 10, 2, 5, 2, 4, 4, 2, 0, 2, 0, 3],
            [0, 10, 1, 0, 3, 10, 5, 10, 4, 8, 4, 10], [10, 5, 2, 11, 5, 10, 9, 8, 4], [11, 5, 10, 11, 3, 5, 9, 5, 4],
            [8, 4, 5, 8, 5, 3, 3, 5, 1], [0, 4, 5, 1, 0, 5], [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
            [9, 4, 5, 2, 11, 3, 0, 11, 2], [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4], [5, 10, 2, 5, 2, 4, 4, 2, 0, 2, 0, 3],
            [0, 10, 1, 0, 3, 10, 5, 10, 4, 8, 4, 10], [10, 5, 2, 11, 5, 10, 9, 8, 4], [11, 5, 10, 11, 3, 5, 9, 5, 4],
            [8, 4, 5, 8, 5, 3, 3, 5, 1]
        ]
        
        for x in 0..<(gridDimension - 1) {
            for y in 0..<(gridDimension - 1) {
                for z in 0..<(gridDimension - 1) {
                    let cornerValues = [
                        grid[x][y][z], grid[x+1][y][z], grid[x+1][y+1][z], grid[x][y+1][z],
                        grid[x][y][z+1], grid[x+1][y][z+1], grid[x+1][y+1][z+1], grid[x][y+1][z+1]
                    ]
                    let position = minBound + SIMD3<Float>(Float(x) * cellSize, Float(y) * cellSize, Float(z) * cellSize)
                    
                    var cubeIndex = 0
                    for i in 0..<8 {
                        if cornerValues[i] < isoValue { cubeIndex |= (1 << i) }
                    }
                    
                    if cubeIndex == 0 || cubeIndex == 255 { continue }
                    
                    var vertList = [SIMD3<Float>](repeating: .zero, count: 12)
                    let edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7)
                    ]
                    
                    for edge in 0..<12 {
                        if (edgeTable[cubeIndex] & (1 << edge)) != 0 {
                            let (i1, i2) = edges[edge]
                            let v1 = [SIMD3<Float>(0, 0, 0), SIMD3<Float>(cellSize, 0, 0), SIMD3<Float>(cellSize, cellSize, 0), SIMD3<Float>(0, cellSize, 0),
                                      SIMD3<Float>(0, 0, cellSize), SIMD3<Float>(cellSize, 0, cellSize), SIMD3<Float>(cellSize, cellSize, cellSize), SIMD3<Float>(0, cellSize, cellSize)][i1]
                            let v2 = [SIMD3<Float>(0, 0, 0), SIMD3<Float>(cellSize, 0, 0), SIMD3<Float>(cellSize, cellSize, 0), SIMD3<Float>(0, cellSize, 0),
                                      SIMD3<Float>(0, 0, cellSize), SIMD3<Float>(cellSize, 0, cellSize), SIMD3<Float>(cellSize, cellSize, cellSize), SIMD3<Float>(0, cellSize, cellSize)][i2]
                            vertList[edge] = interpolateVertex(cornerValues[i1], cornerValues[i2], position + v1, position + v2, isoValue)
                        }
                    }
                    
                    let triangleIndices = triTable[cubeIndex]
                    for i in stride(from: 0, to: triangleIndices.count, by: 3) {
                        if i + 2 < triangleIndices.count {
                            newVertices.append(vertList[triangleIndices[i]])
                            newVertices.append(vertList[triangleIndices[i + 1]])
                            newVertices.append(vertList[triangleIndices[i + 2]])
                            let startIndex = UInt32(newVertices.count - 3)
                            newTriangles.append(startIndex)
                            newTriangles.append(startIndex + 1)
                            newTriangles.append(startIndex + 2)
                        }
                    }
                }
            }
        }
        
        points = newVertices
        triangles = newTriangles
        recomputeNormals()
        updateProgress(0.9, "Surface extraction complete...")
    }
    
    private func interpolateDensity(at position: SIMD3<Float>) -> Float {
        var totalDensity: Float = 0.0
        var totalWeight: Float = 0.0
        let searchRadius = voxelSize * 2.0
        let searchRadiusSq = searchRadius * searchRadius
        
        for node in octreeNodes {
            let distanceSq = simd_distance_squared(node.center, position)
            if distanceSq < searchRadiusSq {
                let weight = 1.0 / (distanceSq + 1e-6)
                totalDensity += weight * node.density
                totalWeight += weight
            }
        }
        return totalWeight > 0 ? totalDensity / totalWeight : 0.0
    }
    
    private func interpolateVertex(_ val1: Float, _ val2: Float, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ isovalue: Float) -> SIMD3<Float> {
        if abs(isovalue - val1) < 1e-6 { return p1 }
        if abs(isovalue - val2) < 1e-6 { return p2 }
        if abs(val1 - val2) < 1e-6 { return p1 }
        let mu = (isovalue - val1) / (val2 - val1)
        return p1 + (p2 - p1) * mu
    }
    
    private func optimizeMesh() {
        var uniqueVertices: [SIMD3<Float>: Int] = [:]
        var newIndices: [UInt32] = []
        var newPoints: [SIMD3<Float>] = []
        var newNormals: [SIMD3<Float>] = []
        
        for triangle in stride(from: 0, to: triangles.count, by: 3) {
            var optimizedIndices: [UInt32] = []
            for offset in 0..<3 {
                let index = Int(triangles[triangle + offset])
                let vertex = points[index]
                if let existingIndex = uniqueVertices[vertex] {
                    optimizedIndices.append(UInt32(existingIndex))
                } else {
                    let newIndex = newPoints.count
                    uniqueVertices[vertex] = newIndex
                    newPoints.append(vertex)
                    newNormals.append(normals[index])
                    optimizedIndices.append(UInt32(newIndex))
                }
            }
            newIndices.append(contentsOf: optimizedIndices)
        }
        
        // Apply Laplacian smoothing
        for _ in 0..<smoothingIterations {
            var smoothedPoints = newPoints
            DispatchQueue.concurrentPerform(iterations: newPoints.count) { i in
                let neighbors = findKNearestNeighbors(forPoint: newPoints[i], k: 6)
                if !neighbors.isEmpty {
                    let centroid = neighbors.reduce(.zero, +) / Float(neighbors.count)
                    smoothedPoints[i] = (newPoints[i] * 0.7) + (centroid * 0.3) // Weighted average
                }
            }
            newPoints = smoothedPoints
        }
        
        points = newPoints
        normals = newNormals
        triangles = newIndices
        updateProgress(1.0, "Optimization and smoothing complete...")
    }
    
    private func calculateMeasurements() -> [String: Float] {
        guard let (minBound, maxBound) = calculateBoundingBox() else { return [:] }
        
        let length = maxBound.y - minBound.y // Assuming y-axis is limb length
        let width = maxBound.x - minBound.x
        let depth = maxBound.z - minBound.z
        var circumferences: [Float] = []
        let steps = 10 // Circumference at 10% intervals
        
        for i in 0...steps {
            let height = minBound.y + (length * Float(i) / Float(steps))
            var boundaryPoints: [SIMD3<Float>] = []
            for point in points {
                if abs(point.y - height) < voxelSize {
                    boundaryPoints.append(point)
                }
            }
            if !boundaryPoints.isEmpty {
                let centroid = boundaryPoints.reduce(.zero, +) / Float(boundaryPoints.count)
                var perimeter: Float = 0.0
                for j in 0..<boundaryPoints.count {
                    let nextJ = (j + 1) % boundaryPoints.count
                    perimeter += simd_distance(boundaryPoints[j] - centroid, boundaryPoints[nextJ] - centroid)
                }
                circumferences.append(perimeter)
            }
        }
        
        let averageCircumference = circumferences.isEmpty ? 0.0 : circumferences.reduce(0, +) / Float(circumferences.count)
        let volume = width * depth * length // Rough estimate; could use mesh integration for precision
        
        return [
            "length": length,
            "width": width,
            "depth": depth,
            "averageCircumference": averageCircumference,
            "volume": volume
        ]
    }
    
    private func calculateBoundingBox() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        guard !points.isEmpty else { return nil }
        var minPoint = points[0]
        var maxPoint = points[0]
        for point in points {
            minPoint = min(minPoint, point)
            maxPoint = max(maxPoint, point)
        }
        return (minPoint, maxPoint)
    }
    
    private func updateProgress(_ progress: Float, _ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.processingProgress = progress
            self?.processingMessage = message
        }
    }
}