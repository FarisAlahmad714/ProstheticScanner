//
//  MeshProcessor.swift
//  ProstheticScanner
//
//  Created by Faris Alahmad on 11/9/24.
//
import Foundation
import simd


// MARK: - Supporting Types
// MARK: - Core MeshProcessor Class
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
    private let syncQueue = DispatchQueue(label: "com.prostheticscanner.meshsync", attributes: .concurrent)
    
    // Processing Properties
    private var points: [SIMD3<Float>] = []
    private var normals: [SIMD3<Float>] = []
    private var confidences: [Float] = []
    private var colors: [SIMD3<Float>] = []
    private var triangles: [UInt32] = []
    private var vertexIndices: [Int: Int] = [:]
    private var octreeNodes: [Octree] = []
    private let nodePool: NSMutableArray = []
    private var activeNodes: Set<Int> = []
    
    // MARK: - Processing Constants
    private let voxelSize: Float = 0.03
    private let maxOctreeDepth = 8
    private let solverIterations = 300
    private let maxParallelTasks = 4
    private let smoothingIterations = 3
    private let adaptiveThreshold: Float = 0.01
    private let memoryChunkSize = 500
    private let processingTimeoutInterval: TimeInterval = 180.0
    private let maxProcessingBatchSize = 500
    private let batchSize = 2000
    private let isoValue: Float = 0.5
    private let poissonDepth = 5
    private let samplesPerNode = 8
    private let minPointsPerNode = 5
    private let maxSolverIterations = 50
    private let convergenceThreshold: Float = 1e-6
    private let minTrianglesThreshold = 100
    
    func reset() {
        // Reset mesh processing-related properties and data
        isProcessing = false
        processingProgress = 0.0
        processingMessage = ""
        vertexCount = 0
        triangleCount = 0
        meshData = nil
        points.removeAll()
        normals.removeAll()
        confidences.removeAll()
        colors.removeAll()
        triangles.removeAll()
        vertexIndices.removeAll()
        octreeNodes.removeAll()
        activeNodes.removeAll()
        print("MeshProcessor has been reset.")
    }
    private func generateTriangles() -> [UInt32] {
        guard points.count >= 3 else { return [] }
        
        var triangles: [UInt32] = []
        let maxDistance: Float = voxelSize * 2.0  // Maximum distance between connected vertices
        
        // For each point, find nearby points to form triangles
        for i in 0..<points.count {
            let p1 = points[i]
            
            // Find closest points to p1
            let neighbors = findKNearestNeighbors(forPoint: p1, k: 8)
            
            // Create triangles from neighboring points
            for j in 0..<neighbors.count-1 {
                let p2 = neighbors[j]
                let p2Index = points.firstIndex(of: p2)!
                
                for k in (j+1)..<neighbors.count {
                    let p3 = neighbors[k]
                    let p3Index = points.firstIndex(of: p3)!
                    
                    // Check if points form a valid triangle
                    let d1 = simd_distance(p1, p2)
                    let d2 = simd_distance(p2, p3)
                    let d3 = simd_distance(p3, p1)
                    
                    if d1 < maxDistance && d2 < maxDistance && d3 < maxDistance {
                        // Add triangle (ensure proper winding order)
                        let normal = normalize(cross(p2 - p1, p3 - p1))
                        if dot(normal, normals[i]) > 0 {
                            triangles.append(UInt32(i))
                            triangles.append(UInt32(p2Index))
                            triangles.append(UInt32(p3Index))
                        } else {
                            triangles.append(UInt32(i))
                            triangles.append(UInt32(p3Index))
                            triangles.append(UInt32(p2Index))
                        }
                    }
                }
            }
        }
        
        return triangles
    }
    
    // MARK: - Initialization
    private init() {}
    // MARK: - Public Methods
    func processScanData(_ scanData: ScanData, completion: @escaping (Result<MeshData, MeshError>) -> Void) {
        guard scanData.points.count >= 1000 else {
            completion(.failure(.insufficientPoints))
            return
        }
        
        isProcessing = true
        
        // Copy data
        self.points = scanData.points
        self.normals = scanData.normals
        self.confidences = scanData.confidences
        
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                // Preprocess point cloud
                self.removeOutliers()
                self.normalizePointDistribution()
                self.recomputeNormals()
                
                // Generate mesh
                let meshData = MeshData(
                    vertices: self.points,
                    normals: self.normals,
                    triangles: self.generateTriangles()
                )
                
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
    
        
        // MARK: - Octree Implementation
        private class Octree {
            var center: SIMD3<Float>
            var size: Float
            var children: [Octree?]
            var points: [SIMD3<Float>]
            var normals: [SIMD3<Float>]
            private var _density: Float
            var density: Float {
                get {
                    MeshProcessor.shared.syncQueue.sync { _density }
                }
                set {
                    MeshProcessor.shared.syncQueue.async(flags: .barrier) {
                        self._density = newValue
                    }
                }
            }
            var index: Int
            
            init(center: SIMD3<Float>, size: Float, index: Int) {
                self.center = center
                self.size = size
                self.children = Array(repeating: nil, count: 8)
                self.points = []
                self.normals = []
                self._density = 0
                self.index = index
            }
        }
        
        private struct LaplacianMatrix {
            var rows: [Int]
            var cols: [Int]
            var values: [Float]
            var size: Int
            
            init(size: Int) {
                self.rows = []
                self.cols = []
                self.values = []
                self.size = size
            }
            
            mutating func addEntry(row: Int, col: Int, value: Float) {
                rows.append(row)
                cols.append(col)
                values.append(value)
            }
        }
        
        private struct MarchingCubesTable {
            static let edgeTable: [Int] = [
                0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
                0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
                0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
                0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
                0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
                0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
                0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
                0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0
            ]
            
            static let triTable: [[Int]] = [
                [], [0, 8, 3], [0, 1, 9], [1, 8, 3, 9, 8, 1], [1, 2, 10],
                [0, 8, 3, 1, 2, 10], [9, 2, 10, 0, 2, 9], [2, 8, 3, 2, 10, 8, 10, 9, 8],
                [3, 11, 2], [0, 11, 2, 8, 11, 0], [1, 9, 0, 2, 3, 11],
                [1, 11, 2, 1, 9, 11, 9, 8, 11], [3, 10, 1, 11, 10, 3],
                [0, 10, 1, 0, 8, 10, 8, 11, 10], [3, 9, 0, 3, 11, 9, 11, 10, 9],
                [9, 8, 10, 10, 8, 11]
            ]
        }
    
    private func outer(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> simd_float3x3 {
                   return simd_float3x3(
                       SIMD3<Float>(a.x * b.x, a.x * b.y, a.x * b.z),
                       SIMD3<Float>(a.y * b.x, a.y * b.y, a.y * b.z),
                       SIMD3<Float>(a.z * b.x, a.z * b.y, a.z * b.z)
                   )
               }
    
    // MARK: - Core Processing Methods
    private func computeEigenvalues(_ matrix: simd_float3x3) -> (Float, Float, Float) {
                    let iterations = 10
                    var vector = SIMD3<Float>(1, 1, 1)
                    
                    for _ in 0..<iterations {
                        vector = matrix * vector
                        vector = normalize(vector)
                    }
                    
                    let eigenvalue1 = length(matrix * vector)
                    let projection = matrix - eigenvalue1 * outer(vector, vector)
                    
                    var vector2 = SIMD3<Float>(1, -1, 1)
                    for _ in 0..<iterations {
                        vector2 = projection * vector2
                        vector2 = normalize(vector2)
                    }
                    
                    let eigenvalue2 = length(projection * vector2)
                    let eigenvalue3 = matrix.determinant / (eigenvalue1 * eigenvalue2)
                    
                    return (eigenvalue1, eigenvalue2, eigenvalue3)
                }
    
    
    private func estimateNormal(_ localPoints: [SIMD3<Float>]) -> SIMD3<Float> {
                   guard localPoints.count > 3 else { return SIMD3<Float>(0, 0, 1) }
                   let centroid = localPoints.reduce(SIMD3<Float>(0,0,0), +) / Float(localPoints.count)
                   let covariance = calculateCovarianceMatrix(points: localPoints, centroid: centroid)
                   let eigenvalues = computeEigenvalues(covariance)
                   return normalize(SIMD3<Float>(eigenvalues.0, eigenvalues.1, eigenvalues.2))
               }
    
    private func calculateCovarianceMatrix(points: [SIMD3<Float>], centroid: SIMD3<Float>) -> simd_float3x3 {
                    var covariance = simd_float3x3()
                    
                    for point in points {
                        let diff = point - centroid
                        covariance[0][0] += diff.x * diff.x
                        covariance[0][1] += diff.x * diff.y
                        covariance[0][2] += diff.x * diff.z
                        covariance[1][0] += diff.y * diff.x
                        covariance[1][1] += diff.y * diff.y
                        covariance[1][2] += diff.y * diff.z
                        covariance[2][0] += diff.z * diff.x
                        covariance[2][1] += diff.z * diff.y
                        covariance[2][2] += diff.z * diff.z
                    }
                    
                    let n = Float(points.count)
                    return covariance * (1.0 / n)
                }
    private func estimateNormalFromNeighbors(_ point: SIMD3<Float>, neighbors: [SIMD3<Float>]) -> SIMD3<Float> {
        guard neighbors.count >= 3 else {
            return SIMD3<Float>(0, 1, 0)
        }
        
        // Calculate centroid
        let centroid = neighbors.reduce(SIMD3<Float>(0, 0, 0), +) / Float(neighbors.count)
        
        // Calculate covariance matrix
        var covariance = matrix_float3x3()
        for neighbor in neighbors {
            let diff = neighbor - centroid
            let scale = 1.0 / Float(neighbors.count)
            covariance.columns.0 += diff * diff.x * scale
            covariance.columns.1 += diff * diff.y * scale
            covariance.columns.2 += diff * diff.z * scale
        }
        
        // Get normal from smallest eigenvector
        let normal = normalize(covariance.columns.0)
        return normal
    }
    
    
    private func findKNearestNeighbors(forPoint point: SIMD3<Float>, k: Int) -> [SIMD3<Float>] {
        let sortedPoints = points.enumerated()
            .filter { $0.offset != points.firstIndex(of: point) }
            .sorted { simd_distance_squared(point, points[$0.offset]) < simd_distance_squared(point, points[$1.offset]) }
        return Array(sortedPoints.prefix(k).map { points[$0.offset] })
    }


    private func recomputeNormals() {
        guard points.count >= 3 else { return }
        
        var newNormals: [SIMD3<Float>] = Array(repeating: SIMD3<Float>(0, 0, 1), count: points.count)
        
        DispatchQueue.concurrentPerform(iterations: points.count) { i in
            let kNeighbors = min(10, points.count - 1) // Make sure we don't exceed array bounds
            let neighbors = findKNearestNeighbors(forPoint: points[i], k: kNeighbors)
            if !neighbors.isEmpty {
                let normal = estimateNormalFromNeighbors(points[i], neighbors: neighbors)
                newNormals[i] = normal
            }
        }
        
        normals = newNormals
    }
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
                
                if avgDistance > voxelSize * 3.0 {
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
                    Int(point.x / gridSize),
                    Int(point.y / gridSize),
                    Int(point.z / gridSize)
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

    
    private func insertPointIntoOctree(_ point: SIMD3<Float>) {
        let syncQueue = DispatchQueue(label: "com.scanner.octree")
        var currentNodeIndex = 0
        var depth = 0
        
        while depth < maxOctreeDepth {
            syncQueue.sync {
                guard currentNodeIndex < octreeNodes.count else { return }
                
                let node = octreeNodes[currentNodeIndex]
                
                if node.points.count >= samplesPerNode && depth < maxOctreeDepth - 1 {
                    let childIndex = getChildIndex(point: point, nodeCenter: node.center)
                    
                    if node.children[childIndex] == nil {
                        let childCenter = computeChildCenter(
                            parentCenter: node.center,
                            childIndex: childIndex,
                            size: node.size
                        )
                        
                        let newNode = Octree(
                            center: childCenter,
                            size: node.size * 0.5,
                            index: octreeNodes.count
                        )
                        
                        octreeNodes.append(newNode)
                        node.children[childIndex] = newNode
                        octreeNodes[currentNodeIndex] = node
                        currentNodeIndex = newNode.index
                    } else if let nextNode = node.children[childIndex] {
                        currentNodeIndex = nextNode.index
                    }
                } else {
                    node.points.append(point)
                    octreeNodes[currentNodeIndex] = node
                    depth = maxOctreeDepth // Exit loop
                }
                
                depth += 1
            }
        }
    }

        
        private func buildOctree() -> Bool {
            print("\n=== STARTING OCTREE CONSTRUCTION ===")
            
            let startTime = Date()
            var processedPoints = 0
            
            guard let boundingBox = calculateBoundingBox() else {
                print("Failed to calculate bounding box")
                return false
            }
            
            let center = (boundingBox.0 + boundingBox.1) * 0.5
            let size = max(boundingBox.1.x - boundingBox.0.x,
                          max(boundingBox.1.y - boundingBox.0.y,
                              boundingBox.1.z - boundingBox.0.z))
            
            // Create root node
            octreeNodes.removeAll(keepingCapacity: true)
            let rootNode = Octree(center: center, size: size * 1.1, index: 0)
            octreeNodes = [rootNode]
            
            // Process points in batches
            let insertionQueue = DispatchQueue(label: "com.prosthetic.insertion", qos: .userInitiated)
            
            while processedPoints < points.count {
                insertionQueue.sync {
                    autoreleasepool {
                        let endIndex = min(processedPoints + batchSize, points.count)
                        let batch = Array(points[processedPoints..<endIndex])
                        
                        for point in batch {
                            insertPointIntoOctree(point)
                        }
                        
                        processedPoints = endIndex
                        
                        let progress = Float(processedPoints) / Float(self.points.count)
                        updateProgress(progress, "Building octree... \(Int(progress * 100))%")
                    }
                }
                
                if Date().timeIntervalSince(startTime) > processingTimeoutInterval {
                    print("Octree construction timed out")
                    return false
                }
            }
            
            return true
        }
        
        private func computeDensityFunction() {
            let startTime = Date()
            
            let totalBatches = (octreeNodes.count + batchSize - 1) / batchSize
            let computeQueue = DispatchQueue(label: "com.prosthetic.density.compute",
                                           attributes: .concurrent)
            let group = DispatchGroup()
            let updateQueue = DispatchQueue(label: "com.prosthetic.density.update")
            
            for batchIndex in 0..<totalBatches {
                let start = batchIndex * batchSize
                let end = min(start + batchSize, octreeNodes.count)
                
                computeQueue.async(group: group) {
                    autoreleasepool {
                        for i in start..<end {
                            guard i < self.octreeNodes.count else { continue }
                            
                            let node = self.octreeNodes[i]
                            let totalPoints = node.points.count
                            
                            guard totalPoints > 0 else { continue }
                            
                            var totalDensity: Float = 0.0
                            let searchRadius = self.voxelSize * 2.0
                            let searchRadiusSq = searchRadius * searchRadius
                            
                            for point in node.points {
                                let distanceSq = simd_distance_squared(node.center, point)
                                if distanceSq < searchRadiusSq {
                                    let influence = exp(-distanceSq / (2.0 * self.voxelSize * self.voxelSize))
                                    totalDensity += influence
                                }
                            }
                            
                            let density = totalDensity / Float(totalPoints)
                            
                            updateQueue.async {
                                if i < self.octreeNodes.count {
                                    self.octreeNodes[i].density = density
                                }
                            }
                        }
                    }
                }
                
                let progress = Float(batchIndex) / Float(totalBatches)
                updateProgress(progress, "Computing density function... \(Int(progress * 100))%")
                
                if Date().timeIntervalSince(startTime) > processingTimeoutInterval {
                    print("Density computation timed out")
                    return
                }
            }
            
            group.wait()
        }
    
    private func findOctreeNode(containing point: SIMD3<Float>) -> Octree? {
            var currentNode = octreeNodes.first
            
            while let node = currentNode {
                let halfSize = node.size * 0.5
                let minBound = node.center - SIMD3<Float>(repeating: halfSize)
                let maxBound = node.center + SIMD3<Float>(repeating: halfSize)
                
                // Check if point is in node bounds
                if point.x >= minBound.x && point.x <= maxBound.x &&
                   point.y >= minBound.y && point.y <= maxBound.y &&
                   point.z >= minBound.z && point.z <= maxBound.z {
                    
                    // Check children
                    let childIndex = getChildIndex(point: point, nodeCenter: node.center)
                    if let child = node.children[childIndex] {
                        currentNode = child
                    } else {
                        return node
                    }
                } else {
                    return nil
                }
            }
            
            return nil
        }
    private func findNeighboringNodes(for node: Octree) -> [Octree] {
            var neighbors: [Octree] = []
            let searchRadius = node.size
            
            // Check all nodes that might be neighbors
            for otherNode in octreeNodes {
                if otherNode.index != node.index {
                    let distance = simd_distance(node.center, otherNode.center)
                    if distance < searchRadius + otherNode.size {
                        neighbors.append(otherNode)
                    }
                }
            }
            
            return neighbors
        }
    
    private func findVertexNeighbors(index: Int) -> [SIMD3<Float>] {
            guard index >= 0 && index < points.count else {
                print("Index \(index) out of bounds")
                return []
            }

            let point = points[index]
            var neighbors: [SIMD3<Float>] = []
            let searchRadius = voxelSize * 2.0  // Doubled search radius
            let searchRadiusSq = searchRadius * searchRadius

            // Use spatial partitioning from octree
            if let node = findOctreeNode(containing: point) {
                // Check points in current node and neighboring nodes
                for otherPoint in node.points {
                    let distSq = simd_distance_squared(point, otherPoint)
                    if distSq < searchRadiusSq && distSq > 1e-6 { // Avoid self
                        neighbors.append(otherPoint)
                    }
                }
                
                // Check neighboring nodes
                for neighborNode in findNeighboringNodes(for: node) {
                    for otherPoint in neighborNode.points {
                        let distSq = simd_distance_squared(point, otherPoint)
                        if distSq < searchRadiusSq {
                            neighbors.append(otherPoint)
                        }
                    }
                }
            }

            if neighbors.isEmpty {
                // Fallback to brute force with larger radius if no neighbors found
                let fallbackRadius = voxelSize * 4.0
                let fallbackRadiusSq = fallbackRadius * fallbackRadius
                
                for i in 0..<points.count {
                    if i != index {
                        let distSq = simd_distance_squared(point, points[i])
                        if distSq < fallbackRadiusSq {
                            neighbors.append(points[i])
                        }
                    }
                    
                    // Limit number of neighbors for performance
                    if neighbors.count >= 20 {
                        break
                    }
                }
            }

            return neighbors
        }

   
    private func solvePoissonEquation() -> ([Float], [SIMD3<Float>]) {
           print("\n=== STARTING POISSON EQUATION SOLVER ===")
           
           let numVertices = points.count
           guard numVertices > 0 else {
               return ([], [])
           }
           
            var solution = Array(repeating: Float(0.0), count: numVertices)
            let vertices = points
            var sparseMatrix = Array(repeating: [(Int, Float)](), count: numVertices)
            var diagonal = Array(repeating: Float(0.0), count: numVertices)
            var residual = Array(repeating: Float(0.0), count: numVertices)
            var direction = Array(repeating: Float(0.0), count: numVertices)
            
            // Build matrix in batches
            for startIdx in stride(from: 0, to: numVertices, by: batchSize) {
                autoreleasepool {
                    let endIdx = min(startIdx + batchSize, numVertices)
                    
                    for i in startIdx..<endIdx {
                        let neighbors = findVertexNeighbors(index: i)
                        guard !neighbors.isEmpty else { continue }
                        
                        let weight = -1.0 / Float(max(neighbors.count, 1))
                        
                        for neighbor in neighbors {
                            if let neighborIndex = points.firstIndex(of: neighbor),
                               neighborIndex < numVertices {
                                sparseMatrix[i].append((neighborIndex, weight))
                                diagonal[i] -= weight
                            }
                        }
                        sparseMatrix[i].append((i, 1.0 + diagonal[i]))
                    }
                }
            }
            
            // Solve system
            var iterCount = 0
            while iterCount < maxSolverIterations {
                var alpha: Float = 0.0
                var dAd: Float = 0.0
                var delta: Float = 0.0
                
                for startIdx in stride(from: 0, to: numVertices, by: batchSize) {
                    autoreleasepool {
                        let endIdx = min(startIdx + batchSize, numVertices)
                        
                        for i in startIdx..<endIdx {
                            var ad: Float = 0.0
                            for (j, weight) in sparseMatrix[i] {
                                guard j < numVertices else { continue }
                                ad += weight * direction[j]
                            }
                            dAd += direction[i] * ad
                            delta += residual[i] * residual[i]
                        }
                    }
                }
                
                if abs(dAd) < convergenceThreshold * 10 {
                    break
                }
                
                alpha = delta / (dAd + 1e-10)
                
                for i in 0..<numVertices {
                    solution[i] += alpha * direction[i]
                    var ad: Float = 0.0
                    for (j, weight) in sparseMatrix[i] {
                        guard j < numVertices else { continue }
                        ad += weight * direction[j]
                    }
                    residual[i] -= alpha * ad
                }
                
                let beta = delta / (dAd + 1e-10)
                for i in 0..<numVertices {
                    direction[i] = residual[i] + beta * direction[i]
                }
                
                iterCount += 1
            }
            
            return (solution, vertices)
        }
    // MARK: - Surface Extraction Methods
        private func extractSurface(solution: [Float], vertices: [SIMD3<Float>]) {
            guard let (minBound, maxBound) = calculateBoundingBox() else {
                print("Error: Could not calculate bounding box")
                return
            }
            
            let gridDimension = 32
            let cellSize = simd_distance(maxBound, minBound) / Float(gridDimension)
            let gridSpacing = cellSize * 0.5
            
            var grid = Array(repeating: Array(repeating: Array(repeating: Float(0.0),
                                                              count: gridDimension),
                                            count: gridDimension),
                            count: gridDimension)
            
            let chunkSize = 8
            let numberOfChunks = (gridDimension + chunkSize - 1) / chunkSize
            var processedCells = 0
            let totalCells = gridDimension * gridDimension * gridDimension
            
            let processingQueue = DispatchQueue(label: "com.prosthetic.surface.extraction",
                                              attributes: .concurrent)
            let group = DispatchGroup()
            
            // Process grid in parallel chunks
            for xChunk in 0..<numberOfChunks {
                for yChunk in 0..<numberOfChunks {
                    for zChunk in 0..<numberOfChunks {
                        group.enter()
                        processingQueue.async { [weak self] in
                            guard let self = self else {
                                group.leave()
                                return
                            }
                            
                            self.processGridChunk(xChunk: xChunk, yChunk: yChunk, zChunk: zChunk,
                                                chunkSize: chunkSize, gridDimension: gridDimension,
                                                minBound: minBound, gridSpacing: gridSpacing,
                                                solution: solution, vertices: vertices,
                                                grid: &grid, processedCells: &processedCells,
                                                totalCells: totalCells)
                            group.leave()
                        }
                    }
                }
            }
            
            group.wait()
            
            generateMeshFromGrid(grid: grid, gridDimension: gridDimension,
                               minBound: minBound, gridSpacing: gridSpacing)
        }
        
    
    private func interpolateSolution(solution: [Float],
                                   vertices: [SIMD3<Float>],
                                   point: SIMD3<Float>,
                                   searchRadius: Float) -> Float {
        var interpolatedValue: Float = 0.0
        var totalWeight: Float = 0.0
        let searchRadiusSq = searchRadius * searchRadius
        
        for (index, vertex) in vertices.enumerated() {
            let distSq = simd_distance_squared(vertex, point)
            if distSq < searchRadiusSq {
                let weight = 1.0 / (distSq + 1e-6)
                interpolatedValue += weight * solution[index]
                totalWeight += weight
            }
        }
        
        return totalWeight > 0 ? interpolatedValue / totalWeight : 0.0
    }

    
    
        private func processGridChunk(xChunk: Int, yChunk: Int, zChunk: Int,
                                    chunkSize: Int, gridDimension: Int,
                                    minBound: SIMD3<Float>, gridSpacing: Float,
                                    solution: [Float], vertices: [SIMD3<Float>],
                                    grid: inout [[[Float]]], processedCells: inout Int,
                                    totalCells: Int) {
            autoreleasepool {
                let startX = xChunk * chunkSize
                let startY = yChunk * chunkSize
                let startZ = zChunk * chunkSize
                let endX = min(startX + chunkSize, gridDimension)
                let endY = min(startY + chunkSize, gridDimension)
                let endZ = min(startZ + chunkSize, gridDimension)
                
                for x in startX..<endX {
                    for y in startY..<endY {
                        for z in startZ..<endZ {
                            let position = minBound + SIMD3<Float>(
                                Float(x) * gridSpacing,
                                Float(y) * gridSpacing,
                                Float(z) * gridSpacing
                            )
                            
                            grid[x][y][z] = interpolateSolution(
                                solution: solution,
                                vertices: vertices,
                                point: position,
                                searchRadius: gridSpacing * 1.5
                            )
                            
                            processedCells += 1
                            if processedCells % 10000 == 0 {
                                let progress = Float(processedCells) / Float(totalCells)
                                updateProgress(progress, "Extracting surface... \(Int(progress * 100))%")
                            }
                        }
                    }
                }
            }
        }
    
    
    
    
    private func interpolateVertex(
            _ val1: Float, _ val2: Float,
            _ p1: SIMD3<Float>, _ p2: SIMD3<Float>,
            _ isovalue: Float
        ) -> SIMD3<Float> {
            if abs(isovalue - val1) < 1e-6 { return p1 }
            if abs(isovalue - val2) < 1e-6 { return p2 }
            if abs(val1 - val2) < 1e-6 { return p1 }
            
            let mu = (isovalue - val1) / (val2 - val1)
            return p1 + (p2 - p1) * mu
        }
    
    
    private func marchingCubesVertices(cornerValues: [Float], position: SIMD3<Float>, cellSize: Float) -> [SIMD3<Float>] {
            let isovalue: Float = 0.5
            var vertices: [SIMD3<Float>] = []
            
            // Calculate cube configuration
            var cubeIndex: Int = 0
            for i in 0..<8 {
                if cornerValues[i] < isovalue {
                    cubeIndex |= (1 << i)
                }
            }
            
            // Early exit if no intersections
            if MarchingCubesTable.edgeTable[cubeIndex] == 0 {
                return vertices
            }
            
            var vertList = Array(repeating: SIMD3<Float>(0, 0, 0), count: 12)
            
            // Safely access edge table
            guard cubeIndex < MarchingCubesTable.edgeTable.count else {
                print("⚠️ Invalid cube index: \(cubeIndex)")
                return vertices
            }
            
            // Create edge lookup table for vertex positions
            let edgeToVertices: [(SIMD3<Float>, SIMD3<Float>)] = [
                (SIMD3<Float>(0,0,0), SIMD3<Float>(cellSize,0,0)),    // Edge 0
                (SIMD3<Float>(cellSize,0,0), SIMD3<Float>(cellSize,cellSize,0)),  // Edge 1
                (SIMD3<Float>(cellSize,cellSize,0), SIMD3<Float>(0,cellSize,0)),  // Edge 2
                (SIMD3<Float>(0,cellSize,0), SIMD3<Float>(0,0,0)),    // Edge 3
                (SIMD3<Float>(0,0,cellSize), SIMD3<Float>(cellSize,0,cellSize)),  // Edge 4
                (SIMD3<Float>(cellSize,0,cellSize), SIMD3<Float>(cellSize,cellSize,cellSize)),  // Edge 5
                (SIMD3<Float>(cellSize,cellSize,cellSize), SIMD3<Float>(0,cellSize,cellSize)),  // Edge 6
                (SIMD3<Float>(0,cellSize,cellSize), SIMD3<Float>(0,0,cellSize)),  // Edge 7
                (SIMD3<Float>(0,0,0), SIMD3<Float>(0,0,cellSize)),    // Edge 8
                (SIMD3<Float>(cellSize,0,0), SIMD3<Float>(cellSize,0,cellSize)),  // Edge 9
                (SIMD3<Float>(cellSize,cellSize,0), SIMD3<Float>(cellSize,cellSize,cellSize)),  // Edge 10
                (SIMD3<Float>(0,cellSize,0), SIMD3<Float>(0,cellSize,cellSize))   // Edge 11
            ]
            
            let edgeTable = MarchingCubesTable.edgeTable[cubeIndex]
            
            // Calculate intersection vertices
            for edge in 0..<12 {
                if (edgeTable & (1 << edge)) != 0 {
                    let (v1, v2) = edgeToVertices[edge]
                    let p1 = position + v1
                    let p2 = position + v2
                    
                    let idx1 = edge
                    let idx2 = (edge + 1) % 8
                    
                    guard idx1 < cornerValues.count && idx2 < cornerValues.count else {
                        continue
                    }
                    
                    let val1 = cornerValues[idx1]
                    let val2 = cornerValues[idx2]
                    
                    vertList[edge] = interpolateVertex(
                        val1, val2,
                        p1, p2,
                        isovalue
                    )
                }
            }
            
            // Generate triangles using triTable
            guard cubeIndex < MarchingCubesTable.triTable.count else {
                return vertices
            }
            
            let triTableEntry = MarchingCubesTable.triTable[cubeIndex]
            
            for i in stride(from: 0, to: triTableEntry.count, by: 3) {
                guard i + 2 < triTableEntry.count else { break }
                
                let index1 = triTableEntry[i]
                let index2 = triTableEntry[i + 1]
                let index3 = triTableEntry[i + 2]
                
                guard index1 < vertList.count &&
                      index2 < vertList.count &&
                      index3 < vertList.count else {
                    continue
                }
                
                vertices.append(vertList[index1])
                vertices.append(vertList[index2])
                vertices.append(vertList[index3])
            }
            
            return vertices
        }
        
        private func generateMeshFromGrid(grid: [[[Float]]], gridDimension: Int,
                                        minBound: SIMD3<Float>, gridSpacing: Float) {
            var meshVertices: [SIMD3<Float>] = []
            var meshTriangles: [UInt32] = []
            
            for x in 0..<(gridDimension - 1) {
                for y in 0..<(gridDimension - 1) {
                    for z in 0..<(gridDimension - 1) {
                        let cornerValues = [
                            grid[x][y][z],
                            grid[x+1][y][z],
                            grid[x+1][y+1][z],
                            grid[x][y+1][z],
                            grid[x][y][z+1],
                            grid[x+1][y][z+1],
                            grid[x+1][y+1][z+1],
                            grid[x][y+1][z+1]
                        ]
                        
                        let cellPosition = minBound + SIMD3<Float>(
                            Float(x) * gridSpacing,
                            Float(y) * gridSpacing,
                            Float(z) * gridSpacing
                        )
                        
                        let cellVertices = marchingCubesVertices(
                            cornerValues: cornerValues,
                            position: cellPosition,
                            cellSize: gridSpacing
                        )
                        
                        addCellVerticesToMesh(cellVertices: cellVertices,
                                            meshVertices: &meshVertices,
                                            meshTriangles: &meshTriangles)
                    }
                }
            }
            
            points = meshVertices
            triangles = meshTriangles
            recomputeNormals()
        }
    private func addCellVerticesToMesh(cellVertices: [SIMD3<Float>], meshVertices: inout [SIMD3<Float>], meshTriangles: inout [UInt32]) {
        // Add each vertex in `cellVertices` to `meshVertices`, and track indices for triangle creation.
        for vertex in cellVertices {
            meshVertices.append(vertex)
        }
        
        // Generate triangles based on `cellVertices` indices.
        let startIndex = UInt32(meshVertices.count - cellVertices.count)
        for i in stride(from: 0, to: cellVertices.count, by: 3) {
            guard i + 2 < cellVertices.count else { break }
            meshTriangles.append(startIndex + UInt32(i))
            meshTriangles.append(startIndex + UInt32(i + 1))
            meshTriangles.append(startIndex + UInt32(i + 2))
        }
    }

    
        // MARK: - Mesh Optimization Methods
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
            
            points = newPoints
            normals = newNormals
            triangles = newIndices
        }
        
        // MARK: - Utility Methods
        private func updateProgress(_ progress: Float, _ message: String) {
            DispatchQueue.main.async { [weak self] in
                self?.processingProgress = progress
                self?.processingMessage = message
            }
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
        
        private func lerp(start: SIMD3<Float>, end: SIMD3<Float>, t: Float) -> SIMD3<Float> {
            return start + (end - start) * t
        }
        
        private func optimizeMemoryUsage() {
            autoreleasepool {
                for i in stride(from: 0, to: points.count, by: memoryChunkSize) {
                    let endIndex = min(i + memoryChunkSize, points.count)
                    let chunk = Array(points[i..<endIndex])
                    processPointChunk(chunk)
                }
            }
        }
    private func findLocalPoints(around point: SIMD3<Float>) -> [SIMD3<Float>] {
        // Define the search radius as twice the voxel size
        let searchRadius = voxelSize * 2.0

        // Use a temporary array to collect points within the search radius
        var localPoints: [SIMD3<Float>] = []

        // Iterate over `points` array efficiently using stride to minimize memory overhead
        for candidatePoint in points {
            // Calculate the distance only if it's necessary, avoiding redundant calculations
            let distanceSquared = simd_distance_squared(point, candidatePoint)

            // Check if the point is within the specified search radius
            if distanceSquared < (searchRadius * searchRadius) {
                localPoints.append(candidatePoint)
            }

            // Optional: Early exit if too many points are found (e.g., for performance)
            if localPoints.count > maxProcessingBatchSize {
                break
            }
        }

        return localPoints
    }
    
    private func updateVoxel(at key: SIMD3<Int>, point: SIMD3<Float>, normal: SIMD3<Float>) {
                   let confidence: Float = 1.0
                   let color = SIMD3<Float>(0.5, 0.5, 0.5)
                   
                   let voxelKey = key
                   if let existing = vertexIndices[voxelKey.hashValue] {
                       if confidence > confidences[existing] {
                           points[existing] = point
                           normals[existing] = normal
                           confidences[existing] = confidence
                           colors[existing] = color
                       }
                   } else {
                       let newIndex = points.count
                       vertexIndices[voxelKey.hashValue] = newIndex
                       points.append(point)
                       normals.append(normal)
                       confidences.append(confidence)
                       colors.append(color)
                   }
               }
    
    
    private func processPointChunk(_ points: [SIMD3<Float>]) {
                for point in points {
                    autoreleasepool {
                        processPoint(point)
                    }
                }
            }
            
            private func processPoint(_ point: SIMD3<Float>) {
                let voxelKey = SIMD3<Int>(
                    Int(point.x / voxelSize),
                    Int(point.y / voxelSize),
                    Int(point.z / voxelSize)
                )
                
                autoreleasepool {
                    let localPoints = findLocalPoints(around: point)
                    if !localPoints.isEmpty {
                        let normal = estimateNormal(localPoints)
                        updateVoxel(at: voxelKey, point: point, normal: normal)
                    }
                }
            }
        
        // Add remaining utility methods here...
    }
