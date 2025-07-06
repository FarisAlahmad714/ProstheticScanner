///  SharedTypes.swift
//  ProstheticScanner
//
//  Created by Faris Alahmad on 11/10/24.
//

import Foundation
import simd

// MARK: - Shared Data Structures

/// Represents the raw data captured during scanning.
struct ScanData {
    let points: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let confidences: [Float]
    let colors: [SIMD3<Float>]
}

/// Represents the processed mesh data.
struct MeshData {
    let vertices: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let triangles: [UInt32]
}

/// Represents surface data used in processing.
struct SurfaceData {
    let vertices: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
}

// MARK: - Error Types

/// Errors that can occur during mesh processing.
enum MeshError: Error {
    case insufficientPoints
    case processingFailed
    case processingTimeout
    case octreeConstructionFailed
    case surfaceReconstructionFailed
    case meshGenerationFailed
    case deviceNotSupported
}

/// Errors that can occur during scanning.
enum ScanError: Error {
    case insufficientPoints
    case processingTimeout
    case meshGenerationFailed
    case deviceNotSupported
}

// MARK: - Support Structures

/// Represents a single vertex with position and normal.
struct Vertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
}

/// Represents a sparse Laplacian matrix for mesh processing.
struct LaplacianMatrix {
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

// MARK: - Constants and Settings

/// Settings used throughout the processing pipeline.
struct ProcessingSettings {
    static let minPoints = 1000
    static let voxelSize: Float = 0.03
    static let maxProcessingTime: TimeInterval = 60
    static let octreeDepth = 8
    static let solverIterations = 300
    static let maxParallelTasks = 4
    static let smoothingIterations = 3
    static let adaptiveThreshold: Float = 0.01
    static let memoryChunkSize = 500
    static let processingTimeoutInterval: TimeInterval = 180.0
    static let maxProcessingBatchSize = 500
    static let batchSize = 2000
    static let isoValue: Float = 0.5
    static let poissonDepth = 5
    static let samplesPerNode = 8
    static let minPointsPerNode = 5
    static let maxSolverIterations = 50
    static let convergenceThreshold: Float = 1e-6
    static let minTrianglesThreshold = 100
}

// MARK: - Lookup Tables

/// Lookup tables for the Marching Cubes algorithm.
struct MarchingCubesTable {
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