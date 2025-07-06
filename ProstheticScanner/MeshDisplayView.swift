<<<<<<< HEAD
import SwiftUI
import SceneKit

struct MeshDisplayView: View {
    let meshData: MeshData

    var body: some View {
        VStack {
            // SceneKit view to render the mesh
            SceneView(
                scene: createScene(),
                options: [.autoenablesDefaultLighting, .allowsCameraControl]
            )
            .frame(height: 300)

            // Display mesh statistics
            Text("Vertices: \(meshData.vertices.count)")
            Text("Triangles: \(meshData.triangles.count / 3)")
        }
    }

    private func createScene() -> SCNScene {
        let scene = SCNScene()
        let geometry = SCNGeometry(
            vertices: meshData.vertices.map { SCNVector3($0.x, $0.y, $0.z) },
            indices: meshData.triangles
        )
        let node = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(node)
        return scene
    }
}
=======
import SwiftUI
import SceneKit

struct MeshDisplayView: View {
    let meshData: MeshData

    var body: some View {
        VStack {
            // SceneKit view to render the mesh
            SceneView(
                scene: createScene(),
                options: [.autoenablesDefaultLighting, .allowsCameraControl]
            )
            .frame(height: 300)

            // Display mesh statistics
            Text("Vertices: \(meshData.vertices.count)")
            Text("Triangles: \(meshData.triangles.count / 3)")
        }
    }

    private func createScene() -> SCNScene {
        let scene = SCNScene()
        
        // Create geometry sources
        let vertexSource = SCNGeometrySource(vertices: meshData.vertices.map {
            SCNVector3($0.x, $0.y, $0.z)
        })
        
        let normalSource = SCNGeometrySource(normals: meshData.normals.map {
            SCNVector3($0.x, $0.y, $0.z)
        })
        
        // Create geometry element for triangles
        let indexCount = meshData.triangles.count
        let indexData = Data(bytes: meshData.triangles, count: indexCount * MemoryLayout<UInt32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indexCount / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )
        
        // Create geometry from sources and elements
        let geometry = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])
        
        // Create and add node
        let node = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(node)
        
        return scene
    }
}
>>>>>>> 57d48373c35c6526189f0514e3f11772729acb9c
