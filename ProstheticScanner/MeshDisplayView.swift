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
        
        // Create vertex sources from mesh data
        let vertexSource = SCNGeometrySource(
            vertices: meshData.vertices.map { SCNVector3($0.x, $0.y, $0.z) }
        )
        
        let normalSource = SCNGeometrySource(
            normals: meshData.normals.map { SCNVector3($0.x, $0.y, $0.z) }
        )
        
        // Create element for indices
        let element = SCNGeometryElement(
            indices: meshData.triangles,
            primitiveType: .triangles
        )
        
        // Create geometry with sources and elements
        let geometry = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])
        
        // Create and add node
        let node = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(node)
        
        return scene
    }
}