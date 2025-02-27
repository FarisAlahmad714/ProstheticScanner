import SwiftUI

struct ProcessingView: View {
    @ObservedObject var meshProcessor: MeshProcessor = .shared
    let onProcessingComplete: () -> Void

    var body: some View {
        VStack {
            Text(meshProcessor.processingMessage)
                .font(.headline)
            ProgressView(value: meshProcessor.processingProgress)
                .progressViewStyle(LinearProgressViewStyle())
        }
        .onChange(of: meshProcessor.isProcessing) { isProcessing in
            if !isProcessing, meshProcessor.meshData != nil {
                onProcessingComplete()
            }
        }
    }
}