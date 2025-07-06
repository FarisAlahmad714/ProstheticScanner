import SwiftUI
import RealityKit
import ARKit
import Combine

// Container for AR view that integrates with ScanningManager
struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var scanningManager = ScanningManager.shared

    func makeUIView(context: Context) -> ARView {
        // Create AR view
        let arView = ARView(frame: .zero)
        
        // Configure the AR view
        setupARView(arView)
        
        // Set up the scanning manager
        scanningManager.setup(arView: arView)
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        // Update the AR view based on scanning state
        if scanningManager.isScanning {
            // Ensure AR session is running
            if uiView.session.configuration == nil {
                setupARView(uiView)
            }
        } else {
            // Optionally pause the session when not scanning
            // uiView.session.pause()
        }
    }
    
    private func setupARView(_ arView: ARView) {
        // Configure AR view with appropriate settings
        arView.automaticallyConfigureSession = false
        
        // Enable debug options in development builds
        #if DEBUG
        arView.debugOptions = [.showSceneUnderstanding]
        #endif
        
        // Safely configure environment - only using supported features
        // Avoid using .personSegmentation which isn't available
        arView.environment.sceneUnderstanding.options = [.occlusion, .physics]
        
        // Configure rendering options
        arView.renderOptions = [.disableMotionBlur, .disableDepthOfField]
        
        // Set up coaching overlay for better user guidance
        addCoachingOverlay(to: arView)
    }
    
    private func addCoachingOverlay(to arView: ARView) {
        let coachingOverlay = ARCoachingOverlayView()
        coachingOverlay.frame = arView.bounds
        coachingOverlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        
        // Use a well-defined goal enum value
        coachingOverlay.goal = .anyPlane  // Valid enum value
        
        // Set session
        coachingOverlay.session = arView.session
        
        // Add to view
        arView.addSubview(coachingOverlay)
    }
}
