//
//  ProcessingVizualization.swift
//  ProstheticScanner
//
//  Created by Faris Alahmad on 11/9/24.
//

import SwiftUI

struct ProcessingView: View {
    @ObservedObject var meshProcessor: MeshProcessor
    var onProcessingComplete: (MeshData) -> Void
    
    // Visual states for animations
    @State private var showProgress = false
    @State private var rotationAngle = 0.0
    
    var body: some View {
        ZStack {
            // Background
            Color.black
                .opacity(0.9)
                .edgesIgnoringSafeArea(.all)
            
            VStack(spacing: 30) {
                Spacer()
                
                // Processing animation
                ZStack {
                    Circle()
                        .stroke(lineWidth: 4)
                        .opacity(0.3)
                        .foregroundColor(.blue)
                        .frame(width: 120, height: 120)
                    
                    Circle()
                        .trim(from: 0, to: CGFloat(meshProcessor.processingProgress))
                        .stroke(style: StrokeStyle(
                            lineWidth: 4,
                            lineCap: .round
                        ))
                        .foregroundColor(.blue)
                        .rotationEffect(Angle(degrees: -90))
                        .frame(width: 120, height: 120)
                        .rotationEffect(.degrees(rotationAngle))
                        .onAppear {
                            withAnimation(Animation.linear(duration: 2).repeatForever(autoreverses: false)) {
                                rotationAngle = 360
                            }
                        }
                }
                
                // Progress information
                VStack(spacing: 15) {
                    Text("\(Int(meshProcessor.processingProgress * 100))%")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundColor(.white)
                        .opacity(showProgress ? 1 : 0)
                    
                    Text(meshProcessor.processingMessage)
                        .font(.headline)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                        .opacity(showProgress ? 1 : 0)
                    
                    if meshProcessor.vertexCount > 0 {
                        VStack(spacing: 8) {
                            Text("Vertices: \(meshProcessor.vertexCount)")
                            Text("Triangles: \(meshProcessor.triangleCount)")
                        }
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .opacity(showProgress ? 1 : 0)
                    }
                }
                
                Spacer()
                
                // Processing tips
                VStack(spacing: 10) {
                    Text("Processing Tips:")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Text("• Keep the app open during processing\n• Processing may take several minutes\n• Larger scans take longer to process")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                }
                .padding()
                .opacity(showProgress ? 1 : 0)
            }
            .padding()
        }
        .onAppear {
            withAnimation(.easeIn(duration: 0.5)) {
                showProgress = true
            }
        }
        // Monitor processing state
        .onChange(of: meshProcessor.isProcessing) {
            if !meshProcessor.isProcessing, let meshData = meshProcessor.meshData {
                onProcessingComplete(meshData)
            }
        }
    }
}
