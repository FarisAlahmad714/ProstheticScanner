
import SwiftUI

struct GuideView: View {
    @Binding var showGuide: Bool
    @State private var currentStep = 1
    
    var body: some View {
        ZStack {
            Color.black.opacity(0.8) // Dark background
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Text("3D Scanning Guide")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .padding()
                
                TabView(selection: $currentStep) {
                    GuideStepView(
                        step: 1,
                        title: "Prepare the Environment",
                        description: "Ensure good lighting and remove any reflective or moving objects from the scene.",
                        imageName: "lightbulb.fill"
                    )
                    .tag(1)
                    
                    GuideStepView(
                        step: 2,
                        title: "Position the Device",
                        description: "Hold your device steady and aim at the object you want to scan. Keep a distance of about 1-2 feet.",
                        imageName: "viewfinder"
                    )
                    .tag(2)
                    
                    GuideStepView(
                        step: 3,
                        title: "Start Scanning",
                        description: "Tap 'Start Scanning' and slowly move around the object. Try to capture all angles.",
                        imageName: "camera.fill"
                    )
                    .tag(3)
                    
                    GuideStepView(
                        step: 4,
                        title: "Complete the Scan",
                        description: "Once you've captured all angles, tap 'Stop Scanning' and wait for processing to complete.",
                        imageName: "checkmark.circle.fill"
                    )
                    .tag(4)
                }
                .tabViewStyle(PageTabViewStyle(indexDisplayMode: .always))
                .padding(.bottom)
                
                // Progress indicators
                HStack(spacing: 8) {
                    ForEach(1...4, id: \.self) { step in
                        Circle()
                            .fill(currentStep == step ? Color.blue : Color.gray)
                            .frame(width: 8, height: 8)
                    }
                }
                .padding(.bottom)
                
                Button(action: {
                    if currentStep < 4 {
                        withAnimation {
                            currentStep += 1
                        }
                    } else {
                        showGuide = false
                    }
                }) {
                    Text(currentStep < 4 ? "Next" : "Start Scanning")
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                .padding(.horizontal)
                .padding(.bottom, 30)
            }
        }
    }
}
struct GuideStepView: View {
    let step: Int
    let title: String
    let description: String
    let imageName: String
    
    var body: some View {
        VStack(spacing: 30) {
            Spacer()
            
            Image(systemName: imageName)
                .resizable()
                .scaledToFit()
                .frame(height: 80)
                .foregroundColor(.blue)
                .padding(.top)
            
            VStack(spacing: 20) {
                Text("Step \(step): \(title)")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                
                Text(description)
                    .font(.body)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding()
            
            Spacer()
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.clear)
    }
}

// Preview provider for testing in Xcode
struct GuideView_Previews: PreviewProvider {
    static var previews: some View {
        GuideView(showGuide: .constant(true))
            .preferredColorScheme(.dark)
    }
}
