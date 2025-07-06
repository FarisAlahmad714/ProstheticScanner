import SwiftUI

struct GuideView: View {
    @Binding var showGuide: Bool
    
    var body: some View {
        VStack(spacing: 20) {
            Text("3D Prosthetic Scanner")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Image(systemName: "scanner.fill")
                .font(.system(size: 60))
                .foregroundColor(.blue)
                .padding()
            
            VStack(alignment: .leading, spacing: 15) {
                GuideStep(number: 1, text: "Position the device 30-50cm from the limb")
                GuideStep(number: 2, text: "Move slowly around the entire limb to capture all surfaces")
                GuideStep(number: 3, text: "Maintain consistent lighting and avoid fast movements")
                GuideStep(number: 4, text: "Complete the scan when you've covered the entire area")
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(10)
            
            Button(action: {
                showGuide = false
            }) {
                Text("Start Scanning")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
            }
            .padding(.horizontal, 40)
            .padding(.top, 20)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(20)
        .shadow(radius: 10)
        .padding()
    }
}

struct GuideStep: View {
    let number: Int
    let text: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 15) {
            Text("\(number)")
                .font(.headline)
                .foregroundColor(.white)
                .frame(width: 26, height: 26)
                .background(Color.blue)
                .clipShape(Circle())
            
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// Preview provider for testing in Xcode
struct GuideView_Previews: PreviewProvider {
    static var previews: some View {
        GuideView(showGuide: .constant(true))
            .preferredColorScheme(.dark)
    }
}
