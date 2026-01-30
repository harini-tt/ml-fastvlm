//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import AVFoundation
import MLXLMCommon
import SwiftUI
import Video

// support swift 6
extension CVImageBuffer: @unchecked @retroactive Sendable {}
extension CMSampleBuffer: @unchecked @retroactive Sendable {}

// delay between frames -- controls the frame rate of the updates
let FRAME_DELAY = Duration.milliseconds(1)

struct ContentView: View {
    @State private var camera = CameraController()
    @State private var model = FastVLMModel()
    @State private var researchContext = ResearchContext()
    @State private var insightStore = InsightStore()

    /// stream of frames -> VideoFrameView, see distributeVideoFrames
    @State private var framesToDisplay: AsyncStream<CVImageBuffer>?

    @State private var prompt = "Describe what this shows. State specific values and trends."
    @State private var promptSuffix = ""
    @State private var useResearchContext = true

    @State private var isShowingInfo: Bool = false
    @State private var isMemoryExpanded: Bool = false
    @State private var expandedInsightId: UUID? = nil

    @State private var selectedCameraType: CameraType = .single
    @State private var isEditingPrompt: Bool = false
    @StateObject private var speech = SpeechRecognizer()

    var toolbarItemPlacement: ToolbarItemPlacement {
        var placement: ToolbarItemPlacement = .navigation
        #if os(iOS)
        placement = .topBarLeading
        #endif
        return placement
    }

    var statusTextColor : Color {
        return model.evaluationState == .processingPrompt ? .black : .white
    }

    var statusBackgroundColor : Color {
        switch model.evaluationState {
        case .idle:
            return .gray
        case .generatingResponse:
            return .green
        case .processingPrompt:
            return .yellow
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 10.0) {
                        if let framesToDisplay {
                            VideoFrameView(
                                frames: framesToDisplay,
                                cameraType: selectedCameraType,
                                action: { frame in
                                    processSingleFrame(frame)
                                })
                                // Because we're using the AVCaptureSession preset
                                // `.vga640x480`, we can assume this aspect ratio
                                .aspectRatio(4/3, contentMode: .fit)
                                #if os(macOS)
                                .frame(maxWidth: 750)
                                #endif
                                .overlay(alignment: .top) {
                                    if !model.promptTime.isEmpty {
                                        Text("TTFT \(model.promptTime)")
                                            .font(.caption)
                                            .foregroundStyle(.white)
                                            .monospaced()
                                            .padding(.vertical, 4.0)
                                            .padding(.horizontal, 6.0)
                                            .background(alignment: .center) {
                                                RoundedRectangle(cornerRadius: 8)
                                                    .fill(Color.black.opacity(0.6))
                                            }
                                            .padding(.top)
                                    }
                                }
                                #if !os(macOS)
                                .overlay(alignment: .topTrailing) {
                                    CameraControlsView(
                                        backCamera: $camera.backCamera,
                                        device: $camera.device,
                                        devices: $camera.devices)
                                    .padding()
                                }
                                #endif
                                .overlay(alignment: .bottom) {
                                    if selectedCameraType == .continuous {
                                        Group {
                                            if model.evaluationState == .processingPrompt {
                                                HStack {
                                                    ProgressView()
                                                        .tint(self.statusTextColor)
                                                        .controlSize(.small)

                                                    Text(model.evaluationState.rawValue)
                                                }
                                            } else if model.evaluationState == .idle {
                                                HStack(spacing: 6.0) {
                                                    Image(systemName: "clock.fill")
                                                        .font(.caption)

                                                    Text(model.evaluationState.rawValue)
                                                }
                                            }
                                            else {
                                                // I'm manually tweaking the spacing to
                                                // better match the spacing with ProgressView
                                                HStack(spacing: 6.0) {
                                                    Image(systemName: "lightbulb.fill")
                                                        .font(.caption)

                                                    Text(model.evaluationState.rawValue)
                                                }
                                            }
                                        }
                                        .foregroundStyle(self.statusTextColor)
                                        .font(.caption)
                                        .bold()
                                        .padding(.vertical, 6.0)
                                        .padding(.horizontal, 8.0)
                                        .background(self.statusBackgroundColor)
                                        .clipShape(.capsule)
                                        .padding(.bottom)
                                    }
                                }
                                #if os(macOS)
                                .frame(maxWidth: .infinity)
                                .frame(minWidth: 500)
                                .frame(minHeight: 375)
                                #endif
                        }
                    }
                }
                .listRowInsets(EdgeInsets())
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)

                promptSections

                memorySection

                Section {
                    if model.output.isEmpty && model.running {
                        ProgressView()
                            .controlSize(.large)
                            .frame(maxWidth: .infinity)
                    } else {
                        ScrollView {
                            Text(model.output)
                                .foregroundStyle(isEditingPrompt ? .secondary : .primary)
                                .textSelection(.enabled)
                                #if os(macOS)
                                .font(.headline)
                                .fontWeight(.regular)
                                #endif
                        }
                        .frame(minHeight: 50.0, maxHeight: 200.0)
                    }
                } header: {
                    HStack {
                        Text("Response")
                        Spacer()
                        if !model.output.isEmpty {
                            Button(action: saveToMemory) {
                                HStack(spacing: 4) {
                                    Image(systemName: "brain")
                                    Text("Save")
                                }
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.purple.opacity(0.2))
                                .foregroundStyle(.purple)
                                .cornerRadius(6)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    #if os(macOS)
                    .font(.headline)
                    .padding(.bottom, 2.0)
                    #endif
                }

                #if os(macOS)
                Spacer()
                #endif
            }

            #if os(iOS)
            .listSectionSpacing(0)
            #elseif os(macOS)
            .padding()
            #endif
            .task {
                camera.start()
            }
            .task {
                await model.load()
                researchContext.load()
            }

            #if !os(macOS)
            .onAppear {
                // Prevent the screen from dimming or sleeping due to inactivity
                UIApplication.shared.isIdleTimerDisabled = true
            }
            .onDisappear {
                // Resumes normal idle timer behavior
                UIApplication.shared.isIdleTimerDisabled = false
            }
            #endif

            // task to distribute video frames -- this will cancel
            // and restart when the view is on/off screen.  note: it is
            // important that this is here (attached to the VideoFrameView)
            // rather than the outer view because this has the correct lifecycle
            .task {
                if Task.isCancelled {
                    return
                }

                await distributeVideoFrames()
            }

            .navigationTitle("Zing AI")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: toolbarItemPlacement) {
                    Button {
                        isShowingInfo.toggle()
                    }
                    label: {
                        Image(systemName: "info.circle")
                    }
                }

                ToolbarItem(placement: .primaryAction) {
                    if isEditingPrompt {
                        Button {
                            isEditingPrompt.toggle()
                        }
                        label: {
                            Text("Done")
                                .fontWeight(.bold)
                        }
                    }
                    else {
                        Button {
                            isEditingPrompt.toggle()
                        } label: {
                            Image(systemName: "pencil.circle")
                                .font(.title2)
                        }
                    }
                }
            }
            .sheet(isPresented: $isShowingInfo) {
                InfoView()
            }
        }
    }

    var promptSummary: some View {
        Section {
            Text(prompt)
                .foregroundStyle(.secondary)
        } header: {
            Text("Question")
        }
    }

    var promptForm: some View {
        Section {
            #if os(iOS)
            HStack(alignment: .top, spacing: 8.0) {
                TextField("Ask a question...", text: $prompt, axis: .vertical)
                    .lineLimit(2...4)

                dictationButton
                .buttonStyle(.borderedProminent)
                .tint(speech.isRecording ? .red : .blue)
                .accessibilityLabel(speech.isRecording ? "Stop dictation" : "Start dictation")
            }
            #else
            TextField("Ask a question...", text: $prompt, axis: .vertical)
                .lineLimit(2...4)
            #endif
        } header: {
            Text("Question")
        }
        #if os(iOS)
        .onChange(of: speech.transcript) { _, newValue in
            if speech.isRecording && !newValue.isEmpty {
                prompt = newValue
            }
        }
        #endif
    }

    #if os(iOS)
    private var dictationButton: some View {
        Button(action: { speech.toggleRecording() }) {
            Image(systemName: speech.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                .font(.title2)
        }
    }
    #endif

    var promptSections: some View {
        Group {
            #if os(iOS)
            if isEditingPrompt {
                promptForm
            }
            else {
                promptSummary
            }
            #elseif os(macOS)
            promptForm
            #endif
        }
    }

    var memorySection: some View {
        Section {
            if insightStore.recentInsights.isEmpty {
                Text("No insights saved yet")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            } else {
                DisclosureGroup(
                    isExpanded: $isMemoryExpanded,
                    content: {
                        ForEach(insightStore.recentInsights.prefix(5)) { insight in
                            insightRow(insight)
                        }
                        .onDelete { indexSet in
                            for index in indexSet {
                                if index < insightStore.recentInsights.count {
                                    insightStore.delete(insightStore.recentInsights[index].id)
                                }
                            }
                        }
                    },
                    label: {
                        HStack {
                            Image(systemName: "brain")
                                .foregroundStyle(.purple)
                            Text("Memory")
                            Spacer()
                            Text("\(insightStore.recentInsights.count) insights")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                )
            }
        } header: {
            Text("Memory")
                #if os(macOS)
                .font(.headline)
                .padding(.bottom, 2.0)
                #endif
        }
    }

    func insightRow(_ insight: Insight) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(relativeTimeString(from: insight.timestamp))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                if !insight.sources.isEmpty {
                    Text("\(insight.sources.count) sources")
                        .font(.caption2)
                        .foregroundStyle(.purple)
                }
            }

            Text(insight.query)
                .font(.caption)
                .foregroundStyle(.primary)
                .lineLimit(expandedInsightId == insight.id ? nil : 1)

            if expandedInsightId == insight.id {
                Text(insight.output)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)

                if !insight.sources.isEmpty {
                    HStack {
                        ForEach(insight.sources, id: \.self) { source in
                            Text(source)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.purple.opacity(0.1))
                                .cornerRadius(4)
                        }
                    }
                    .padding(.top, 2)
                }
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation {
                if expandedInsightId == insight.id {
                    expandedInsightId = nil
                } else {
                    expandedInsightId = insight.id
                }
            }
        }
    }

    func relativeTimeString(from date: Date) -> String {
        let interval = Date().timeIntervalSince(date)

        if interval < 60 {
            return "Just now"
        } else if interval < 3600 {
            let minutes = Int(interval / 60)
            return "\(minutes) min ago"
        } else if interval < 86400 {
            let hours = Int(interval / 3600)
            return "\(hours) hr ago"
        } else {
            let days = Int(interval / 86400)
            return days == 1 ? "Yesterday" : "\(days) days ago"
        }
    }

    func buildPrompt() -> String {
        let basePrompt = "\(prompt) \(promptSuffix)"
        if useResearchContext && researchContext.isLoaded {
            return researchContext.buildContextPrompt(for: prompt, basePrompt: basePrompt)
        }
        return basePrompt
    }

    func saveToMemory() {
        guard !model.output.isEmpty else { return }

        var sources: [String] = []
        if useResearchContext && researchContext.isLoaded {
            let relevantChunks = researchContext.search(query: prompt, topK: 3)
            sources = relevantChunks.map { $0.source.replacingOccurrences(of: ".pdf", with: "") }
        }

        let insight = Insight(
            query: prompt,
            output: model.output,
            sources: sources
        )
        insightStore.save(insight)
    }


    func analyzeVideoFrames(_ frames: AsyncStream<CVImageBuffer>) async {
        for await frame in frames {
            let fullPrompt = await MainActor.run {
                buildPrompt()
            }
            let userInput = UserInput(
                prompt: .text(fullPrompt),
                images: [.ciImage(CIImage(cvPixelBuffer: frame))]
            )

            // generate output for a frame and wait for generation to complete
            let t = await model.generate(userInput)
            _ = await t.result

            do {
                try await Task.sleep(for: FRAME_DELAY)
            } catch { return }
        }
    }

    func distributeVideoFrames() async {
        // attach a stream to the camera -- this code will read this
        let frames = AsyncStream<CMSampleBuffer>(bufferingPolicy: .bufferingNewest(1)) {
            camera.attach(continuation: $0)
        }

        let (framesToDisplay, framesToDisplayContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        self.framesToDisplay = framesToDisplay

        // Only create analysis stream if in continuous mode
        let (framesToAnalyze, framesToAnalyzeContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )

        // set up structured tasks (important -- this means the child tasks
        // are cancelled when the parent is cancelled)
        async let distributeFrames: () = {
            for await sampleBuffer in frames {
                if let frame = sampleBuffer.imageBuffer {
                    framesToDisplayContinuation.yield(frame)
                    // Only send frames for analysis in continuous mode
                    if await selectedCameraType == .continuous {
                        framesToAnalyzeContinuation.yield(frame)
                    }
                }
            }

            // detach from the camera controller and feed to the video view
            await MainActor.run {
                self.framesToDisplay = nil
                self.camera.detatch()
            }

            framesToDisplayContinuation.finish()
            framesToAnalyzeContinuation.finish()
        }()

        // Only analyze frames if in continuous mode
        if selectedCameraType == .continuous {
            async let analyze: () = analyzeVideoFrames(framesToAnalyze)
            await distributeFrames
            await analyze
        } else {
            await distributeFrames
        }
    }

    /// Perform FastVLM inference on a single frame.
    /// - Parameter frame: The frame to analyze.
    func processSingleFrame(_ frame: CVImageBuffer) {
        // Reset Response UI (show spinner)
        Task { @MainActor in
            model.output = ""
        }

        // Construct request to model with context
        let fullPrompt = buildPrompt()
        let userInput = UserInput(
            prompt: .text(fullPrompt),
            images: [.ciImage(CIImage(cvPixelBuffer: frame))]
        )

        // Post request to FastVLM
        Task {
            await model.generate(userInput)
        }
    }
}

#Preview {
    ContentView()
}
