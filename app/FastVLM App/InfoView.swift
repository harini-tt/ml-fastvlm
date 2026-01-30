//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

struct InfoView: View {
    @Environment(\.dismiss) var dismiss

    let paragraph1 = "Zing AI is an on-device ML research assistant. Point your camera at figures, graphs, and tables from machine learning papers to get instant analysis."
    let paragraph2 = "Features:\n• Runs entirely on-device using Apple Silicon\n• References your indexed paper library for context\n• Saves insights to persistent memory across sessions\n• Uses efficient vision-language models via MLX"
    let footer = "Powered by FastVLM (CVPR 2025) and MLX."

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 20.0) {
                // I'm not going to lie, this doesn't make sense...
                // Wrapping `String`s with `.init()` turns them into `LocalizedStringKey`s
                // which gives us all of the fun Markdown formatting while retaining the
                // ability to use `String` variables. ¯\_(ツ)_/¯
                Text("\(.init(paragraph1))\n\n\(.init(paragraph2))\n\n")
                    .font(.body)

                Spacer()

                Text(.init(footer))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .textSelection(.enabled)
            .navigationTitle("Information")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                #if os(iOS)
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle")
                            .resizable()
                            .frame(width: 25, height: 25)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                #elseif os(macOS)
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                    .buttonStyle(.bordered)
                }
                #endif
            }
        }
    }
}

#Preview {
    InfoView()
}
