import Foundation
import Accelerate
import NaturalLanguage

struct PaperChunk: Codable {
    let text: String
    let source: String
    let page: Int
    let embedding: [Float]
}

@MainActor
class ResearchContext: ObservableObject {
    @Published var isLoaded = false
    @Published var chunkCount = 0

    private var chunks: [PaperChunk] = []

    func load() {
        if let url = Bundle.main.url(forResource: "paper_embeddings", withExtension: "json") {
            loadFromURL(url)
        } else {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let url = docs.appendingPathComponent("paper_embeddings.json")
            if FileManager.default.fileExists(atPath: url.path) {
                loadFromURL(url)
            }
        }
    }

    private func loadFromURL(_ url: URL) {
        do {
            let data = try Data(contentsOf: url)
            chunks = try JSONDecoder().decode([PaperChunk].self, from: data)
            chunkCount = chunks.count
            isLoaded = true
            print("Loaded \(chunks.count) paper chunks")
        } catch {
            print("Failed to load embeddings: \(error)")
        }
    }

    func search(query: String, topK: Int = 3) -> [PaperChunk] {
        guard isLoaded, !chunks.isEmpty else { return [] }

        let queryLower = query.lowercased()
        let tagger = NLTagger(tagSchemes: [.lemma])
        tagger.string = queryLower

        var keywords: Set<String> = []
        tagger.enumerateTags(in: queryLower.startIndex..<queryLower.endIndex,
                             unit: .word, scheme: .lemma) { tag, range in
            let word = String(queryLower[range])
            if word.count > 2 {
                keywords.insert(tag?.rawValue ?? word)
                keywords.insert(word)
            }
            return true
        }

        // important ML terms
        let mlTerms = ["rl", "reinforcement", "learning", "ppo", "dpo", "rlhf", "reward",
                       "policy", "training", "loss", "model", "llm", "language", "sft",
                       "grpo", "alignment", "fine-tuning", "optimization"]
        for term in mlTerms {
            if queryLower.contains(term) {
                keywords.insert(term)
            }
        }

        var scored: [(chunk: PaperChunk, score: Float)] = []

        for chunk in chunks {
            let textLower = chunk.text.lowercased()
            var score: Float = 0

            for keyword in keywords {
                if textLower.contains(keyword) {
                    // Weight longer/rarer terms higher
                    score += Float(keyword.count) / 3.0
                }
            }

            if score > 0 {
                scored.append((chunk, score))
            }
        }

        scored.sort { $0.score > $1.score }
        return Array(scored.prefix(topK).map { $0.chunk })
    }

    func searchByEmbedding(_ queryEmbedding: [Float], topK: Int = 3) -> [PaperChunk] {
        guard isLoaded, !chunks.isEmpty else { return [] }

        var scored: [(chunk: PaperChunk, similarity: Float)] = []

        for chunk in chunks {
            let sim = cosineSimilarity(queryEmbedding, chunk.embedding)
            scored.append((chunk, sim))
        }

        scored.sort { $0.similarity > $1.similarity }
        return Array(scored.prefix(topK).map { $0.chunk })
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(a.count))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(b.count))

        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dotProduct / denom : 0
    }

    func buildContextPrompt(for query: String, basePrompt: String) -> String {
        let relevant = search(query: query, topK: 3)

        if relevant.isEmpty {
            return basePrompt
        }

        var context = "Relevant research context:\n"
        for (i, chunk) in relevant.enumerated() {
            context += "\n[\(i+1)] From \(chunk.source) (p.\(chunk.page)):\n"
            context += chunk.text.prefix(400) + "...\n"
        }

        return """
        \(context)

        Based on the above research context and what you see in the image:
        \(basePrompt)
        """
    }
}
