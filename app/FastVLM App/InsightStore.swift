//
// InsightStore.swift
// Zing AI
//
// Persistent memory for research insights using SQLite3
//

import Foundation
import SQLite3

// SQLITE_TRANSIENT equivalent for Swift - tells SQLite to make its own copy of the string
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

struct Insight: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let query: String
    let output: String
    let sources: [String]
    let imageHash: String?

    init(id: UUID = UUID(), timestamp: Date = Date(), query: String, output: String, sources: [String] = [], imageHash: String? = nil) {
        self.id = id
        self.timestamp = timestamp
        self.query = query
        self.output = output
        self.sources = sources
        self.imageHash = imageHash
    }
}

@MainActor
class InsightStore: ObservableObject {
    @Published var recentInsights: [Insight] = []

    private var db: OpaquePointer?
    private let dbPath: String

    init() {
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        dbPath = documentsDir.appendingPathComponent("insights.db").path

        openDatabase()
        createTable()
        recentInsights = loadRecent(limit: 20)
    }

    nonisolated deinit {
        sqlite3_close(db)
    }

    private func openDatabase() {
        if sqlite3_open(dbPath, &db) != SQLITE_OK {
            print("InsightStore: Failed to open database at \(dbPath)")
            db = nil
        }
    }

    private func createTable() {
        guard let db = db else { return }

        let createTableSQL = """
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                query TEXT NOT NULL,
                output TEXT NOT NULL,
                sources TEXT,
                image_hash TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_timestamp ON insights(timestamp DESC);
            """

        var errMsg: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, createTableSQL, nil, nil, &errMsg) != SQLITE_OK {
            if let errMsg = errMsg {
                print("InsightStore: Failed to create table: \(String(cString: errMsg))")
                sqlite3_free(errMsg)
            }
        }
    }

    func save(_ insight: Insight) {
        guard let db = db else { return }

        let insertSQL = "INSERT OR REPLACE INTO insights (id, timestamp, query, output, sources, image_hash) VALUES (?, ?, ?, ?, ?, ?);"

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, insertSQL, -1, &stmt, nil) == SQLITE_OK else {
            print("InsightStore: Failed to prepare insert statement")
            return
        }
        defer { sqlite3_finalize(stmt) }

        let idString = insight.id.uuidString
        let timestamp = Int64(insight.timestamp.timeIntervalSince1970)
        let sourcesJSON = (try? JSONEncoder().encode(insight.sources)).flatMap { String(data: $0, encoding: .utf8) } ?? "[]"

        sqlite3_bind_text(stmt, 1, idString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(stmt, 2, timestamp)
        sqlite3_bind_text(stmt, 3, insight.query, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 4, insight.output, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 5, sourcesJSON, -1, SQLITE_TRANSIENT)

        if let imageHash = insight.imageHash {
            sqlite3_bind_text(stmt, 6, imageHash, -1, SQLITE_TRANSIENT)
        } else {
            sqlite3_bind_null(stmt, 6)
        }

        if sqlite3_step(stmt) == SQLITE_DONE {
            // Reload recent insights to update UI
            recentInsights = loadRecent(limit: 20)
        } else {
            print("InsightStore: Failed to insert insight")
        }
    }

    func loadRecent(limit: Int = 20) -> [Insight] {
        guard let db = db else { return [] }

        let querySQL = "SELECT id, timestamp, query, output, sources, image_hash FROM insights ORDER BY timestamp DESC LIMIT ?;"

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, querySQL, -1, &stmt, nil) == SQLITE_OK else {
            print("InsightStore: Failed to prepare select statement")
            return []
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_int(stmt, 1, Int32(limit))

        var insights: [Insight] = []

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let idStr = sqlite3_column_text(stmt, 0),
                  let queryText = sqlite3_column_text(stmt, 2),
                  let outputText = sqlite3_column_text(stmt, 3) else {
                continue
            }

            let id = UUID(uuidString: String(cString: idStr)) ?? UUID()
            let timestamp = Date(timeIntervalSince1970: Double(sqlite3_column_int64(stmt, 1)))
            let query = String(cString: queryText)
            let output = String(cString: outputText)

            var sources: [String] = []
            if let sourcesPtr = sqlite3_column_text(stmt, 4) {
                let sourcesStr = String(cString: sourcesPtr)
                if let data = sourcesStr.data(using: .utf8),
                   let decoded = try? JSONDecoder().decode([String].self, from: data) {
                    sources = decoded
                }
            }

            var imageHash: String? = nil
            if let hashPtr = sqlite3_column_text(stmt, 5) {
                imageHash = String(cString: hashPtr)
            }

            let insight = Insight(id: id, timestamp: timestamp, query: query, output: output, sources: sources, imageHash: imageHash)
            insights.append(insight)
        }

        return insights
    }

    func delete(_ id: UUID) {
        guard let db = db else { return }

        let deleteSQL = "DELETE FROM insights WHERE id = ?;"

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, deleteSQL, -1, &stmt, nil) == SQLITE_OK else {
            print("InsightStore: Failed to prepare delete statement")
            return
        }
        defer { sqlite3_finalize(stmt) }

        let idString = id.uuidString
        sqlite3_bind_text(stmt, 1, idString, -1, SQLITE_TRANSIENT)

        if sqlite3_step(stmt) == SQLITE_DONE {
            recentInsights.removeAll { $0.id == id }
        } else {
            print("InsightStore: Failed to delete insight")
        }
    }

    func search(query: String) -> [Insight] {
        guard let db = db else { return [] }

        let searchSQL = "SELECT id, timestamp, query, output, sources, image_hash FROM insights WHERE query LIKE ? OR output LIKE ? ORDER BY timestamp DESC LIMIT 20;"

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, searchSQL, -1, &stmt, nil) == SQLITE_OK else {
            return []
        }
        defer { sqlite3_finalize(stmt) }

        let searchPattern = "%\(query)%"
        sqlite3_bind_text(stmt, 1, searchPattern, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, searchPattern, -1, SQLITE_TRANSIENT)

        var insights: [Insight] = []

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let idStr = sqlite3_column_text(stmt, 0),
                  let queryText = sqlite3_column_text(stmt, 2),
                  let outputText = sqlite3_column_text(stmt, 3) else {
                continue
            }

            let id = UUID(uuidString: String(cString: idStr)) ?? UUID()
            let timestamp = Date(timeIntervalSince1970: Double(sqlite3_column_int64(stmt, 1)))
            let queryStr = String(cString: queryText)
            let output = String(cString: outputText)

            var sources: [String] = []
            if let sourcesPtr = sqlite3_column_text(stmt, 4) {
                let sourcesStr = String(cString: sourcesPtr)
                if let data = sourcesStr.data(using: .utf8),
                   let decoded = try? JSONDecoder().decode([String].self, from: data) {
                    sources = decoded
                }
            }

            var imageHash: String? = nil
            if let hashPtr = sqlite3_column_text(stmt, 5) {
                imageHash = String(cString: hashPtr)
            }

            let insight = Insight(id: id, timestamp: timestamp, query: queryStr, output: output, sources: sources, imageHash: imageHash)
            insights.append(insight)
        }

        return insights
    }
}
