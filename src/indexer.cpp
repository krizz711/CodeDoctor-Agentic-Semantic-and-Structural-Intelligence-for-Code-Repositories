#include "codedoctor/indexer.hpp"

#include "codedoctor/chunker.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace codedoctor {

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string encode(const std::string& value) {
    std::string result;
    for (char ch : value) {
        if (ch == '\\') result += "\\\\";
        else if (ch == '\t') result += "\\t";
        else if (ch == '\n') result += "\\n";
        else result += ch;
    }
    return result;
}

std::string decode(const std::string& value) {
    std::string result;
    bool escaped = false;
    for (char ch : value) {
        if (escaped) {
            result += ch == 'n' ? '\n' : (ch == 't' ? '\t' : ch);
            escaped = false;
        } else if (ch == '\\') {
            escaped = true;
        } else {
            result += ch;
        }
    }
    if (escaped) result += '\\';
    return result;
}

} // namespace

std::string SymbolIndexer::normalizeIdentifier(const std::string& value) {
    std::string result;
    for (char ch : value) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return result;
}

void SymbolIndexer::indexFile(const std::string& file_path, const std::string& source, const std::string& commit_hash) {
    (void)commit_hash;
    // This minimal implementation stores chunks and symbol names in memory.
    // The full repository build is intentionally CPU-light and framework-free for this MVP.
    auto chunked = Chunker::chunkSource(source, file_path);
    chunks_.insert(chunks_.end(), chunked.begin(), chunked.end());

    for (const auto& chunk : chunked) {
        SymbolRecord symbol;
        symbol.chunk_id = chunk.file_path + ":" + std::to_string(chunk.start_byte) + ":" + std::to_string(chunk.end_byte);
        symbol.name = chunk.name;
        symbol.normalized_name = normalizeIdentifier(chunk.name);
        symbol.type = chunk.type;
        symbol.file_path = chunk.file_path;
        symbol.line = chunk.start_line;
        symbols_.push_back(symbol);
    }
}

void SymbolIndexer::clear() {
    chunks_.clear();
    symbols_.clear();
}

bool SymbolIndexer::save(const std::string& index_path) const {
    std::ofstream output(index_path, std::ios::trunc);
    if (!output) return false;
    output << "CODEDOCTOR_INDEX_V1\n";
    for (const auto& chunk : chunks_) {
        output << encode(chunk.file_path) << '\t' << encode(chunk.type) << '\t'
               << encode(chunk.name) << '\t' << encode(chunk.parent) << '\t'
               << chunk.start_byte << '\t' << chunk.end_byte << '\t'
               << chunk.start_line << '\t' << chunk.end_line << '\t'
               << encode(chunk.signature) << '\t' << encode(chunk.content) << '\n';
    }
    return output.good();
}

bool SymbolIndexer::load(const std::string& index_path) {
    std::ifstream input(index_path);
    if (!input) return false;
    std::string line;
    if (!std::getline(input, line) || line != "CODEDOCTOR_INDEX_V1") return false;
    clear();
    while (std::getline(input, line)) {
        std::vector<std::string> fields;
        std::string field;
        bool escaped = false;
        for (char ch : line) {
            if (ch == '\t' && !escaped) {
                fields.push_back(field);
                field.clear();
            } else {
                field += ch;
            }
            if (ch == '\\' && !escaped) escaped = true;
            else escaped = false;
        }
        fields.push_back(field);
        if (fields.size() != 10) continue;
        Chunk chunk;
        chunk.file_path = decode(fields[0]);
        chunk.type = decode(fields[1]);
        chunk.name = decode(fields[2]);
        chunk.parent = decode(fields[3]);
        chunk.start_byte = static_cast<std::size_t>(std::stoull(fields[4]));
        chunk.end_byte = static_cast<std::size_t>(std::stoull(fields[5]));
        chunk.start_line = std::stoi(fields[6]);
        chunk.end_line = std::stoi(fields[7]);
        chunk.signature = decode(fields[8]);
        chunk.content = decode(fields[9]);
        chunks_.push_back(chunk);
        SymbolRecord symbol;
        symbol.chunk_id = chunk.file_path + ":" + std::to_string(chunk.start_byte) + ":" + std::to_string(chunk.end_byte);
        symbol.name = chunk.name;
        symbol.normalized_name = normalizeIdentifier(chunk.name);
        symbol.type = chunk.type;
        symbol.file_path = chunk.file_path;
        symbol.line = chunk.start_line;
        symbols_.push_back(symbol);
    }
    return true;
}

std::vector<SymbolRecord> SymbolIndexer::searchSymbols(const std::string& query) const {
    const std::string normalized = normalizeIdentifier(query);
    std::vector<SymbolRecord> matches;
    for (const auto& symbol : symbols_) {
        if (symbol.normalized_name == normalized) {
            matches.push_back(symbol);
        }
    }
    return matches;
}

std::vector<SearchResult> SymbolIndexer::keywordSearch(const std::string& query) const {
    std::vector<SearchResult> results;
    const std::string lowered = toLower(query);
    for (const auto& chunk : chunks_) {
        const std::string haystack = toLower(chunk.content + " " + chunk.name + " " + chunk.signature);
        if (haystack.find(lowered) != std::string::npos) {
            SearchResult result;
            result.file_path = chunk.file_path;
            result.name = chunk.name;
            result.type = chunk.type;
            result.score = 1.0;
            result.snippet = chunk.content;
            results.push_back(result);
        }
    }
    return results;
}

std::vector<Chunk> SymbolIndexer::chunks() const {
    return chunks_;
}

std::size_t SymbolIndexer::fileCount() const {
    std::set<std::string> files;
    for (const auto& chunk : chunks_) files.insert(chunk.file_path);
    return files.size();
}

} // namespace codedoctor
