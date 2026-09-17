#pragma once

#include <string>
#include <vector>

#include "codedoctor/types.hpp"

namespace codedoctor {

class SymbolIndexer {
public:
    void indexFile(const std::string& file_path, const std::string& source, const std::string& commit_hash = "head");
    void clear();
    bool save(const std::string& index_path) const;
    bool load(const std::string& index_path);
    std::vector<SymbolRecord> searchSymbols(const std::string& query) const;
    std::vector<SearchResult> keywordSearch(const std::string& query) const;
    std::vector<Chunk> chunks() const;
    std::size_t fileCount() const;

private:
    std::vector<Chunk> chunks_;
    std::vector<SymbolRecord> symbols_;
    static std::string normalizeIdentifier(const std::string& value);
};

} // namespace codedoctor
