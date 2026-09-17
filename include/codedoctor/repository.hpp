#pragma once

#include <string>
#include <vector>

#include "codedoctor/graph.hpp"
#include "codedoctor/indexer.hpp"
#include "codedoctor/types.hpp"

namespace codedoctor {

struct IndexReport {
    std::size_t files_indexed = 0;
    std::size_t chunks_indexed = 0;
    std::size_t files_skipped = 0;
};

class RepositoryIndexer {
public:
    IndexReport index(const std::string& root, SymbolIndexer& indexer, CodeGraph& graph) const;
    static bool supportedFile(const std::string& path);
    static bool ignoredDirectory(const std::string& name);

private:
    static std::string relativePath(const std::string& root, const std::string& path);
    static void extractEdges(const std::string& file_path, const std::string& source, CodeGraph& graph);
};

} // namespace codedoctor
