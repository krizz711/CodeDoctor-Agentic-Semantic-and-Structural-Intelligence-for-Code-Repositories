#pragma once

#include <string>
#include <vector>

namespace codedoctor {

struct Chunk {
    std::string file_path;
    std::string type;
    std::string name;
    std::string parent;
    std::string content;
    std::string signature;
    std::size_t start_byte = 0;
    std::size_t end_byte = 0;
    int start_line = 1;
    int end_line = 1;
};

struct SymbolRecord {
    std::string chunk_id;
    std::string name;
    std::string normalized_name;
    std::string type;
    std::string file_path;
    int line = 1;
};

struct SearchResult {
    std::string file_path;
    std::string name;
    std::string type;
    double score = 0.0;
    std::string snippet;
};

struct Edge {
    std::string source;
    std::string target;
    std::string relation;
    double confidence = 0.0;
    std::string file_path;
    int line = 1;
};

struct RepoMetadata {
    std::string repository_name;
    std::string url;
    std::string current_commit;
    std::string branch;
    std::vector<std::string> file_tree;
    std::vector<std::string> supported_extensions;
};

} // namespace codedoctor
