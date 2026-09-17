#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "codedoctor/graph.hpp"
#include "codedoctor/repository.hpp"
#include "codedoctor/retriever.hpp"

namespace {

void printUsage() {
    std::cout << "CodeDoctor CPU repository intelligence\n\n"
              << "Usage:\n"
              << "  codedoctor index <repository> [index-file]\n"
              << "  codedoctor search <index-file> <query>\n"
              << "  codedoctor symbol <index-file> <symbol>\n"
              << "  codedoctor ask <index-file> <question>\n"
              << "  codedoctor callers <index-file> <symbol>\n"
              << "  codedoctor stats <index-file>\n";
}

std::string joinArguments(int argc, char** argv, int first) {
    std::ostringstream result;
    for (int i = first; i < argc; ++i) {
        if (i > first) result << ' ';
        result << argv[i];
    }
    return result.str();
}

void printResult(const codedoctor::SearchResult& result) {
    std::cout << result.file_path << ": " << result.name
              << " [" << result.type << "] score=" << result.score << "\n";
    std::cout << result.snippet << "\n\n";
}

int indexRepository(const std::string& root, const std::string& indexPath) {
    codedoctor::SymbolIndexer indexer;
    codedoctor::CodeGraph graph;
    const auto report = codedoctor::RepositoryIndexer().index(root, indexer, graph);
    if (!indexer.save(indexPath)) {
        std::cerr << "Failed to write index: " << indexPath << "\n";
        return 1;
    }
    graph.save(indexPath + ".graph");
    std::cout << "Indexed repository: " << root << "\n"
              << "Files: " << report.files_indexed << "\n"
              << "Chunks: " << report.chunks_indexed << "\n"
              << "Skipped: " << report.files_skipped << "\n"
              << "Index: " << indexPath << "\n";
    return 0;
}

bool loadIndex(const std::string& path, codedoctor::SymbolIndexer& indexer) {
    if (!indexer.load(path)) {
        std::cerr << "Unable to load index: " << path << "\n";
        return false;
    }
    return true;
}

int searchIndex(const std::string& indexPath, const std::string& query, bool exact) {
    codedoctor::SymbolIndexer indexer;
    if (!loadIndex(indexPath, indexer)) return 1;
    if (exact) {
        const auto symbols = indexer.searchSymbols(query);
        for (const auto& symbol : symbols) {
            std::cout << symbol.file_path << ":" << symbol.line << " "
                      << symbol.name << " [" << symbol.type << "]\n";
        }
        std::cout << "Matches: " << symbols.size() << "\n";
        return symbols.empty() ? 2 : 0;
    }
    codedoctor::Retriever retriever(indexer);
    const auto results = retriever.search(query);
    for (const auto& result : results) printResult(result);
    std::cout << "Results: " << results.size() << "\n";
    return results.empty() ? 2 : 0;
}

int askIndex(const std::string& indexPath, const std::string& question) {
    codedoctor::SymbolIndexer indexer;
    if (!loadIndex(indexPath, indexer)) return 1;
    std::cout << "Question: " << question << "\n\nEvidence:\n";
    const auto results = codedoctor::Retriever(indexer).search(question);
    if (results.empty()) {
        std::cout << "No indexed evidence found. The answer is intentionally unresolved.\n";
        return 2;
    }
    const std::size_t limit = std::min<std::size_t>(results.size(), 5);
    for (std::size_t i = 0; i < limit; ++i) printResult(results[i]);
    std::cout << "Grounding: answer only from the evidence above.\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    const std::string command = argv[1];
    if (command == "index" && (argc == 3 || argc == 4)) {
        return indexRepository(argv[2], argc == 4 ? argv[3] : "build/index.codedoctor");
    }
    if ((command == "search" || command == "symbol" || command == "ask") && argc >= 4) {
        const std::string query = joinArguments(argc, argv, 3);
        if (command == "symbol") return searchIndex(argv[2], query, true);
        if (command == "ask") return askIndex(argv[2], query);
        return searchIndex(argv[2], query, false);
    }
    if (command == "stats" && argc == 3) {
        codedoctor::SymbolIndexer indexer;
        if (!loadIndex(argv[2], indexer)) return 1;
        codedoctor::CodeGraph graph;
        graph.load(std::string(argv[2]) + ".graph");
        std::cout << "Files: " << indexer.fileCount() << "\n"
                  << "Chunks: " << indexer.chunks().size() << "\n"
                  << "Edges: " << graph.size() << "\n";
        return 0;
    }
    if (command == "callers" && argc >= 4) {
        codedoctor::CodeGraph graph;
        if (!graph.load(std::string(argv[2]) + ".graph")) {
            std::cerr << "Unable to load graph: " << argv[2] << ".graph\n";
            return 1;
        }
        const auto callers = graph.callersOf(joinArguments(argc, argv, 3));
        for (const auto& caller : callers) std::cout << caller << '\n';
        std::cout << "Callers: " << callers.size() << "\n";
        return callers.empty() ? 2 : 0;
    }
    printUsage();
    return 1;
}
