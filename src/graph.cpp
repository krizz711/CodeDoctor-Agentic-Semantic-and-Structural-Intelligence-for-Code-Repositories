#include "codedoctor/graph.hpp"

#include <fstream>
#include <sstream>

namespace codedoctor {

void CodeGraph::addEdge(const std::string& source, const std::string& target,
                        const std::string& relation, const std::string& file_path,
                        int line, double confidence) {
    Edge edge;
    edge.source = source;
    edge.target = target;
    edge.relation = relation;
    edge.file_path = file_path;
    edge.line = line;
    edge.confidence = confidence;
    edges_.push_back(edge);
}

bool CodeGraph::save(const std::string& path) const {
    std::ofstream output(path, std::ios::trunc);
    if (!output) return false;
    for (const auto& edge : edges_) {
        output << edge.source << '\t' << edge.target << '\t' << edge.relation << '\t'
               << edge.confidence << '\t' << edge.file_path << '\t' << edge.line << '\n';
    }
    return output.good();
}

bool CodeGraph::load(const std::string& path) {
    std::ifstream input(path);
    if (!input) return false;
    edges_.clear();
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        Edge edge;
        std::string confidence;
        std::string line_number;
        if (!std::getline(row, edge.source, '\t') || !std::getline(row, edge.target, '\t') ||
            !std::getline(row, edge.relation, '\t') || !std::getline(row, confidence, '\t') ||
            !std::getline(row, edge.file_path, '\t') || !std::getline(row, line_number)) continue;
        edge.confidence = std::stod(confidence);
        edge.line = std::stoi(line_number);
        edges_.push_back(edge);
    }
    return true;
}

std::vector<Edge> CodeGraph::edgesFor(const std::string& symbol) const {
    std::vector<Edge> matches;
    for (const auto& edge : edges_) {
        if (edge.source == symbol || edge.target == symbol) {
            matches.push_back(edge);
        }
    }
    return matches;
}

std::vector<std::string> CodeGraph::callersOf(const std::string& symbol) const {
    std::vector<std::string> callers;
    for (const auto& edge : edges_) {
        if (edge.target == symbol && edge.relation == "CALLS") {
            callers.push_back(edge.source);
        }
    }
    return callers;
}

std::size_t CodeGraph::size() const {
    return edges_.size();
}

} // namespace codedoctor
