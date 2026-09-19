#pragma once

#include <string>
#include <vector>

#include "codedoctor/types.hpp"

namespace codedoctor {

class CodeGraph {
public:
    void addEdge(const std::string& source, const std::string& target, const std::string& relation,
                 const std::string& file_path, int line, double confidence = 1.0);
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    std::vector<Edge> edgesFor(const std::string& symbol) const;
    std::vector<std::string> callersOf(const std::string& symbol) const;
    std::size_t size() const;

private:
    std::vector<Edge> edges_;
};

} // namespace codedoctor
