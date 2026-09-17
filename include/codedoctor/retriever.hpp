#pragma once

#include <string>
#include <vector>

#include "codedoctor/indexer.hpp"
#include "codedoctor/types.hpp"

namespace codedoctor {

class Retriever {
public:
    Retriever(const SymbolIndexer& indexer);
    std::vector<SearchResult> search(const std::string& query) const;

private:
    const SymbolIndexer& indexer_;
};

} // namespace codedoctor
