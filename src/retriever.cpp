#include "codedoctor/retriever.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace codedoctor {

Retriever::Retriever(const SymbolIndexer& indexer) : indexer_(indexer) {}

std::vector<SearchResult> Retriever::search(const std::string& query) const {
    std::vector<SearchResult> results;

    const auto addUnique = [&results](SearchResult result) {
        for (const auto& existing : results) {
            if (existing.file_path == result.file_path && existing.name == result.name && existing.type == result.type) {
                return;
            }
        }
        results.push_back(std::move(result));
    };

    const auto exact = indexer_.searchSymbols(query);
    for (const auto& symbol : exact) {
        SearchResult result;
        result.file_path = symbol.file_path;
        result.name = symbol.name;
        result.type = symbol.type;
        result.score = 10.0;
        result.snippet = symbol.name;
        addUnique(std::move(result));
    }

    const auto keyword = indexer_.keywordSearch(query);
    for (auto item : keyword) addUnique(std::move(item));

    std::vector<std::string> terms;
    const std::regex term_pattern("[A-Za-z][A-Za-z0-9_]+");
    for (std::sregex_iterator it(query.begin(), query.end(), term_pattern), end; it != end; ++it) {
        terms.push_back((*it).str());
    }
    for (const auto& term : terms) {
        for (auto item : indexer_.keywordSearch(term)) {
            item.score = 1.0;
            addUnique(std::move(item));
        }
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& left, const SearchResult& right) {
        return left.score > right.score;
    });

    return results;
}

} // namespace codedoctor
