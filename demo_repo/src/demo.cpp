#include <iostream>
#include <string>
#include <vector>

#include "codedoctor/chunker.hpp"
#include "codedoctor/indexer.hpp"
#include "codedoctor/retriever.hpp"
#include "codedoctor/graph.hpp"

int main() {
    const std::string source =
        "export function loginUser(email, password) {\n"
        "    if (!email) {\n"
        "        throw new Error('missing email');\n"
        "    }\n"
        "    return { email, password };\n"
        "}\n\n"
        "class AuthService {\n"
        "    async getUserData() {\n"
        "        return { ok: true };\n"
        "    }\n"
        "}\n";

    codedoctor::SymbolIndexer indexer;
    indexer.indexFile("src/auth.js", source, "abc123");

    auto matches = indexer.searchSymbols("loginUser");
    std::cout << "Matches: " << matches.size() << "\n";

    codedoctor::Retriever retriever(indexer);
    auto results = retriever.search("loginUser");
    std::cout << "Results: " << results.size() << "\n";

    codedoctor::CodeGraph graph;
    graph.addEdge("loginUser", "AuthService", "CALLS", "src/auth.js", 3, 0.92);

    auto callers = graph.callersOf("AuthService");
    std::cout << "Callers: " << callers.size() << "\n";

    return 0;
}
