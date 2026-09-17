#include <cassert>
#include <string>

#include "codedoctor/chunker.hpp"
#include "codedoctor/graph.hpp"
#include "codedoctor/indexer.hpp"
#include "codedoctor/retriever.hpp"

int main() {
    const std::string source =
        "export function loginUser(email, password) {\n"
        "    return { email, password };\n"
        "}\n\n"
        "class AuthService {\n"
        "    async getUserData() { return { ok: true }; }\n"
        "}\n";

    const auto chunks = codedoctor::Chunker::chunkSource(source, "src/auth.js");
    assert(chunks.size() == 3);
    for (const auto& chunk : chunks) {
        assert(chunk.content == source.substr(chunk.start_byte, chunk.end_byte - chunk.start_byte));
    }

    codedoctor::SymbolIndexer indexer;
    indexer.indexFile("src/auth.js", source, "abc123");
    assert(indexer.searchSymbols("loginUser").size() == 1);
    assert(indexer.searchSymbols("get_user_data").size() == 1);
    assert(!indexer.keywordSearch("password").empty());

    codedoctor::Retriever retriever(indexer);
    assert(!retriever.search("loginUser").empty());

    codedoctor::CodeGraph graph;
    graph.addEdge("loginUser", "AuthService", "CALLS", "src/auth.js", 2, 0.92);
    const auto callers = graph.callersOf("AuthService");
    assert(callers.size() == 1);
    assert(callers[0] == "loginUser");

    return 0;
}
