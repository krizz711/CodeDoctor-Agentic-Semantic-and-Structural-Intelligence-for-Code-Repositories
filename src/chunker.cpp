#include "codedoctor/chunker.hpp"

#include <algorithm>
#include <cstddef>
#include <regex>
#include <string>
#include <vector>

namespace codedoctor {

namespace {

std::string extractFunctionSignature(const std::string& snippet) {
    const std::regex re("(?:async\\s+)?function\\s+[A-Za-z0-9_]+\\s*\\([^)]*\\)");
    std::smatch match;
    if (std::regex_search(snippet, match, re)) {
        return match.str();
    }
    return snippet.substr(0, 200);
}

std::size_t findMatchingBrace(const std::string& source, std::size_t open_brace_index) {
    int depth = 0;
    for (std::size_t i = open_brace_index; i < source.size(); ++i) {
        if (source[i] == '{') {
            ++depth;
        } else if (source[i] == '}') {
            --depth;
            if (depth == 0) {
                return i + 1;
            }
        }
    }
    return source.size();
}

} // namespace

std::vector<Chunk> Chunker::chunkSource(const std::string& source, const std::string& file_path) {
    std::vector<Chunk> chunks;

    const std::regex function_re(
        R"((?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{)");

    std::sregex_iterator it(source.begin(), source.end(), function_re);
    std::sregex_iterator end;
    for (; it != end; ++it) {
        const std::smatch match = *it;
        const std::string name = match[1].str();
        const std::size_t start = static_cast<std::size_t>(match.position(0));
        const std::size_t brace_pos = source.find('{', start);
        const std::size_t end_byte = findMatchingBrace(source, brace_pos);
        if (end_byte <= start) continue;

        Chunk chunk;
        chunk.file_path = file_path;
        chunk.type = "FUNCTION";
        chunk.name = name;
        chunk.parent = "";
        chunk.content = source.substr(start, end_byte - start);
        chunk.signature = extractFunctionSignature(match.str());
        chunk.start_byte = start;
        chunk.end_byte = end_byte;
        chunk.start_line = 1 + static_cast<int>(std::count(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(start), '\n'));
        chunk.end_line = 1 + static_cast<int>(std::count(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(end_byte), '\n'));
        chunks.push_back(chunk);
    }

    const std::regex class_re(R"((?:export\s+)?class\s+([A-Za-z0-9_]+)\s*\{)");
    for (std::sregex_iterator it(source.begin(), source.end(), class_re), end; it != end; ++it) {
        const std::smatch match = *it;
        const std::string name = match[1].str();
        const std::size_t start = static_cast<std::size_t>(match.position(0));
        const std::size_t brace_pos = source.find('{', start);
        const std::size_t end_byte = findMatchingBrace(source, brace_pos);
        if (end_byte <= start) continue;

        bool duplicate = false;
        for (const auto& c : chunks) {
            if (c.start_byte == start && c.name == name) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        Chunk chunk;
        chunk.file_path = file_path;
        chunk.type = "CLASS";
        chunk.name = name;
        chunk.content = source.substr(start, end_byte - start);
        chunk.signature = "class " + name;
        chunk.start_byte = start;
        chunk.end_byte = end_byte;
        chunk.start_line = 1 + static_cast<int>(std::count(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(start), '\n'));
        chunk.end_line = 1 + static_cast<int>(std::count(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(end_byte), '\n'));
        chunks.push_back(chunk);
    }

    const std::regex method_re(R"((?:async\s+)?([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{)");
    for (std::sregex_iterator it(source.begin(), source.end(), method_re), end; it != end; ++it) {
        const std::smatch match = *it;
        const std::string name = match[1].str();
        const std::size_t start = static_cast<std::size_t>(match.position(0));
        bool inside_class = false;
        for (const auto& c : chunks) {
            if (c.type == "CLASS" && start > c.start_byte && start < c.end_byte) {
                inside_class = true;
                break;
            }
        }
        if (!inside_class) continue;
        const std::size_t brace_pos = source.find('{', start);
        const std::size_t end_byte = findMatchingBrace(source, brace_pos);
        if (end_byte <= start) continue;

        bool duplicate = false;
        for (const auto& c : chunks) {
            if (c.start_byte == start) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        Chunk chunk;
        chunk.file_path = file_path;
        chunk.type = "METHOD";
        chunk.name = name;
        chunk.content = source.substr(start, end_byte - start);
        chunk.signature = match.str();
        chunk.start_byte = start;
        chunk.end_byte = end_byte;
        chunk.start_line = 1 + static_cast<int>(std::count(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(start), '\n'));
        chunk.end_line = 1 + static_cast<int>(std::count(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(end_byte), '\n'));
        chunks.push_back(chunk);
    }

    if (chunks.empty()) {
        Chunk chunk;
        chunk.file_path = file_path;
        chunk.type = "FILE_SUMMARY";
        chunk.name = "file";
        chunk.content = source;
        chunk.signature = "file";
        chunk.start_byte = 0;
        chunk.end_byte = source.size();
        chunk.start_line = 1;
        chunk.end_line = 1 + static_cast<int>(std::count(source.begin(), source.end(), '\n'));
        chunks.push_back(chunk);
    }

    return chunks;
}

} // namespace codedoctor
