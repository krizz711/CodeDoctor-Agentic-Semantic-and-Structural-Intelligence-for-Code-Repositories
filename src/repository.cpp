#include "codedoctor/repository.hpp"

#include <algorithm>
#include <cctype>
#include <dirent.h>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <sys/stat.h>

namespace codedoctor {

bool RepositoryIndexer::supportedFile(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string extension = path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return extension == ".js" || extension == ".jsx" || extension == ".mjs" ||
           extension == ".cjs" || extension == ".ts" || extension == ".tsx";
}

bool RepositoryIndexer::ignoredDirectory(const std::string& name) {
    static const std::set<std::string> ignored = {
        ".git", "node_modules", "dist", "build", "coverage", ".cache", ".idea", ".vscode"
    };
    return ignored.count(name) != 0;
}

std::string RepositoryIndexer::relativePath(const std::string& root, const std::string& path) {
    std::string relative = path;
    if (relative.compare(0, root.size(), root) == 0) relative = relative.substr(root.size());
    while (!relative.empty() && (relative.front() == '/' || relative.front() == '\\')) relative.erase(relative.begin());
    std::replace(relative.begin(), relative.end(), '\\', '/');
    return relative;
}

void RepositoryIndexer::extractEdges(const std::string& file_path, const std::string& source, CodeGraph& graph) {
    const std::regex function_re(R"((?:function\s+|class\s+|(?:async\s+)?)([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{)");
    struct FunctionDef {
        std::string name;
        std::size_t position;
        int line;
    };
    std::vector<FunctionDef> defs;
    std::set<std::string> unique_defs;

    for (std::sregex_iterator it(source.begin(), source.end(), function_re), end; it != end; ++it) {
        std::size_t pos = it->position(0);
        int line = 1 + static_cast<int>(std::count(source.begin(), source.begin() + pos, '\n'));
        defs.push_back({(*it)[1].str(), pos, line});
        unique_defs.insert((*it)[1].str());
    }

    for (const auto& definition : unique_defs) {
        const std::regex call_re("\\b" + definition + "\\s*\\(");
        for (std::sregex_iterator it(source.begin(), source.end(), call_re), end; it != end; ++it) {
            const auto position = static_cast<std::size_t>((*it).position(0));
            const int line = 1 + static_cast<int>(std::count(source.begin(), source.begin() + position, '\n'));

            bool is_def_site = false;
            for (const auto& d : defs) {
                if (d.name == definition && d.line == line) {
                    is_def_site = true;
                    break;
                }
            }
            if (is_def_site) continue;

            std::string caller = file_path;
            for (auto rit = defs.rbegin(); rit != defs.rend(); ++rit) {
                if (rit->position < position) {
                    caller = rit->name;
                    break;
                }
            }

            graph.addEdge(caller, definition, "CALLS", file_path, line, 0.55);
        }
    }
}

IndexReport RepositoryIndexer::index(const std::string& root, SymbolIndexer& indexer, CodeGraph& graph) const {
    IndexReport report;
    struct Walker {
        const RepositoryIndexer& owner;
        const std::string& root;
        SymbolIndexer& indexer;
        CodeGraph& graph;
        IndexReport& report;

        void visit(const std::string& directory) {
            DIR* handle = opendir(directory.c_str());
            if (!handle) return;
            while (const dirent* entry = readdir(handle)) {
                const std::string name = entry->d_name;
                if (name == "." || name == "..") continue;
                const std::string path = directory + "/" + name;
                struct stat info {};
                if (stat(path.c_str(), &info) != 0) continue;
                if (S_ISDIR(info.st_mode)) {
                    if (!RepositoryIndexer::ignoredDirectory(name)) visit(path);
                    continue;
                }
                if (!S_ISREG(info.st_mode) || !RepositoryIndexer::supportedFile(path)) {
                    ++report.files_skipped;
                    continue;
                }
                std::ifstream input(path, std::ios::binary);
                if (!input) {
                    ++report.files_skipped;
                    continue;
                }
                std::ostringstream buffer;
                buffer << input.rdbuf();
                const std::string source = buffer.str();
                const std::string relative = RepositoryIndexer::relativePath(root, path);
                indexer.indexFile(relative, source);
                RepositoryIndexer::extractEdges(relative, source, graph);
                ++report.files_indexed;
            }
            closedir(handle);
        }
    } walker{*this, root, indexer, graph, report};
    walker.visit(root);
    report.chunks_indexed = indexer.chunks().size();
    return report;
}

} // namespace codedoctor
