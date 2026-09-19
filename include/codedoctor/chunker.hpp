#pragma once

#include <string>
#include <vector>

#include "codedoctor/types.hpp"

namespace codedoctor {

class Chunker {
public:
    static std::vector<Chunk> chunkSource(const std::string& source, const std::string& file_path);
};

} // namespace codedoctor
