#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
#include <algorithm>

namespace py = pybind11;

std::vector<py::dict> hybrid_search(const std::string& query, 
                                    const std::vector<py::dict>& dense_results, 
                                    const std::vector<py::dict>& lexical_results) {
    std::map<int, double> fusion_scores;
    
    int rank = 0;
    for (const auto& item : dense_results) {
        int chunk_id = -1;
        if (item.contains("chunk_id")) chunk_id = item["chunk_id"].cast<int>();
        else if (item.contains("id")) chunk_id = item["id"].cast<int>();
        
        if (chunk_id != -1) {
            fusion_scores[chunk_id] += 1.0 / (rank + 60.0);
        }
        rank++;
    }
    
    rank = 0;
    for (const auto& item : lexical_results) {
        int chunk_id = -1;
        if (item.contains("chunk_id")) chunk_id = item["chunk_id"].cast<int>();
        else if (item.contains("id")) chunk_id = item["id"].cast<int>();
        
        if (chunk_id != -1) {
            fusion_scores[chunk_id] += 1.0 / (rank + 60.0);
        }
        rank++;
    }
    
    // Convert to vector of pairs to sort
    std::vector<std::pair<int, double>> sorted_results(fusion_scores.begin(), fusion_scores.end());
    std::sort(sorted_results.begin(), sorted_results.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
              
    std::vector<py::dict> final_results;
    for (const auto& pair : sorted_results) {
        py::dict res;
        res["chunk_id"] = pair.first;
        res["score"] = pair.second;
        final_results.push_back(res);
    }
    
    return final_results;
}

PYBIND11_MODULE(codedoctor_cpp, m) {
    m.doc() = "CodeDoctor C++ Extension for Hybrid Search";
    m.def("hybrid_search", &hybrid_search, "Perform reciprocal rank fusion on dense and lexical result sets");
}
