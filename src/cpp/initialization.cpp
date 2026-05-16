#include "initialization.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

RandomPointsInit::RandomPointsInit(uint32_t seed) : gen_(seed) {}

void RandomPointsInit::build(const std::vector<std::vector<float>>& dataset) {
    dataset_ = dataset;
}

std::vector<SearchResult> RandomPointsInit::search(const std::vector<float>& query, size_t k) {
    size_t num_points = dataset_.size();
    if (k > num_points) {
        throw std::invalid_argument("k cannot be larger than the dataset size.");
    }
    
    // 1. Sample k random indices
    std::vector<uint32_t> all_indices(num_points);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    std::vector<uint32_t> sampled_indices(k);
    std::sample(all_indices.begin(), all_indices.end(), sampled_indices.begin(), k, gen_);
    
    // 2. Compute distances
    std::vector<SearchResult> results;
    results.reserve(k);
    for (uint32_t idx : sampled_indices) {
        float dist = compute_l2_distance(dataset_[idx], query);
        results.push_back({idx, dist});
    }
    
    // 3. Sort by distance
    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });
    
    return results;
}

MedoidInit::MedoidInit() : medoid_index_(0) {}

void MedoidInit::build(const std::vector<std::vector<float>>& dataset) {
    dataset_ = dataset;
    if (dataset_.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }
    size_t num_points = dataset_.size();
    size_t dims = dataset_[0].size();
    
    // Calculate centroid
    std::vector<float> centroid(dims, 0.0f);
    for (const auto& point : dataset_) {
        for (size_t i = 0; i < dims; ++i) {
            centroid[i] += point[i];
        }
    }
    for (size_t i = 0; i < dims; ++i) {
        centroid[i] /= num_points;
    }
    
    // Find point closest to centroid
    float min_dist_sq = std::numeric_limits<float>::max();
    medoid_index_ = 0;
    
    for (size_t i = 0; i < num_points; ++i) {
        float dist_sq = 0.0f;
        for (size_t j = 0; j < dims; ++j) {
            float diff = dataset_[i][j] - centroid[j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            medoid_index_ = i;
        }
    }
}

std::vector<SearchResult> MedoidInit::search(const std::vector<float>& query, size_t k) {
    // Medoid approach returns the pre-calculated medoid
    float dist = compute_l2_distance(dataset_[medoid_index_], query);
    return {{medoid_index_, dist}};
}
