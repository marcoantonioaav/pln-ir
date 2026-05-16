#include "initialization.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <fstream>
#include <string>
#include <unordered_set>

// Helper to get current Resident Set Size (RSS) in bytes
size_t get_current_rss_bytes() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            size_t i = 7;
            while (i < line.size() && !std::isdigit(line[i])) i++;
            if (i < line.size()) {
                size_t value = std::stoul(line.substr(i));
                return value * 1024; // Convert kB to bytes
            }
        }
    }
    return 0;
}

// Helper to calculate vector of vectors memory
size_t calculate_dataset_memory(const std::vector<std::vector<float>>& dataset) {
    if (dataset.empty()) return 0;
    size_t total = dataset.size() * sizeof(std::vector<float>);
    total += dataset.size() * dataset[0].size() * sizeof(float);
    return total;
}

RandomPointsInit::RandomPointsInit(uint32_t seed) : gen_(seed) {}

void RandomPointsInit::build(const std::vector<std::vector<float>>& dataset) {
    dataset_ = dataset;
}

size_t RandomPointsInit::get_memory_usage() const {
    return calculate_dataset_memory(dataset_);
}

size_t RandomPointsInit::get_index_size() const {
    return 0; // Random points approach has no persistent index structure
}

std::vector<SearchResult> RandomPointsInit::search(const std::vector<float>& query, size_t k) {
    size_t num_points = dataset_.size();
    if (k > num_points) {
        throw std::invalid_argument("k cannot be larger than the dataset size.");
    }
    
    // 1. Sample k random indices efficiently (O(k) expected time)
    std::unordered_set<uint32_t> sampled_indices;
    std::uniform_int_distribution<uint32_t> dist_gen(0, num_points - 1);
    while (sampled_indices.size() < k) {
        sampled_indices.insert(dist_gen(gen_));
    }
    
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

size_t MedoidInit::get_memory_usage() const {
    return calculate_dataset_memory(dataset_) + sizeof(medoid_index_);
}

size_t MedoidInit::get_index_size() const {
    return sizeof(medoid_index_);
}

std::vector<SearchResult> MedoidInit::search(const std::vector<float>& query, size_t k) {
    float dist = compute_l2_distance(dataset_[medoid_index_], query);
    return {{medoid_index_, dist}};
}

// ---------------------------------------------------------
// FLANN Implementation
// ---------------------------------------------------------
#include <flann/flann.hpp>

struct FlannState {
    std::unique_ptr<flann::Matrix<float>> flann_dataset;
    std::unique_ptr<flann::Index<flann::L2<float>>> index;
};

FlannKDTreeInit::FlannKDTreeInit(int trees, int checks) 
    : num_trees_(trees), checks_(checks), state_(new FlannState()) {}

FlannKDTreeInit::~FlannKDTreeInit() {
    delete state_;
}

void FlannKDTreeInit::build(const std::vector<std::vector<float>>& dataset) {
    size_t rss_start = get_current_rss_bytes();
    dataset_ = dataset;
    
    if (dataset_.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }

    size_t num_points = dataset_.size();
    size_t dims = dataset_[0].size();

    float* data_ptr = new float[num_points * dims];
    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < dims; ++j) {
            data_ptr[i * dims + j] = dataset_[i][j];
        }
    }

    state_->flann_dataset = std::make_unique<flann::Matrix<float>>(data_ptr, num_points, dims);
    
    flann::KDTreeIndexParams params(num_trees_);
    state_->index = std::make_unique<flann::Index<flann::L2<float>>>(*state_->flann_dataset, params);
    
    size_t rss_before_index = get_current_rss_bytes();
    state_->index->buildIndex();
    size_t rss_after_index = get_current_rss_bytes();

    size_t rss_end = get_current_rss_bytes();
    
    // Total footprint during indexing
    memory_usage_ = (rss_end > rss_start) ? (rss_end - rss_start) : calculate_dataset_memory(dataset_);
    // Actual index structure size
    index_size_ = (rss_after_index > rss_before_index) ? (rss_after_index - rss_before_index) : 0;
}

std::vector<SearchResult> FlannKDTreeInit::search(const std::vector<float>& query, size_t k) {
    size_t dims = query.size();
    std::vector<float> query_copy = query;
    flann::Matrix<float> flann_query(query_copy.data(), 1, dims);

    std::vector<int> indices(k);
    std::vector<float> dists(k);
    flann::Matrix<int> flann_indices(indices.data(), 1, k);
    flann::Matrix<float> flann_dists(dists.data(), 1, k);

    flann::SearchParams params(checks_);
    state_->index->knnSearch(flann_query, flann_indices, flann_dists, k, params);
    
    distance_computations_ += checks_;

    std::vector<SearchResult> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        if (indices[i] >= 0) {
            results.push_back({(uint32_t)indices[i], std::sqrt(dists[i])});
        }
    }

    return results;
}

size_t FlannKDTreeInit::get_memory_usage() const {
    return memory_usage_;
}

size_t FlannKDTreeInit::get_index_size() const {
    return index_size_;
}
