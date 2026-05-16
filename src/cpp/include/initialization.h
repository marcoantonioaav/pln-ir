#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <utility>
#include <cmath>

struct SearchResult {
    uint32_t index;
    float distance;
};

class InitializationApproach {
protected:
    mutable size_t distance_computations_ = 0;
    std::vector<std::vector<float>> dataset_;

    float compute_l2_distance(const std::vector<float>& v1, const std::vector<float>& v2) const {
        distance_computations_++;
        float dist_sq = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            float diff = v1[i] - v2[i];
            dist_sq += diff * diff;
        }
        return std::sqrt(dist_sq);
    }

public:
    virtual ~InitializationApproach() = default;
    
    // Build offline index / initialize required components
    virtual void build(const std::vector<std::vector<float>>& dataset) = 0;
    
    // Search online, returning a sorted list of (index, distance)
    virtual std::vector<SearchResult> search(const std::vector<float>& query, size_t k) = 0;

    size_t get_distance_computations() const { return distance_computations_; }
    void reset_distance_computations() { distance_computations_ = 0; }
};

class RandomPointsInit : public InitializationApproach {
private:
    std::mt19937 gen_;
public:
    RandomPointsInit(uint32_t seed = 42);
    void build(const std::vector<std::vector<float>>& dataset) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
};

class MedoidInit : public InitializationApproach {
private:
    uint32_t medoid_index_;
public:
    MedoidInit();
    void build(const std::vector<std::vector<float>>& dataset) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
};
