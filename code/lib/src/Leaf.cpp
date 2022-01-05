#include "Leaf.hpp"

Leaf::Leaf(const ClassCounter pred) : predictions_(std::move(pred)) {}
