#ifndef DECISIONTREE_CALCULATIONS_HPP
#define DECISIONTREE_CALCULATIONS_HPP

#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include <boost/timer/timer.hpp>
#include "Question.hpp"
#include "Utils.hpp"

using ClassCounter = std::unordered_map<std::string, int>;

// new type of class counter adapted to a dataset transformed in int
using ClassCounterInt = std::unordered_map<int, int>;

namespace Calculations {

	std::tuple<const Data, const Data> partition(const Data& data, const Question& q);

	std::tuple<std::vector<VecI*>, std::vector<VecI*>> partition(const std::vector<VecI*> VecPtrVecI, const Question& q, const MetaData& meta);

	const double gini(const ClassCounter& counts, double N);

	const double gini(const ClassCounterInt& counts, double N);

	std::tuple<const double, const Question> find_best_split(const Data& rows, const MetaData& meta);

	std::tuple<const double, const Question> find_best_split(const std::vector<VecI*> VecPtrVecI, const MetaData& meta);

	std::tuple<std::string, double> determine_best_threshold_numeric(const Data& data, int col);

	std::tuple<std::string, double> determine_best_threshold_cat(const Data& data, int col);

	std::tuple<std::string, double> determine_best_threshold(const std::vector<VecI*> VecPtrVecI, int col, bool isnumeric, const ClassCounterInt decision_counts, const double decision_gini);

	const ClassCounterInt classCounts(const std::vector<VecI*> VecPtrVecI);

	const ClassCounter classCounts(const Data& data);


} // namespace Calculations

#endif //DECISIONTREE_CALCULATIONS_HPP
