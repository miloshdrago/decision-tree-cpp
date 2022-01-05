#include <cmath>
#include <algorithm>
#include <iterator>
#include "Calculations.hpp"
#include "Utils.hpp"

using std::tuple;
using std::pair;
using std::forward_as_tuple;
using std::vector;
using std::string;
using std::unordered_map;


// Partition the dataset in two subsets
tuple<std::vector<VecI*>, std::vector<VecI*>> Calculations::partition(const std::vector<VecI*> VecPtrVecI, const Question& q, const MetaData& meta) {
	std::vector<VecI*> true_rows; // vector for dataset S1
	std::vector<VecI*> false_rows; // vector for dataset S2
	int split_value; // the value spliting the dataset S

	// check if the column is of ordinal type
	if (meta.isnumeric[q.column_]) {
		// initialize the best split value from the question q
		split_value = stoi(q.value_);
		// loop over all rows of data
		for (size_t row = 0; row < VecPtrVecI.size(); row++) {
			// if ordinal value is greater or equal than best split value, the row goes to true partition
			if (VecPtrVecI[row]->at(q.column_) >= split_value) {
				true_rows.push_back(VecPtrVecI[row]);
			}
			// if ordinal value is less than best split value, the row goes to false partition
			else {
				false_rows.push_back(VecPtrVecI[row]);
			}
		}
	}
	// column is of categorical type
	else {
		// initialize the best split value from the question q and map the string to its integer value
		split_value = meta.mapS2I[q.column_].at(q.value_);
		// loop over all rows of data
		for (size_t row = 0; row < VecPtrVecI.size(); row++) {
			// if categorical value is equal to best split value, the row goes to true partition
			if (VecPtrVecI[row]->at(q.column_) == split_value) {
				true_rows.push_back(VecPtrVecI[row]);
			}
			// if categorical value is different to best split value, the row goes to false partition
			else {
				false_rows.push_back(VecPtrVecI[row]);
			}
		}
	}
	return forward_as_tuple(true_rows, false_rows);
}

// Find the best split question and gain
tuple<const double, const Question> Calculations::find_best_split(const std::vector<VecI*> VecPtrVecI, const MetaData& meta) {
	double best_gain = 0.0;  // keep track of the best information gain
	auto best_question = Question();  //keep track of the feature / value that produced it
	tuple<std::string, double> curcolgain; // stores the best gain in the current column
	ClassCounterInt decision_counts; // the class count of the decision column
	double decision_gini_score; // the gini score of the dataset


	// get the total counts from the decision column
	decision_counts = classCounts(VecPtrVecI);
	// compute the gini score for the dataset Gini(S)
	decision_gini_score = gini(decision_counts, VecPtrVecI.size());
	// loop through each column to find the best threshold of the column
	for (size_t col = 0; col < (meta.labels.size() - 1); col++) {
		curcolgain = determine_best_threshold(VecPtrVecI, col, meta.isnumeric[col], decision_counts, decision_gini_score);
		// compare current column gain to best gain, if it is better store the column id, the question value and the information gain
		if (std::get<1>(curcolgain) > best_gain) {
			best_question.column_ = col;
			best_gain = std::get<1>(curcolgain);
			// if column is ordinal then we can directly store the string representing the ordinal
			if (meta.isnumeric[col]) {
				best_question.value_ = std::get<0>(curcolgain);
			}
			// if column is categorical we need to look back the original data string represented by the current question string value
			// for example the string value "7" could in fact represent the original string "R2D2" read in the dataset
			else {
				best_question.value_ = meta.mapI2S[col].at(stoi(std::get<0>(curcolgain)));
			}
		}
	}
	return forward_as_tuple(best_gain, best_question);
}

// Calculates the Gini score based on the relative frequency of a class
const double Calculations::gini(const ClassCounterInt& counts, double N) {
	double impurity = 1.0;
	// loop for each class and substract the relative impurity according to Gini formula
	for (const auto& n : counts) {
		impurity = impurity - pow(((double)n.second / N), 2);
	}
	return impurity;
}

// Find the best threshold value in one column with highest gain
tuple<std::string, double> Calculations::determine_best_threshold(const std::vector<VecI*> VecPtrVecI, int col, bool isnumeric, ClassCounterInt decision_counts, const double decision_gini) {
	double best_gain = 0; // the best gain
	double gain = 0; // the current value gain
	double value_gini_score, not_value_gini_score; // gini scores for S1 and S2 datasets
	std::string best_thresh; // the question value representing the best threshold
	ClassCounterInt value_counts, not_value_counts; // class counters for S1 and S2 datasets
	int current_value = 0; // current value being analysed
	int next_value = 0; // next value , this is used to detect a change of value in the column and to trigger the gini and gain calculations
	size_t total_value_count = 0; // total class count for S1
	size_t total_decision_count = VecPtrVecI.size(); // total class count for S (decision column)
	size_t decision_col = VecPtrVecI[0]->size() - 1; // index of the decision column in the data table
	size_t max_rows = total_decision_count; // number of rows in dataset S
	vector<tuple<int, int>> mapValDec; // mapping table between column value and decision value, used for quicker sorting
	bool calc_gain = false; // boolean trigger used to trigger the calculation of the gini and information gain

	// Create a mapping table between the numerical value and its row number in the data table
	// This is used to sort the column values instead of the big data table as it is quite faster
	for (size_t row = 0; row < max_rows; row++) {
		mapValDec.push_back(std::make_tuple(VecPtrVecI[row]->at(col), VecPtrVecI[row]->at(decision_col)));
	}

	// Sort ascending the mapping table based on the value of the column we look for the best threshold
	std::sort(mapValDec.begin(), mapValDec.end(), [](tuple<int, int> a, tuple<int, int> b) { return std::get<0>(a) < std::get<0>(b); });

	//initialize the current_value with the first value in the column, as it has been sorted ascending should be the smallest one
	current_value = std::get<0>(mapValDec[0]);

	for (size_t row = 0; row < max_rows; row++) {
		// increase the counters based on the decision column value
		value_counts[std::get<1>(mapValDec[row])] += 1;
		total_value_count++;
		// while we are not at the last row compare current row value to the one in the next row to detect a change
		if (row < (max_rows - 1)) {
			// check if next value is the same, if not we need to calculate the gain for the current value
			if (current_value != std::get<0>(mapValDec[row + 1])) {
				calc_gain = true;
				next_value = std::get<0>(mapValDec[row + 1]);
			}
		}
		else {
			// we are at the last row, we force the calculation of the gain for the last possible value
			calc_gain = true;
			next_value = current_value;
		}
		if (calc_gain) {
			// The value in the sorted column changed, we need to calculate the gain for the previous value
			value_gini_score = gini(value_counts, total_value_count);
			// Calculate the gini score for the other values in this column 
			not_value_counts = ClassCounterInt();
			for (const auto& n : decision_counts) {
				not_value_counts[n.first] = decision_counts[n.first] - value_counts[n.first];
			}

			not_value_gini_score = gini(not_value_counts, total_decision_count - total_value_count);
			gain = decision_gini - ((total_value_count * value_gini_score) / total_decision_count) - (((total_decision_count - total_value_count)) * not_value_gini_score / total_decision_count);
			if (gain > best_gain) {
				best_gain = gain;
				best_thresh = std::to_string(next_value);
			}
			// If the column is categorical type reinitialize totals to count for next value in the sorted column
			// For ordinal type we cumulate the totals throughout the table
			if (isnumeric == false) {
				value_counts = ClassCounterInt();
				total_value_count = 0;
			}
			current_value = next_value;
			calc_gain = false;
		}
	}
	return forward_as_tuple(best_thresh, best_gain);
}

// Counts the total number of instances of each class in the decision column
const ClassCounterInt Calculations::classCounts(const std::vector<VecI*> VecPtrVecI) {
	ClassCounterInt decision_counts; // a class counter for dataset S
	size_t max_rows = VecPtrVecI.size(); // number of rows in dataset S
	size_t decision_col = VecPtrVecI[0]->size() - 1; // index of the decision column in the data table

	for (size_t row = 0; row < max_rows; row++) {
		decision_counts[VecPtrVecI[row]->at(decision_col)]++;
	}
	return decision_counts;
}
