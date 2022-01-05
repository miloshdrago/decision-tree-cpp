#ifndef DECISIONTREE_DECISIONTREE_HPP
#define DECISIONTREE_DECISIONTREE_HPP

#include "Calculations.hpp"
#include "DataReader.hpp"
#include "Node.hpp"
#include "TreeTest.hpp"
#include "Utils.hpp"

class DecisionTree {
public:
	DecisionTree() = delete;
	explicit DecisionTree(const DataReader& dr);
	explicit DecisionTree(DataReader& dr, DataInt& bootstrap_int);
	void print() const;
	void test() const;

	inline Data testData() { return dr_.testData(); }
	inline std::shared_ptr<Node> root() { return std::make_shared<Node>(root_); }

	Node root_;

private:
	// replaced the following line as it was keeping a copy of the dataset instead of the address of the dataset
	// 	   It was  consuming a lot of memory especially for the bagging
	//DataReader dr_;
	const DataReader& dr_;
	static Node buildTree(const MetaData& meta, const std::vector<std::vector<int>*> VecPtrVecI);

	//const Node buildTree(const Data& rows, const MetaData &meta);
	void print(const std::shared_ptr<Node> root, std::string spacing = "") const;
};

#endif //DECISIONTREE_DECISIONTREE_HPP
