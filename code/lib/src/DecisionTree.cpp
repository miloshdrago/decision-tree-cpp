#include "DecisionTree.hpp"
#include <future>
#include <chrono>


using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;
using Calculations::find_best_split;
using Calculations::partition;
using std::tuple;
using std::future;


DecisionTree::DecisionTree(const DataReader& dr) : root_(Node()), dr_(dr) {
	std::vector<VecI*> VecPtrVecI; // vector of pointers to vectors of int
	DataInt* ptrtheTable; // pointer to the DataInt dataset

	// initialize pointer to the beginning of the traning dataset
	ptrtheTable = (DataInt*)&dr.trainDataInt();
	// Initialize the vector of pointer to point to each vector of int which is equivalent to one data row
	for (size_t row = 0; row < ptrtheTable->size(); row++) {
		VecPtrVecI.push_back(&ptrtheTable->at(row));
	}
	cpu_timer timer;
	// build the tree
	root_ = buildTree(dr.metaData(), VecPtrVecI);
	std::cout << "Done. " << timer.format() << std::endl;
}


DecisionTree::DecisionTree(DataReader& dr, DataInt& bootstrap_int) : root_(Node()), dr_(dr) {
	std::vector<VecI*> VecPtrVecI; // vector of pointers to vectors of int
	DataInt* ptrtheTable; // pointer to the DataInt dataset

	// initialize pointer to the beginning of the bootstrap dataset
	ptrtheTable = (DataInt*)&bootstrap_int;
	// Initialize the vector of pointer to point to each vector of int which is equivalent to one data row
	for (size_t row = 0; row < ptrtheTable->size(); row++) {
		VecPtrVecI.push_back(&ptrtheTable->at(row));
	}
	cpu_timer timer;
	// build the tree
	root_ = buildTree(dr.metaData(), VecPtrVecI);
	std::cout << "Done. " << timer.format() << std::endl;
}


Node DecisionTree::buildTree(const MetaData& meta, const std::vector<std::vector<int>*> VecPtrVecI) {
	tuple< double, Question> thesplit; // the split point
	double thegain; // the gain returned at thes plit point 
	Node left_node, right_node; // pointers to left and right nodes
	Question thequestion; // the question returned at the split point
	std::vector<VecI*> right_VecPtrVecI; // vector of pointers to the S1 dataset
	std::vector<VecI*> left_VecPtrVecI; // vector of pointers to the S2 dataset
	tuple<std::vector<VecI*>, std::vector<VecI*>> thepartition; // the partition containing the S1 and S2 datasets
	future<Node> left_future, right_future; // futures to store the left and right Nodes returned by the asynchronous threads
	size_t decision_col = meta.labels.size() - 1; // index of the decision column
	
	// Find the best split in the dataset S and retrieve the information gain and the split question
	thesplit = find_best_split(VecPtrVecI, meta);
	thegain = std::get<0>(thesplit);
	thequestion = std::get<1>(thesplit);

	// check if the information gain is null then we are on a Leaf Node
	if (thegain == 0) {
		ClassCounter value_counts;
		size_t total_value_count = 0;
		// count the class counts in the decision column
		for (size_t row = 0; row < VecPtrVecI.size(); row++) {
			string str_leaf = meta.mapI2S[decision_col].at(VecPtrVecI[row]->at(decision_col));
			value_counts[str_leaf] += 1;
			total_value_count++;
		}
		return Node(Leaf(value_counts)); // return a Leaf Node
	} 
	// when gain is not null we can partition further down the decision tree
	else { 
		// split the dataset S in two sets S1 and S2
		thepartition = partition(VecPtrVecI, thequestion, meta);
		right_VecPtrVecI = std::get<0>(thepartition); // true rows go on right S1
		left_VecPtrVecI = std::get<1>(thepartition); // false rows go on left S2
		// we only start threads if there are more than 25000 rows in the current dataset S before the split
		// this value has been found by testing on the university servers to be efficient to prevent 
		// the cpu overhead of starting threads for small datasets
		if (VecPtrVecI.size() > 25000) {
			// start two asynchronous threads for each side of the decision tree
			right_future = std::async(std::launch::async, buildTree, meta, right_VecPtrVecI);
			left_future = std::async(std::launch::async, buildTree, meta, left_VecPtrVecI);
			// retrieve the results of both threads 
			right_node = right_future.get();
			left_node = left_future.get();
		}
		// when there are less than 25000 rows in the dataset S we process the building of each side
		// of the decision tree in sequential order as it is too costly to start new threads
		else
		{
			right_node = buildTree(meta, right_VecPtrVecI);
			left_node = buildTree(meta, left_VecPtrVecI);
		}
		return Node(right_node, left_node, thequestion); // return a full Node with pointers to left and right nodes and the split question
	}
}

void DecisionTree::print() const {
	print(make_shared<Node>(root_));
}

void DecisionTree::print(const shared_ptr<Node> root, string spacing) const {
	if (bool is_leaf = root->leaf() != nullptr; is_leaf) {
		const auto& leaf = root->leaf();
		std::cout << spacing + "Predict: "; Utils::print::print_map(leaf->predictions());
		return;
	}
	std::cout << spacing << root->question().toString(dr_.metaData().labels) << "\n";

	std::cout << spacing << "--> True: " << "\n";
	print(root->trueBranch(), spacing + "   ");

	std::cout << spacing << "--> False: " << "\n";
	print(root->falseBranch(), spacing + "   ");
}

void DecisionTree::test() const {
	TreeTest t(dr_.testData(), dr_.metaData(), root_);
}

