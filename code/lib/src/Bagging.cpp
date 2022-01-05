#include "Bagging.hpp"

using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;

Bagging::Bagging(const DataReader& dr, const int ensembleSize, uint seed) :
	dr_(dr),
	ensembleSize_(ensembleSize),
	learners_({}) {
	random_number_generator.seed(seed);
	buildBag();
}


void Bagging::buildBag() {
	cpu_timer timer;
	std::vector<double> timings;
	// initialize a random generator to generate numbers from 0 up to the number of rows - 1
	std::uniform_int_distribution<int> distribution(0, dr_.trainDataInt().size() - 1);
	for (size_t i = 0; i < (size_t) ensembleSize_; i++) {
		DataInt bootstrap_int;
		timer.start();
		//TODO: Implement bagging
		//   Generate a bootstrap sample of the original data
		//   Train an unpruned tree model on this sample
		// 
		// generating a bootstrap sample of the original data in int format
		for (size_t row = 0; row < dr_.trainDataInt().size(); row++) {
			bootstrap_int.push_back(dr_.trainDataInt().at(distribution(random_number_generator)));
		}

		// training unpruned tree model on the bootstrap
		DecisionTree dt(dr_, bootstrap_int);
		// store the learned decision tree 
		learners_.emplace_back(dt);

		auto nanoseconds = boost::chrono::nanoseconds(timer.elapsed().wall);
		auto seconds = boost::chrono::duration_cast<boost::chrono::seconds>(nanoseconds);
		timings.push_back(seconds.count());
	}
	float avg_timing = Utils::iterators::average(std::begin(timings), std::begin(timings) + std::min(5, ensembleSize_));
	std::cout << "Average timing: " << avg_timing << std::endl;
}

void Bagging::test() const {
	TreeTest t;
	float accuracy = 0;
	for (const auto& row : dr_.testData()) {
		static size_t last = row.size() - 1;
		std::vector<std::string> decisions;
		for (int i = 0; i < ensembleSize_; i++) {
			const std::shared_ptr<Node> root = std::make_shared<Node>(learners_.at(i).root_);
			const auto& classification = t.classify(row, root);
			decisions.push_back(Utils::tree::getMax(classification));
		}
		std::string prediction = Utils::iterators::mostCommon(decisions.begin(), decisions.end());
		if (prediction == row[last])
			accuracy += 1;
	}
	std::cout << "Total accuracy: " << (accuracy / dr_.testData().size()) << std::endl;
}


