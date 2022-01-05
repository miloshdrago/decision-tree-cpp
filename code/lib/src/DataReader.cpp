#include <thread>
#include "DataReader.hpp"

using boost::algorithm::split;
using boost::timer::cpu_timer;

DataReader::DataReader(const Dataset& dataset) :
	classLabel_(dataset.classLabel),
	trainData_({}),
	testData_({}),
	trainMetaData_({}),
	testMetaData_({}) {
	std::cout << "Start reading data set." << std::endl; cpu_timer timer;
	std::thread readTestingData([this, &dataset]() {
		return processFile(dataset.train.filename, trainData_, trainMetaData_);
		});

	std::thread readTrainingData([this, &dataset]() {
		return processFile(dataset.test.filename, testData_, testMetaData_);
		});

	readTrainingData.join();
	readTestingData.join();
	std::cout << "Done. " << timer.format() << std::endl;

	if (!classLabel_.empty())
		moveClassLabelToBack();

	if (trainData_.empty())
		throw std::runtime_error("Can't open file: " + dataset.train.filename);

	if (testData_.empty())
		throw std::runtime_error("Can't open file: " + dataset.test.filename);

	// fill in trainDataInt table with the trainData information converted to int format
	InitializeDataInt(trainData_, trainMetaData_);
}

void DataReader::processFile(const std::string& filename, Data& data, MetaData& meta) {
	std::ifstream file(filename);
	if (!file)
		return;

	std::string line;
	bool header_loaded = false;

	while (getline(file, line)) {
		if (!header_loaded) {
			parseHeaderLine(line, meta, header_loaded);
		}
		else {
			parseDataLine(line, data, meta);
		}
	}
	file.close();
	// Trim white spaces from colum names
	trimWhiteSpaces(meta.labels);
}

bool DataReader::parseHeaderLine(const std::string& line, MetaData& meta, bool& header_loaded) {
	if (line.size() == 0) {
		return true;
	}

	if (line[line.find_first_not_of(" ")] == '%') {
		return true;
	}

	if (line.find_first_not_of(" \n\r\t") == line.npos) {
		return true;
	}

	std::string s = line;
	s.erase(0, s.find_first_not_of(" \n\r\t"));
	s.erase(s.find_last_not_of(" \n\r\t") + 1);
	int len = 0;

	len = std::string("@RELATION ").size();
	if (s.size() > (size_t)len
		&& strcasecmp(s.substr(0, len).c_str(), "@RELATION ") == 0) {
		return true;
	}

	len = std::string("@ATTRIBUTE ").size();
	if (s.size() > (size_t)len
		&& strcasecmp(s.substr(0, len).c_str(), "@ATTRIBUTE ") == 0) {
		s.erase(0, len);
		s.erase(0, s.find_first_not_of(" \n\r\t"));
		len = std::string(" NUMERIC").size();
		if (s.size() > (size_t)len
			&& strcasecmp(s.substr(s.size() - len, len).c_str(), " NUMERIC") == 0) {
			s = s.substr(0, s.size() - len);
			meta.labels.push_back(s);
			// initialize additional metadata
			meta.isnumeric.push_back(true); // column information is of numeric type
			meta.mapS2I.push_back({}); // no mappings from String to int needed
			meta.mapI2S.push_back({}); // no mapping from int to String needed
			return true;
		}

		len = std::string(" REAL").size();
		if (s.size() > (size_t)len
			&& strcasecmp(s.substr(s.size() - len, len).c_str(), " REAL") == 0) {
			s = s.substr(0, s.size() - len);
			meta.labels.push_back(s);
			// initialize additional metadata
			meta.isnumeric.push_back(true); // column information is of numeric type
			meta.mapS2I.push_back({}); // no mappings from String to int needed
			meta.mapI2S.push_back({}); // no mapping from int to String needed
			return true;
		}

		{
			// Here we look for the possible values of a categorical string
			// like in tennis.arff : @attribute outlook { Sunny, Overcast, Rain }
			size_t pos = s.find_last_of("{");
			size_t pos2 = s.find_last_of("}");
			// string to hold the possible values ex : "Sunny, Overcast, Rain"
			std::string str_values;
			str_values = s.substr((pos + 1), (pos2 - pos - 1));
			// vector of strings to hold the possible attribute values
			std::vector<std::string> vec_values;
			// convert the comma separated values in the string to a vector of strings
			split(vec_values, str_values, boost::is_any_of(","));
			// remove white spaces
			trimWhiteSpaces(vec_values);
			s = s.substr(0, pos);
			meta.labels.push_back(s); 
			// initialize additional metadata
			meta.isnumeric.push_back(false); // column information is of categorical type
			// create a mapping table from string value to int value
			std::unordered_map<std::string, int> map1;
			// create a mapping table from int value to string value
			std::unordered_map<int, std::string> map2;
			// create the mappings
			// example from tennis.arff file : @attribute outlook { Sunny, Overcast, Rain } 
			for (size_t i = 0; i < vec_values.size(); i++) {
				map1.insert({ vec_values.at(i),i });
				map2.insert({ i,vec_values.at(i) });
			}
			// would map in mapS2I : Sunny to 0, Overcast to 1 and Rain to 2
			meta.mapS2I.push_back(map1);
			// would map in mapI2S: 0 to Sunny, 1 to Overcast and 2 to Rain
			meta.mapI2S.push_back(map2);
			return true;
		}
		return true;
	}

	len = std::string("@DATA").size();
	if (s.size() >= (size_t)len
		&& strcasecmp(s.substr(0, len).c_str(), "@DATA") == 0) {
		if (meta.labels.size() > 0) {
			header_loaded = true;
			return true;
		}
		else {
			return false;
		}
	}

	std::cout << "Symbol not defind " << s.c_str() << std::endl;
	return true;
}


bool DataReader::parseDataLine(const std::string& line, Data& data, MetaData& meta) {
	std::vector<std::string> vec;
	split(vec, line, boost::is_any_of(","));
	trimWhiteSpaces(vec);
	// check that the data line contains as many fields as there are columns in the metadata
	if (vec.size() == meta.labels.size()) {
		if (classLabel_.empty()) {
			data.emplace_back(std::move(vec));
		}
		else {
			moveClassDataToBack(vec, meta.labels);
			data.emplace_back(std::move(vec));
		}
	}
	else {
		std::cout << "Data line does not have same number of fields as the number of attributes\n" << line << "\n";
	}
	return true;
}

void DataReader::moveClassLabelToBack() {
	const auto result = std::find(std::begin(trainMetaData_.labels), std::end(trainMetaData_.labels), classLabel_);
	if (result != std::end(metaData().labels))
		std::iter_swap(result, std::end(trainMetaData_.labels) - 1);
}

void DataReader::moveClassDataToBack(VecS& line, const VecS& labels) const {
	static const auto result = std::find(std::begin(labels), std::end(labels), classLabel_);
	if (result != std::end(labels)) {
		static const auto result_index = std::distance(std::begin(labels), result);
		std::iter_swap(std::begin(line) + result_index, std::end(line) - 1);
	}
}

void DataReader::trimWhiteSpaces(VecS& line) {
	for (auto& val : line)
		boost::trim(val);
}

// Function to convert trainData from std::vector<std::vector<std::string>> to std::vector<std::vector<int>>
void DataReader::InitializeDataInt(Data& data, const MetaData& meta) {
	VecI empty_row; // An empty vector of int, used as a placeholder to generate the empty table
	// Initialize the empty table of DataInt based on the size of data and the number of columns
	trainDataInt_.clear();
	// Reserve in memory a vector of capacity to hold all the rows 
	trainDataInt_.reserve(data.size());
	// Reserve the empty vector of int to be able to hold information of all columns 
	empty_row.resize(meta.labels.size());
	// Push the empty row to the table as many times as there are data rows
	for (size_t row = 0; row < data.size(); row++) {
		trainDataInt_.push_back(empty_row);
	}
	// Now fill in the trainDataInt table with the trainData information converted to int format
	for (size_t col = 0; col < meta.labels.size(); col++) {
		// check if the column is of numeric type
		if (meta.isnumeric.at(col)) {
			for (size_t row = 0; row < data.size(); row++) {
				// the string representing a number is converted to an int
				trainDataInt_[row].at(col) = stoi(data[row].at(col));
			}
		}
		else {
			for (size_t row = 0; row < data.size(); row++) {
				// The string of a categorical field is comnverted to a mapped int value
				// this is based on mappings saved in the MetaData and read from the attribute possible values enumeration
				// example from tennis.arff file : @attribute outlook { Sunny, Overcast, Rain } 
				// would map : Sunny to 0, Overcast to 1 and Rain to 2
				trainDataInt_[row].at(col) = meta.mapS2I[col].at(data[row].at(col));
			}
		}
	}
}
