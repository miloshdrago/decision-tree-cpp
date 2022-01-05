#ifdef _MSC_VER
//not #if defined(_WIN32) || defined(_WIN64) because we have strncasecmp in mingw
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define uint uint32_t
#endif

#ifndef DECISIONTREE_ARFFREADER_HPP
#define DECISIONTREE_ARFFREADER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include "Dataset.hpp"
#include "Utils.hpp"

/**
 * Implementation of a parser for data sets in the ARFF format.
 *
 * The specification of the Attribute-Relation File Format (ARFF) can be found
 * at <https://www.cs.waikato.ac.nz/ml/weka/arff.html>.
 *
 * TODO : A working implementation is provided, although you might want to make
 * some changes to enable faster decision tree learning. The definition of the
 * public methods (including the constructor) can not be altered. All private
 * methods can be modified as you which.
 */
class DataReader
{
public:
	DataReader() = delete;
	DataReader(const Dataset& d);

	inline const Data& trainData() const { return trainData_; }
	inline const Data& testData() const { return testData_; }
	inline const MetaData& metaData() const { return trainMetaData_; }

	// function to retrieve the table containing the trainData information in int format
	inline const DataInt& trainDataInt() const { return trainDataInt_; }
private:
	void processFile(const std::string& strings, Data& data, MetaData& meta);
	void moveClassDataToBack(VecS& line, const VecS& labels) const;
	void moveClassLabelToBack();
	void trimWhiteSpaces(VecS& line);

	bool parseHeaderLine(const std::string& line, MetaData& meta, bool& header_loaded);
	bool parseDataLine(const std::string& line, Data& data, MetaData& meta);
	// function to transform the original Data table to DataInt in the trainDataInt_
	void InitializeDataInt(Data& data, const MetaData& meta);
 

	const std::string classLabel_;
	Data trainData_;
	Data testData_;
	MetaData trainMetaData_;
	MetaData testMetaData_;
	// Table containing the trainData information in int format
	DataInt trainDataInt_;
};

#endif //DECISIONTREE_ARFFREADER_HPP
