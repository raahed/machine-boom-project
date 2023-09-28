#ifndef COLLECTOR_H
#define COLLECTOR_H

#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <iostream>
#include <fstream>

using namespace std;

class Collector {

private:
	string basepath_;
	vector<string> headers_;
	string uniqueIdentifier_;
	string filename_;
	time_t constructStamp_;
	bool setup_;
	ofstream fileStream_;
	bool fileStreamOpen_;

	Collector() {
		/* intern */
		constructStamp_ = time(0);
		delimiter_ = ',';
		setup_ = false;
		fileStreamOpen_ = false;
	}

	void initFile();

	string rowBuilder(vector<string> elements);

	Collector(const Collector&);
	Collector& operator=(const Collector&);

protected:
	char delimiter_;

public:
	~Collector() { closeStream(); }

	/* singleton */
	static Collector& instance() {

		// Based on https ://stackoverflow.com/a/1008289
		static Collector instance;
		return instance;
	}

	void setup(string basepath, string identifierSuffix, vector<string> label);

	/* getter and setter methods */
	char getDelimiter() { return delimiter_; }
	void setDelimiter(char value) { delimiter_ = value; }
	string getUniqueIdentifier() { return uniqueIdentifier_; }
	string getFileName() { return filename_; }
	string getFilePath() { return basepath_ + '/' + filename_; }

	void closeStream();

	void append(const vector<double>& data);
	void append(const vector<string>& data);
};

void Collector::initFile() {

	ofstream file;

	/* clear file */
	file.open(getFilePath(), ofstream::trunc);
	file << rowBuilder(headers_);
	file.close();
};


string Collector::rowBuilder(vector<string> elements) {

	stringstream row;

	for (int i = 0; i < (elements.size() - 1); i++)
		row << elements[i] << delimiter_;

	row << elements[elements.size() - 1] << endl;

	return row.str();
}

void Collector::setup(string basepath, string identifierSuffix, vector<string> label) {

	if (setup_)
		return;

	basepath_ = basepath;

	tm* lm = localtime(&constructStamp_);
	uniqueIdentifier_ = to_string(lm->tm_year + 1900) + to_string(lm->tm_mon + 1) +
		to_string(lm->tm_mday) + '_' + to_string(lm->tm_hour) +
		to_string(lm->tm_min) + to_string(lm->tm_sec) + '_' +
		identifierSuffix;

	filename_ = uniqueIdentifier_ + "_collection.csv";

	headers_ = { "Timestamp" };
	headers_.insert(headers_.end(), label.begin(), label.end());

	initFile();

	setup_ = true;
}
	
void Collector::closeStream() {
	if (!fileStreamOpen_)
		return;

	fileStream_.close();

	fileStreamOpen_ = false;
}

void Collector::append(const vector<double>& data) {
	vector<string> row;
	
	// TODO: Use lambda
	for (double item : data) 
		row.push_back(to_string(item));

	append(row);
}

void Collector::append(const vector<string>& data) {

	/* init checks */
	// TODO: Implement error handling
	//if (data.size() + 1 != headers_.size())
	//	return;

	if (!fileStreamOpen_) {
		fileStream_.open(getFilePath(), ios::app);
		fileStreamOpen_ = true;
	}

	/* prepare */
	time_t now = time(0);
	vector<string> row;
	row = { to_string(now) };
	row.insert(row.end(), data.begin(), data.end());

	/* write */
	fileStream_ << rowBuilder(row);

	// TODO: Performance improve, dont close stream after each line
	closeStream();
}

#endif // !COLLECTOR_H