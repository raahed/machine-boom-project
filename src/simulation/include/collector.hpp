#ifndef COLLECTOR_HPP
#define COLLECTOR_HPP

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
    /* class */
	string basepath_;
    string uniqueIdentifier_;
    time_t constructStamp_;
    string filename_;

    /* file */
	vector<string> headers_;
	bool setup_;

    /* stream */
	ofstream fileStream_;
	bool fileStreamOpen_;

    /* stream performance */
    int writeCounterMax;
    int writeCounter;

	Collector();

	void initFile();

	string rowBuilder(vector<string> elements);

	Collector(const Collector&);
	Collector& operator=(const Collector&);

protected:
	char delimiter_;

public:
	~Collector();

	/* singleton */
	static Collector& instance();

	void setup(string basepath, string identifierSuffix, vector<string> label);

	/* getter and setter methods */
	char getDelimiter();
	void setDelimiter(char value);
	string getUniqueIdentifier();
	string getFileName();
	string getFilePath();
    int getWriteCounterMax();

	void closeStream();

    /* append methods */
	void append(const vector<vector<double>>& data);
	void append(const vector<double>& data);
	void append(const vector<string>& data);
};


#endif // !COLLECTOR_HPP