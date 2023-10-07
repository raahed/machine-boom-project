#include "collector.hpp"

#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <iostream>
#include <fstream>

Collector::Collector() {

    /* intern */
    constructStamp_ = time(0);

    /* FIX: Change delimiter char to use       */
    /*      python arrays inside a single cell */
    delimiter_ = ';';

    writeCounter = 0;
    const writeCounterMax = 10;

    setup_ = false;
    fileStreamOpen_ = false;
}

Collector::~Collector() { closeStream(); }

static Collector& Collector::instance() {

    // Based on https ://stackoverflow.com/a/1008289
    static Collector instance;
    return instance;
}

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

void Collector::append(const vector<vector<double>>& data) {

    std::vector<string> flattend;

    for (auto elements : data)
    {
        stringstream row = '[';

        for (int i = 0; i < (elements.size() - 1); i++)
            row << elements[i] << ',';

        row << elements[elements.size() - 1] << ']' << endl;

        flattend.push_back(row.str())
    }
    append(flattened);
}

void Collector::append(const vector<double>& data) {

    vector<string> row;

    /* Based on https://stackoverflow.com/a/25371915 */
    transform(begin(data),
              end(data),
              back_inserter(row),
              [](double d) { return to_string(d)}
    );

    append(row);
}

void Collector::append(const vector<string>& data) {

    /* init checks */
    if (data.size() + 1 != headers_.size())
        throw "Data count doesn't match label count!";

    /* open stream */
    if (!fileStreamOpen_) {
        fileStream_.open(getFilePath(), ios::app);
        fileStreamOpen_ = true;
    }

    /* prepare */
    vector<string> row;

    /* set timestamp */
    time_t now = time(0);

    row = { to_string(now) };
    row.insert(row.end(), data.begin(), data.end());

    /* write */
    fileStream_ << rowBuilder(row);

    /* performance: let stream open */
    if(++writeCounter == writeCounterMax) {
        closeStream();
        writeCounter = 0;
    }
}

char Collector::getDelimiter() { return delimiter_; }
void Collector::setDelimiter(char value) { delimiter_ = value; }
string Collector::getUniqueIdentifier() { return uniqueIdentifier_; }
string Collector::getFileName() { return filename_; }
string Collector::getFilePath() { return basepath_ + '/' + filename_; }
int Collector::getWriteCounterMax() { return writeCounterMax; }