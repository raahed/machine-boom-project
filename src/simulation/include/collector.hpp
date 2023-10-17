#ifndef COLLECTOR_HPP
#define COLLECTOR_HPP

#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace std;

class Collector {

private:
    /* class */
    string basepath_;
    string uniqueIdentifier_;
    time_t constructStamp_;
    string filename_;

    /* file */
    vector <string> headers_;
    bool setup_;

    /* stream */
    ofstream fileStream_;
    bool fileStreamOpen_;

    /* stream performance */
    int writeCounterMax;
    int writeCounter;

    Collector() {
        /* intern */
        constructStamp_ = time(0);

        /* FIX: Change delimiter char to use       */
        /*      python arrays inside a single cell */
        delimiter_ = ';';

        writeCounter = 0;
        writeCounterMax = 10;

        setup_ = false;
        fileStreamOpen_ = false;
    }


    void initFile() {

        ofstream file;

        /* clear file */
        file.open(getFilePath(), ofstream::trunc);
        file << rowBuilder(headers_);
        file.close();
    }


    string rowBuilder(vector <string> elements) {

        stringstream row;

        for (int i = 0; i < (elements.size() - 1); i++)
            row << elements[i] << delimiter_;

        row << elements[elements.size() - 1] << endl;

        return row.str();
    }


    Collector(const Collector &);

    Collector &operator=(const Collector &);

protected:
    char delimiter_;

public:
    ~Collector() { closeStream(); }

    /* singleton */
    static Collector &instance() {

        // Based on https ://stackoverflow.com/a/1008289
        static Collector instance;
        return instance;
    }

    void setup(string basepath, string identifierSuffix, vector <string> label) {

        if (setup_)
            return;

        basepath_ = basepath;

        tm *lm = localtime(&constructStamp_);
        uniqueIdentifier_ = to_string(lm->tm_year + 1900) + to_string(lm->tm_mon + 1) +
                            to_string(lm->tm_mday) + '_' + to_string(lm->tm_hour) +
                            to_string(lm->tm_min) + to_string(lm->tm_sec) + '_' +
                            identifierSuffix;

        filename_ = uniqueIdentifier_ + "_collection.csv";

        headers_ = {"Timestamp"};
        headers_.insert(headers_.end(), label.begin(), label.end());

        initFile();

        setup_ = true;
    }


    /* getter and setter methods */
    char getDelimiter() { return delimiter_; }

    void setDelimiter(char value) { delimiter_ = value; }

    string getUniqueIdentifier() { return uniqueIdentifier_; }

    string getFileName() { return filename_; }

    string getFilePath() { return basepath_ + '/' + filename_; }

    int getWriteCounterMax() { return writeCounterMax; }


    void closeStream() {

        if (!fileStreamOpen_)
            return;

        fileStream_.close();

        fileStreamOpen_ = false;
    }


    /* append methods */
    void append(vector <vector<double>> &data) {

        std::vector <string> flattend;

        for (auto element : data)
        {
            stringstream ss;

            for(auto doubleElement : element)
            {
                ss << doubleElement << ",";
            }

            std::string formattedString = ss.str();

            if (!formattedString.empty() && formattedString.back() == ',') {
                formattedString.pop_back();
            }

            formattedString = "[" + formattedString + "]";
            flattend.push_back(formattedString);
        }

        append(flattend);
    }


    void append(vector<double> &data) {

        vector <string> row;

        for (auto item: data)
            row.push_back(to_string(item));

        append(row);
    }


    void append(vector <string> &data) {

        /* init checks */
        if (data.size() + 1 != headers_.size())
            throw invalid_argument("Data count doesn't match label count!");

        /* open stream */
        if (!fileStreamOpen_) {
            fileStream_.open(getFilePath(), ios::app);
            fileStreamOpen_ = true;
        }

        /* prepare */
        vector <string> row;

        /* set timestamp */
        time_t now = time(0);

        row = {to_string(now)};
        row.insert(row.end(), data.begin(), data.end());

        /* write */
        fileStream_ << rowBuilder(row);

        /* performance: let stream open */
        if (++writeCounter == writeCounterMax) {
            closeStream();
            writeCounter = 0;
        }
    }

};

#endif // !COLLECTOR_HPP