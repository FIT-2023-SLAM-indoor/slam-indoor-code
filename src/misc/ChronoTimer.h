#pragma once
#include "fstream"
#include "chrono"

using namespace std;
using namespace std::chrono;

class ChronoTimer {
private:
	system_clock::time_point start;
	system_clock::time_point lastPoint;
public:
	ChronoTimer();
	void updateLastPoint();
	void printLastPointDelta(const string &message, ofstream &stream);
	void printStartDelta(const string &message, ofstream &stream);
};
