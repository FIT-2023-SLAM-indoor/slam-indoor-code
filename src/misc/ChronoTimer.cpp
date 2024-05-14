#include "chrono"

#include "ChronoTimer.h"

ChronoTimer::ChronoTimer() {
	this->start = high_resolution_clock::now();
	this->lastPoint = high_resolution_clock::now();
}

void ChronoTimer::updateLastPoint() {
	this->lastPoint = high_resolution_clock::now();
}

void ChronoTimer::printLastPointDelta(const string &message, ostream &stream) {
	stream << message
			<< duration_cast<milliseconds>(
				high_resolution_clock::now() - this->lastPoint
			).count() << endl;
}

void ChronoTimer::printStartDelta(const string &message, ostream &stream) {
	stream << message
		   << duration_cast<milliseconds>(
			   high_resolution_clock::now() - this->start
		   ).count() << endl;
}
