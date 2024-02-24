#include "IOmisc.h"

void sortGlobs(std::vector<String>& paths) {
    std::sort(paths.begin(), paths.end(), [](const String& a, const String& b){
        int aLen = a.length(), bLen = b.length();
        if (aLen != bLen)
            return aLen < bLen;
        int startDigitIndex = 5;
        int aNum = 0, bNum = 0, p = 1;
        while (isdigit(a[aLen - 1 - startDigitIndex])) {
            aNum += (a[aLen - 1 - startDigitIndex] - '0') * p;
            bNum += (b[aLen - 1 - startDigitIndex] - '0') * p;
            p *= 10;
            startDigitIndex++;
        }
        return a < b;
    });
}