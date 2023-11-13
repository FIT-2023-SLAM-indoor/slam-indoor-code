#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#pragma once
/**



KKKKKKKKK    KKKKKKKIIIIIIIIIIRRRRRRRRRRRRRRRRR   IIIIIIIIIILLLLLLLLLLL             LLLLLLLLLLL                  IIIIIIIIIIDDDDDDDDDDDDD      IIIIIIIIII     NNNNNNNN        NNNNNNNN               AAA               HHHHHHHHH     HHHHHHHHH
K:::::::K    K:::::KI::::::::IR::::::::::::::::R  I::::::::IL:::::::::L             L:::::::::L                  I::::::::ID::::::::::::DDD   I::::::::I     N:::::::N       N::::::N              A:::A              H:::::::H     H:::::::H
K:::::::K    K:::::KI::::::::IR::::::RRRRRR:::::R I::::::::IL:::::::::L             L:::::::::L                  I::::::::ID:::::::::::::::DD I::::::::I     N::::::::N      N::::::N             A:::::A             H:::::::H     H:::::::H
K:::::::K   K::::::KII::::::IIRR:::::R     R:::::RII::::::IILL:::::::LL             LL:::::::LL                  II::::::IIDDD:::::DDDDD:::::DII::::::II     N:::::::::N     N::::::N            A:::::::A            HH::::::H     H::::::HH
KK::::::K  K:::::KKK  I::::I    R::::R     R:::::R  I::::I    L:::::L                 L:::::L                      I::::I    D:::::D    D:::::D I::::I       N::::::::::N    N::::::N           A:::::::::A             H:::::H     H:::::H
  K:::::K K:::::K     I::::I    R::::R     R:::::R  I::::I    L:::::L                 L:::::L                      I::::I    D:::::D     D:::::DI::::I       N:::::::::::N   N::::::N          A:::::A:::::A            H:::::H     H:::::H
  K::::::K:::::K      I::::I    R::::RRRRRR:::::R   I::::I    L:::::L                 L:::::L                      I::::I    D:::::D     D:::::DI::::I       N:::::::N::::N  N::::::N         A:::::A A:::::A           H::::::HHHHH::::::H
  K:::::::::::K       I::::I    R:::::::::::::RR    I::::I    L:::::L                 L:::::L                      I::::I    D:::::D     D:::::DI::::I       N::::::N N::::N N::::::N        A:::::A   A:::::A          H:::::::::::::::::H
  K:::::::::::K       I::::I    R::::RRRRRR:::::R   I::::I    L:::::L                 L:::::L                      I::::I    D:::::D     D:::::DI::::I       N::::::N  N::::N:::::::N       A:::::A     A:::::A         H:::::::::::::::::H
  K::::::K:::::K      I::::I    R::::R     R:::::R  I::::I    L:::::L                 L:::::L                      I::::I    D:::::D     D:::::DI::::I       N::::::N   N:::::::::::N      A:::::AAAAAAAAA:::::A        H::::::HHHHH::::::H
  K:::::K K:::::K     I::::I    R::::R     R:::::R  I::::I    L:::::L                 L:::::L                      I::::I    D:::::D     D:::::DI::::I       N::::::N    N::::::::::N     A:::::::::::::::::::::A       H:::::H     H:::::H
KK::::::K  K:::::KKK  I::::I    R::::R     R:::::R  I::::I    L:::::L         LLLLLL  L:::::L         LLLLLL       I::::I    D:::::D    D:::::D I::::I       N::::::N     N:::::::::N    A:::::AAAAAAAAAAAAA:::::A      H:::::H     H:::::H
K:::::::K   K::::::KII::::::IIRR:::::R     R:::::RII::::::IILL:::::::LLLLLLLLL:::::LLL:::::::LLLLLLLLL:::::L     II::::::IIDDD:::::DDDDD:::::DII::::::II     N::::::N      N::::::::N   A:::::A             A:::::A   HH::::::H     H::::::HH
K:::::::K    K:::::KI::::::::IR::::::R     R:::::RI::::::::IL::::::::::::::::::::::LL::::::::::::::::::::::L     I::::::::ID:::::::::::::::DD I::::::::I     N::::::N       N:::::::N  A:::::A               A:::::A  H:::::::H     H:::::::H
K:::::::K    K:::::KI::::::::IR::::::R     R:::::RI::::::::IL::::::::::::::::::::::LL::::::::::::::::::::::L     I::::::::ID::::::::::::DDD   I::::::::I     N::::::N        N::::::N A:::::A                 A:::::A H:::::::H     H:::::::H
KKKKKKKKK    KKKKKKKIIIIIIIIIIRRRRRRRR     RRRRRRRIIIIIIIIIILLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL     IIIIIIIIIIDDDDDDDDDDDDD      IIIIIIIIII     NNNNNNNN         NNNNNNNAAAAAAA                   AAAAAAAHHHHHHHHH     HHHHHHHHH








*/
int videoProcessingCycle(cv::VideoCapture& cap, int featureTrackingBarier, int featureTrackingMaxAcceptableDiff,
	int framesGap, int requiredExtractedPointsCount, int featureExtractingThreshold, char* filename);