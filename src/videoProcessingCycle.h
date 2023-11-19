#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#pragma once
/**
 *

						  _..._                                                                                                                                                                                  .-'''-.                                                                                                                                                                          .-'''-.
					   .-'_..._''.                                                                                                                                                                              '   _    \                                                                                                                                                                       '   _    \
	.--.             .' .'      '.\               _..._                     __.....__                            .--.  __  __   ___                                __.....__                                  /   /` '.   \                                                                                                               .--.    _..._                       .--.             /   /` '.   \
	|__|            / .'                        .'     '.               .-''         '.                          |__| |  |/  `.'   `.                          .-''         '.               .-.          .- .   |     \  '                                                                                           .--./)              |__|  .'     '.                     |__|   .--./)   .   |     \  '
	.--.           . '                         .   .-.   .             /     .-''"'-.  `.                   .|   .--. |   .-.  .-.   '                 .|     /     .-''"'-.  `.              \ \        / / |   '      |  '             .-,.--.                                                                     /.''\\               .--. .   .-.   .                    .--.  /.''\\    |   '      |  ' .-,.--.
	|  |           | |                  __     |  '   '  |            /     /________\   \                .' |_  |  | |  |  |  |  |  |     __        .' |_   /     /________\   \              \ \      / /  \    \     / /              |  .-. |               __                                           __     | |  | |       __     |  | |  '   '  |                    |  | | |  | |   \    \     / /  |  .-. |
	|  |           | |               .:--.'.   |  |   |  |            |                  |        _     .'     | |  | |  |  |  |  |  |  .:--.'.    .'     |  |                  |               \ \    / /    `.   ` ..' /      _    _   | |  | |            .:--.'.          _           _               .:--.'.    \`-' /     .:--.'.   |  | |  |   |  |                    |  |  \`-' /     `.   ` ..' /   | |  | |
	|  |           . '              / |   \ |  |  |   |  |            \    .-------------'      .' |   '--.  .-' |  | |  |  |  |  |  | / |   \ |  '--.  .-'  \    .-------------'                \ \  / /        '-...-'`      | '  / |  | |  | |           / |   \ |       .' |        .' |             / |   \ |   /("'`     / |   \ |  |  | |  |   |  |  ,.--.             |  |  /("'`         '-...-'`    | |  | |
	|  |            \ '.          . `" __ | |  |  |   |  |             \    '-.____...---.     .   | /    |  |   |  | |  |  |  |  |  | `" __ | |     |  |     \    '-.____...---.                 \ `  /                      .' | .' |  | |  '-            `" __ | |      .   | /     .   | /           `" __ | |   \ '---.   `" __ | |  |  | |  |   |  | //    \            |  |  \ '---.                   | |  '-
	|__|             '. `._____.-'/  .'.''| |  |  |   |  |              `.             .'    .'.'| |//    |  |   |__| |__|  |__|  |__|  .'.''| |     |  |      `.             .'                   \  /                       /  | /  |  | |                 .'.''| |    .'.'| |//   .'.'| |//            .'.''| |    /'""'.\   .'.''| |  |__| |  |   |  | \\     |           |__|   /'""'.\                  | |
					   `-.______ /  / /   | |_ |  |   |  |                `''-...... -'    .'.'.-'  /     |  '.'                       / /   | |_    |  '.'      `''-...... -'                     / /                       |   `'.  |  | |                / /   | |_ .'.'.-'  /  .'.'.-'  /            / /   | |_  ||     || / /   | |_      |  |   |  |  `'-) /                  ||     ||                 | |
								`   \ \._,\ '/ |  |   |  |                                 .'   \_.'      |   /                        \ \._,\ '/    |   /                                     |`-' /                        '   .'|  '/ |_|                \ \._,\ '/ .'   \_.'   .'   \_.'             \ \._,\ '/  \'. __//  \ \._,\ '/      |  |   |  |    /.'                   \'. __//                  |_|
									 `--'  `"  '--'   '--'                                                `'-'                          `--'  `"     `'-'                                       '..'                          `-'  `--'                      `--'  `"                                     `--'  `"    `'---'    `--'  `"       '--'   '--'                           `'---'


 * Main video processing cycle.
 * @param cap Video
 * @param featureTrackingBarier This parameter is responsible for changing the density of circular batches in feature tracking.Imho, the best value is 20.
 * @param featureTrackingMaxAcceptableDiff This parameter is responsible for regulating the difference between two tracked batches. The more this value the bigger the difference can be
 * @param framesBatchSize This parameter is the size of frames pool where we trying to find best for tracking.
 * @param requiredExtractedPointsCount This parameter is responsible for choosing frames for tracking. If extracted points count is too small, we skip the frame(go to another in our pool)
 * @param featureExtractingThreshold (The maximal acceptable difference beetwen points in feature finding.
 * @param reportsDirPath path to dir for reports' files WITHOUT last /. Dir must exist!
 * @return 0 if everything is ok, -1 otherwise.
 */
int videoProcessingCycle(cv::VideoCapture& cap, int featureTrackingBarier, int featureTrackingMaxAcceptableDiff,
	int framesBatchSize, int requiredExtractedPointsCount, int featureExtractingThreshold, char* reportsDirPath);