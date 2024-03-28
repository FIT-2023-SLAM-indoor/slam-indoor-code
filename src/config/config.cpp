#include "fstream"
#include "iostream"

#include "config.h"

void ConfigService::setConfigFile(const std::string &pathToJSON) {
	std::ifstream jsonStream(pathToJSON);
	if (!jsonStream.is_open()) {
		std::cerr << "Failed to open config file" << std::endl;
		exit(2);
	}
	try {
		config = json::parse(jsonStream, nullptr, true, true);
	}
	catch (json::parse_error& error) {
		std::cerr << "Failed to parse JSON config" << std::endl
				<< "Make sure you specified path to JSON with correct semantics" << std::endl;
		exit(2);
	}
	checkJSON();
}

void ConfigService::checkJSON() {
	for (const auto & configField : configFields) {
		std::string typeStr = "strange";
		try {
			switch (configField.second.type) {
				case ConfigFieldType::BOOL:
					typeStr = "BOOLEAN";
					getValue<bool>(configField.first);
					break;
				case ConfigFieldType::INTEGER:
					typeStr = "INTEGER";
					getValue<int>(configField.first);
					break;
				case ConfigFieldType::FLOATING:
					typeStr = "FLOATING POINT NUMBER";
					getValue<double>(configField.first);
					break;
				case ConfigFieldType::STRING:
					typeStr = "STRING";
					getValue<std::string>(configField.first);
					break;
			}
		}
		catch (json::type_error& error) {
			std::cerr << "Field \"" << configField.second.key << "\" missed or has incorrect type!" << std::endl
					<< "Correct type is " << typeStr << std::endl;
			exit(2);
		}
	}
}
