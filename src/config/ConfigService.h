#pragma once
#include "nlohmann/json.hpp"
#include "configData.h"

using json = nlohmann::json;

class ConfigService {
private:
	json config;

	void checkJSON();
public:
	void setConfigFile(const std::string &pathToJSON);

	template<typename T>
	T getValue(ConfigFieldEnum enumKey) {
		return config[configFields.at(enumKey).key].template get<T>();
	}
};

