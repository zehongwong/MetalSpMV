//
// Created by Zehong Wang on 2023-04-29.
//

#include "logger.h"
#include <iostream>

void Logger::info(const std::string &message) {
    std::cout << "[INFO] \033[32m" << message << "\033[0m" << std::endl;
}
void Logger::error(const std::string &message) {
    std::cout << "[ERROR] \033[31m" << message << "\033[0m" << std::endl;
}

void Logger::warn(const std::string &message) {
    std::cout << "[WARN] \033[33m" << message << "\033[0m" << std::endl;
}

void Logger::debug(const std::string &message) {
    std::cout << "[DEBUG] \033[33m" << message << "\033[0m" << std::endl;
}

void Logger::time(const std::string &message, high_resolution_clock::time_point start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[BENCHMARK]\033[34m " << message << ". \033[0m"
              << "\033[33m" << duration.count() << "ms" <<  "\033[0m\n";
}