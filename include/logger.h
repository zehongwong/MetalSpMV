//
// Created by Zehong Wang on 2023-04-29.
//
#ifndef METALSPMV_LOGGER_H
#define METALSPMV_LOGGER_H

#include <string>
using namespace std::chrono;

class Logger {
public:
    Logger() = default;

    ~Logger() = default;

    static void info(const std::string &msg);
    static void error(const std::string &msg);
    static void debug(const std::string &msg);
    static void warn(const std::string &msg);

    static void time(const std::string &msg, high_resolution_clock::time_point start_time);
};

#endif//METALSPMV_LOGGER_H
