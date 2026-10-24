#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <stdexcept>

namespace parallelforge::cli {

class CommandParser {
public:
    CommandParser(int argc, const char* const* argv);

    [[nodiscard]] const std::string& command() const noexcept { return command_; }
    [[nodiscard]] bool has_option(const std::string& key) const noexcept;
    
    [[nodiscard]] std::string get_string(const std::string& key, const std::string& default_val = "") const;
    [[nodiscard]] std::size_t get_size(const std::string& key, std::size_t default_val = 0) const;
    [[nodiscard]] unsigned int get_uint(const std::string& key, unsigned int default_val = 0) const;
    [[nodiscard]] float get_float(const std::string& key, float default_val = 0.0f) const;

    [[nodiscard]] bool is_help() const noexcept;
    [[nodiscard]] bool is_version() const noexcept;

private:
    std::string command_;
    std::unordered_map<std::string, std::string> options_;
    bool help_flag_{false};
    bool version_flag_{false};

    void parse(int argc, const char* const* argv);
};

} // namespace parallelforge::cli
