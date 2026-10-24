#include <parallelforge/cli/command_parser.hpp>
#include <parallelforge/core/error.hpp>

namespace parallelforge::cli {

CommandParser::CommandParser(int argc, const char* const* argv) {
    parse(argc, argv);
}

void CommandParser::parse(int argc, const char* const* argv) {
    if (argc <= 1) {
        help_flag_ = true;
        return;
    }

    int i = 1;
    // Check if first arg is a command or an option
    std::string first = argv[1];
    if (first == "-h" || first == "--help") {
        help_flag_ = true;
        return;
    }
    if (first == "-v" || first == "--version") {
        version_flag_ = true;
        return;
    }

    if (first.rfind("--", 0) != 0 && first.rfind("-", 0) != 0) {
        command_ = first;
        i = 2;
    }

    for (; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            help_flag_ = true;
        } else if (arg == "-v" || arg == "--version") {
            version_flag_ = true;
        } else if (arg.rfind("--", 0) == 0) {
            std::string key = arg.substr(2);
            // Check for --key=value
            auto eq = key.find('=');
            if (eq != std::string::npos) {
                options_[key.substr(0, eq)] = key.substr(eq + 1);
            } else if (i + 1 < argc && argv[i + 1][0] != '-') {
                options_[key] = argv[++i];
            } else {
                options_[key] = "true";
            }
        } else if (arg.rfind("-", 0) == 0 && arg.size() > 1) {
            std::string key = arg.substr(1);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                options_[key] = argv[++i];
            } else {
                options_[key] = "true";
            }
        } else {
            // Positional parameter or unexpected argument
            throw core::InvalidConfigurationException("Unexpected argument: " + arg);
        }
    }
}

bool CommandParser::has_option(const std::string& key) const noexcept {
    return options_.find(key) != options_.end();
}

std::string CommandParser::get_string(const std::string& key, const std::string& default_val) const {
    auto it = options_.find(key);
    return it != options_.end() ? it->second : default_val;
}

std::size_t CommandParser::get_size(const std::string& key, std::size_t default_val) const {
    auto it = options_.find(key);
    if (it == options_.end()) return default_val;
    try {
        return std::stoull(it->second);
    } catch (...) {
        throw core::InvalidConfigurationException("Invalid integer value for --" + key + ": " + it->second);
    }
}

unsigned int CommandParser::get_uint(const std::string& key, unsigned int default_val) const {
    auto it = options_.find(key);
    if (it == options_.end()) return default_val;
    try {
        return static_cast<unsigned int>(std::stoul(it->second));
    } catch (...) {
        throw core::InvalidConfigurationException("Invalid unsigned integer value for --" + key + ": " + it->second);
    }
}

float CommandParser::get_float(const std::string& key, float default_val) const {
    auto it = options_.find(key);
    if (it == options_.end()) return default_val;
    try {
        return std::stof(it->second);
    } catch (...) {
        throw core::InvalidConfigurationException("Invalid floating-point value for --" + key + ": " + it->second);
    }
}

bool CommandParser::is_help() const noexcept {
    return help_flag_;
}

bool CommandParser::is_version() const noexcept {
    return version_flag_;
}

} // namespace parallelforge::cli
