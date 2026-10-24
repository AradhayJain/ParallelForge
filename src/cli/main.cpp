#include <parallelforge/cli/command_parser.hpp>
#include <parallelforge/cli/commands.hpp>
#include <parallelforge/core/error.hpp>

#include <iostream>
#include <exception>

int main(int argc, char* argv[]) {
    try {
        parallelforge::cli::CommandParser parser(argc, argv);
        return parallelforge::cli::execute_command(parser);
    } catch (const parallelforge::core::ParallelForgeException& ex) {
        std::cerr << "ParallelForge Error: " << ex.what() << "\n";
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal Error: " << ex.what() << "\n";
        return 2;
    } catch (...) {
        std::cerr << "Fatal Unknown Exception\n";
        return 3;
    }
}
