#pragma once

#include <parallelforge/cli/command_parser.hpp>

namespace parallelforge::cli {

int execute_command(const CommandParser& parser);

void print_help() noexcept;
void print_version() noexcept;

int cmd_info(const CommandParser& parser);
int cmd_matrix(const CommandParser& parser);
int cmd_convolution(const CommandParser& parser);
int cmd_vector(const CommandParser& parser);
int cmd_sort(const CommandParser& parser);
int cmd_memory(const CommandParser& parser);
int cmd_queue_demo(const CommandParser& parser);
int cmd_threadpool_demo(const CommandParser& parser);
int cmd_benchmark(const CommandParser& parser);
int cmd_validate(const CommandParser& parser);

} // namespace parallelforge::cli
