#include <gtest/gtest.h>
#include <parallelforge/cli/command_parser.hpp>
#include <parallelforge/cli/commands.hpp>
#include <parallelforge/core/error.hpp>

using namespace parallelforge::cli;

TEST(CliIntegrationTest, HelpAndVersion) {
    const char* argv_help[] = {"parallelforge", "--help"};
    CommandParser p_help(2, argv_help);
    EXPECT_TRUE(p_help.is_help());
    EXPECT_EQ(execute_command(p_help), 0);

    const char* argv_ver[] = {"parallelforge", "--version"};
    CommandParser p_ver(2, argv_ver);
    EXPECT_TRUE(p_ver.is_version());
    EXPECT_EQ(execute_command(p_ver), 0);
}

TEST(CliIntegrationTest, InfoCommand) {
    const char* argv[] = {"parallelforge", "info"};
    CommandParser p(2, argv);
    EXPECT_EQ(p.command(), "info");
    EXPECT_EQ(execute_command(p), 0);
}

TEST(CliIntegrationTest, MatrixCommandWithVerify) {
    const char* argv[] = {"parallelforge", "matrix", "--size", "64", "--algorithm", "blocked", "--tile-size", "16", "--verify"};
    CommandParser p(9, argv);
    EXPECT_EQ(p.command(), "matrix");
    EXPECT_EQ(p.get_size("size"), 64u);
    EXPECT_EQ(execute_command(p), 0);
}

TEST(CliIntegrationTest, ConvolutionCommandWithVerify) {
    const char* argv[] = {"parallelforge", "convolution", "--width", "64", "--height", "64", "--kernel", "blur", "--algorithm", "threads", "--verify"};
    CommandParser p(11, argv);
    EXPECT_EQ(p.command(), "convolution");
    EXPECT_EQ(execute_command(p), 0);
}

TEST(CliIntegrationTest, VectorCommand) {
    const char* argv[] = {"parallelforge", "vector", "--size", "1000", "--operation", "dot", "--algorithm", "threads"};
    CommandParser p(8, argv);
    EXPECT_EQ(p.command(), "vector");
    EXPECT_EQ(execute_command(p), 0);
}

TEST(CliIntegrationTest, SortCommandWithVerify) {
    const char* argv[] = {"parallelforge", "sort", "--size", "5000", "--algorithm", "threads", "--verify"};
    CommandParser p(7, argv);
    EXPECT_EQ(p.command(), "sort");
    EXPECT_EQ(execute_command(p), 0);
}

TEST(CliIntegrationTest, QueueDemoCommand) {
    const char* argv[] = {"parallelforge", "queue-demo", "--producers", "2", "--consumers", "2", "--items", "500", "--capacity", "10"};
    CommandParser p(10, argv);
    EXPECT_EQ(p.command(), "queue-demo");
    EXPECT_EQ(execute_command(p), 0);
}

TEST(CliIntegrationTest, ThreadPoolDemoCommand) {
    const char* argv[] = {"parallelforge", "threadpool-demo", "--workers", "2", "--tasks", "100"};
    CommandParser p(6, argv);
    EXPECT_EQ(p.command(), "threadpool-demo");
    EXPECT_EQ(execute_command(p), 0);
}

TEST(CliIntegrationTest, ValidateCommand) {
    const char* argv[] = {"parallelforge", "validate"};
    CommandParser p(2, argv);
    EXPECT_EQ(p.command(), "validate");
    EXPECT_EQ(execute_command(p), 0);
}

TEST(CliIntegrationTest, InvalidCommandReturnsError) {
    const char* argv[] = {"parallelforge", "nonexistent_command"};
    CommandParser p(2, argv);
    EXPECT_EQ(execute_command(p), 1);
}
