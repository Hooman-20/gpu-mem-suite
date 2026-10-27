#include "csv.hpp"
#include "doctest.h"

#include <regex>
#include <sstream>
#include <stdexcept>

using namespace gpumem;

TEST_CASE("CSV writer emits comments, header and rows in order") {
  std::ostringstream out;
  CsvWriter csv(out);
  csv.comment("device", "Test GPU");
  csv.comment("l2_bytes", "1024");
  csv.header({"a", "b"});
  csv.row({"1", "2"});
  csv.row({"3", ""});
  CHECK(out.str() == "# device: Test GPU\n# l2_bytes: 1024\na,b\n1,2\n3,\n");
}

TEST_CASE("CSV escaping follows RFC 4180") {
  CHECK(CsvWriter::escape("plain") == "plain");
  CHECK(CsvWriter::escape("") == "");
  CHECK(CsvWriter::escape("a,b") == "\"a,b\"");
  CHECK(CsvWriter::escape("say \"hi\"") == "\"say \"\"hi\"\"\"");
  CHECK(CsvWriter::escape("two\nlines") == "\"two\nlines\"");
}

TEST_CASE("CSV writer escapes fields in rows") {
  std::ostringstream out;
  CsvWriter csv(out);
  csv.header({"name"});
  csv.row({"GeForce, \"big\""});
  CHECK(out.str() == "name\n\"GeForce, \"\"big\"\"\"\n");
}

TEST_CASE("CSV writer enforces structure") {
  std::ostringstream out;
  CsvWriter csv(out);
  CHECK_THROWS_AS(csv.row({"x"}), std::logic_error);
  csv.header({"a", "b"});
  CHECK_THROWS_AS(csv.row({"1"}), std::invalid_argument);
  CHECK_THROWS_AS(csv.row({"1", "2", "3"}), std::invalid_argument);
  CHECK_THROWS_AS(csv.comment("late", "x"), std::logic_error);
  CHECK_THROWS_AS(csv.header({"again"}), std::logic_error);
}

TEST_CASE("CSV comments reject newlines and colons in keys") {
  std::ostringstream out;
  CsvWriter csv(out);
  CHECK_THROWS_AS(csv.comment("a:b", "x"), std::invalid_argument);
  CHECK_THROWS_AS(csv.comment("k", "x\ny"), std::invalid_argument);
  CHECK_NOTHROW(csv.comment("k", "value: with colon"));
}

TEST_CASE("results path and timestamp format") {
  const std::string path = make_results_path("results", "cuda", "read", "20261027-153000");
  CHECK(path == "results/cuda_read_20261027-153000.csv");
  CHECK(std::regex_match(timestamp_now(), std::regex(R"(\d{8}-\d{6})")));
}

TEST_CASE("format_fixed") {
  CHECK(format_fixed(1.23456, 2) == "1.23");
  CHECK(format_fixed(2.0 / 3.0, 3) == "0.667");
  CHECK(format_fixed(1792.0, 1) == "1792.0");
}
