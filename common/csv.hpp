#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace gpumem {

// Minimal RFC 4180 writer with a "# key: value" metadata preamble.
// Order: comment()* -> header() -> row()*. Each row is flushed so a crash keeps earlier sizes.
class CsvWriter {
 public:
  explicit CsvWriter(std::ostream& out) : out_(out) {}

  void comment(const std::string& key, const std::string& value);
  void header(const std::vector<std::string>& columns);
  void row(const std::vector<std::string>& fields);

  static std::string escape(const std::string& field);

 private:
  void write_fields(const std::vector<std::string>& fields);

  std::ostream& out_;
  size_t columns_ = 0;
};

// Fixed-precision decimal ("%.<precision>f").
std::string format_fixed(double value, int precision);

// Local time as YYYYMMDD-HHMMSS.
std::string timestamp_now();

// <dir>/<backend>_<test>_<timestamp>.csv
std::string make_results_path(const std::string& dir, const std::string& backend, const std::string& test,
                              const std::string& timestamp);

}  // namespace gpumem
