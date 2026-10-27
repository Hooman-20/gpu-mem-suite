#include "csv.hpp"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <stdexcept>

namespace gpumem {

void CsvWriter::comment(const std::string& key, const std::string& value) {
  if (columns_ != 0) throw std::logic_error("CSV comments must come before the header");
  if (key.find_first_of(":\n") != std::string::npos || value.find('\n') != std::string::npos) {
    throw std::invalid_argument("CSV comment key/value may not contain newlines (or ':' in the key)");
  }
  out_ << "# " << key << ": " << value << '\n';
}

void CsvWriter::header(const std::vector<std::string>& columns) {
  if (columns_ != 0) throw std::logic_error("CSV header written twice");
  if (columns.empty()) throw std::invalid_argument("CSV header needs at least one column");
  columns_ = columns.size();
  write_fields(columns);
}

void CsvWriter::row(const std::vector<std::string>& fields) {
  if (columns_ == 0) throw std::logic_error("CSV row written before the header");
  if (fields.size() != columns_) {
    throw std::invalid_argument("CSV row has " + std::to_string(fields.size()) + " fields, header has " +
                                std::to_string(columns_));
  }
  write_fields(fields);
}

std::string CsvWriter::escape(const std::string& field) {
  if (field.find_first_of(",\"\r\n") == std::string::npos) return field;
  std::string quoted = "\"";
  for (char c : field) {
    if (c == '"') quoted += '"';
    quoted += c;
  }
  quoted += '"';
  return quoted;
}

void CsvWriter::write_fields(const std::vector<std::string>& fields) {
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i) out_ << ',';
    out_ << escape(fields[i]);
  }
  out_ << '\n';
  out_.flush();
}

std::string format_fixed(double value, int precision) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", precision, value);
  return buf;
}

std::string timestamp_now() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
#if defined(_WIN32)
  localtime_s(&local, &now);
#else
  localtime_r(&now, &local);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &local);
  return buf;
}

std::string make_results_path(const std::string& dir, const std::string& backend, const std::string& test,
                              const std::string& timestamp) {
  return (std::filesystem::path(dir) / (backend + "_" + test + "_" + timestamp + ".csv")).string();
}

}  // namespace gpumem
