#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace llm_local::json {

class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Value {
public:
    using Object = std::map<std::string, Value>;
    using Array = std::vector<Value>;
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

    explicit Value(Storage storage);

    bool is_object() const;
    bool is_array() const;
    bool is_string() const;
    bool is_number() const;
    bool is_bool() const;
    bool is_null() const;
    const Object& as_object() const;
    const Array& as_array() const;
    const std::string& as_string() const;
    double as_number() const;
    bool as_bool() const;

private:
    Storage storage_;
};

Value parse(const std::string& source);

}  // namespace llm_local::json
