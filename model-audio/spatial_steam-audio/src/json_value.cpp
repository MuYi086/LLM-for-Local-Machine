#include "json_value.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace llm_local::json {

Value::Value(Storage storage) : storage_(std::move(storage)) {}
bool Value::is_object() const { return std::holds_alternative<Object>(storage_); }
bool Value::is_array() const { return std::holds_alternative<Array>(storage_); }
bool Value::is_string() const { return std::holds_alternative<std::string>(storage_); }
bool Value::is_number() const { return std::holds_alternative<double>(storage_); }
bool Value::is_bool() const { return std::holds_alternative<bool>(storage_); }
bool Value::is_null() const { return std::holds_alternative<std::nullptr_t>(storage_); }

const Value::Object& Value::as_object() const {
    if (!is_object()) throw ParseError("expected JSON object");
    return std::get<Object>(storage_);
}
const Value::Array& Value::as_array() const {
    if (!is_array()) throw ParseError("expected JSON array");
    return std::get<Array>(storage_);
}
const std::string& Value::as_string() const {
    if (!is_string()) throw ParseError("expected JSON string");
    return std::get<std::string>(storage_);
}
double Value::as_number() const {
    if (!is_number()) throw ParseError("expected JSON number");
    return std::get<double>(storage_);
}
bool Value::as_bool() const {
    if (!is_bool()) throw ParseError("expected JSON boolean");
    return std::get<bool>(storage_);
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& source) : source_(source) {}

    Value parse_document() {
        auto value = parse_value(0);
        skip_space();
        if (position_ != source_.size()) fail("unexpected trailing content");
        return value;
    }

private:
    const std::string& source_;
    std::size_t position_{};

    [[noreturn]] void fail(const std::string& message) const {
        throw ParseError(message + " at byte " + std::to_string(position_));
    }

    void skip_space() {
        while (position_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[position_]))) {
            ++position_;
        }
    }

    char take() {
        if (position_ >= source_.size()) fail("unexpected end of JSON");
        return source_[position_++];
    }

    void expect(char expected) {
        if (take() != expected) fail(std::string("expected '") + expected + "'");
    }

    bool consume(char expected) {
        skip_space();
        if (position_ < source_.size() && source_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    Value parse_value(int depth) {
        if (depth > 64) fail("JSON nesting is too deep");
        skip_space();
        if (position_ >= source_.size()) fail("expected JSON value");
        switch (source_[position_]) {
            case '{': return parse_object(depth + 1);
            case '[': return parse_array(depth + 1);
            case '"': return Value(parse_string());
            case 't': parse_literal("true"); return Value(true);
            case 'f': parse_literal("false"); return Value(false);
            case 'n': parse_literal("null"); return Value(nullptr);
            default: return Value(parse_number());
        }
    }

    void parse_literal(const char* literal) {
        while (*literal) {
            if (take() != *literal++) fail("invalid JSON literal");
        }
    }

    Value parse_object(int depth) {
        expect('{');
        Value::Object object;
        if (consume('}')) return Value(std::move(object));
        while (true) {
            skip_space();
            if (position_ >= source_.size() || source_[position_] != '"') {
                fail("object key must be a string");
            }
            auto key = parse_string();
            skip_space();
            expect(':');
            auto [_, inserted] = object.emplace(std::move(key), parse_value(depth));
            if (!inserted) fail("duplicate object key");
            if (consume('}')) break;
            skip_space();
            expect(',');
        }
        return Value(std::move(object));
    }

    Value parse_array(int depth) {
        expect('[');
        Value::Array array;
        if (consume(']')) return Value(std::move(array));
        while (true) {
            array.push_back(parse_value(depth));
            if (consume(']')) break;
            skip_space();
            expect(',');
        }
        return Value(std::move(array));
    }

    static int hex_digit(char value) {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    }

    unsigned parse_hex_quad() {
        unsigned value = 0;
        for (int index = 0; index < 4; ++index) {
            const auto digit = hex_digit(take());
            if (digit < 0) fail("invalid Unicode escape");
            value = value * 16 + static_cast<unsigned>(digit);
        }
        return value;
    }

    static void append_utf8(std::string& output, unsigned codepoint) {
        if (codepoint <= 0x7f) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string output;
        while (true) {
            const auto value = take();
            if (value == '"') break;
            if (static_cast<unsigned char>(value) < 0x20) fail("control character in string");
            if (value != '\\') {
                output.push_back(value);
                continue;
            }
            const auto escaped = take();
            switch (escaped) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    auto codepoint = parse_hex_quad();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (take() != '\\' || take() != 'u') fail("invalid Unicode surrogate pair");
                        const auto low = parse_hex_quad();
                        if (low < 0xdc00 || low > 0xdfff) fail("invalid Unicode surrogate pair");
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        fail("unpaired Unicode surrogate");
                    }
                    append_utf8(output, codepoint);
                    break;
                }
                default: fail("invalid string escape");
            }
        }
        return output;
    }

    double parse_number() {
        const auto start = position_;
        if (position_ < source_.size() && source_[position_] == '-') ++position_;
        if (position_ >= source_.size()) fail("invalid number");
        if (source_[position_] == '0') {
            ++position_;
        } else if (std::isdigit(static_cast<unsigned char>(source_[position_]))) {
            while (position_ < source_.size() &&
                   std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_;
        } else {
            fail("invalid number");
        }
        if (position_ < source_.size() && source_[position_] == '.') {
            ++position_;
            if (position_ >= source_.size() ||
                !std::isdigit(static_cast<unsigned char>(source_[position_]))) fail("invalid number");
            while (position_ < source_.size() &&
                   std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_;
        }
        if (position_ < source_.size() &&
            (source_[position_] == 'e' || source_[position_] == 'E')) {
            ++position_;
            if (position_ < source_.size() &&
                (source_[position_] == '+' || source_[position_] == '-')) ++position_;
            if (position_ >= source_.size() ||
                !std::isdigit(static_cast<unsigned char>(source_[position_]))) fail("invalid exponent");
            while (position_ < source_.size() &&
                   std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_;
        }
        const auto token = source_.substr(start, position_ - start);
        char* end = nullptr;
        errno = 0;
        const auto value = std::strtod(token.c_str(), &end);
        if (errno == ERANGE || end != token.c_str() + token.size() || !std::isfinite(value)) {
            fail("number is not finite");
        }
        return value;
    }
};

}  // namespace

Value parse(const std::string& source) { return Parser(source).parse_document(); }

}  // namespace llm_local::json
