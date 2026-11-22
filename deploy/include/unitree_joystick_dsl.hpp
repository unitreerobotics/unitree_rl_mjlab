/**
 * Joystick DSL (Domain Specific Language) for Unitree Robotics Joystick
 *
 * This DSL allows users to define complex joystick button combinations and
 * interactions in a human-readable format. It supports logical expressions
 * combining button states with AND (+), OR (|), NOT (!), and parentheses.
 * Example expressions:
 *
 * --- Basic ---
 * - "A"                  # A is pressed
 * - "A.on_pressed"       # A is just pressed (single frame trigger)
 * - "A.on_released"      # A is just released (single frame trigger)
 *
 * --- Multi-key Combinations ---
 * - "A+B"                # A and B are pressed simultaneously
 * - "RB+X.on_pressed"    # RB is pressed + X is just pressed
 *
 * --- Directional Key Combinations ---
 * - "up+right"           # Up and right directions are pressed simultaneously (diagonal)
 *
 * --- Long Press Detection ---
 * - "LT(2s) + up"        # LT is pressed for more than 2 seconds and up is pressed
 * - "LT(3s).pressed"     # Equivalent to above (explicitly specifying .pressed)
 *
 * --- Multi-condition OR ---
 * - "X|Y"                # Either X or Y is pressed
 * - "A.on_pressed|B.on_pressed"  # Either A or B is just pressed
 *
 * --- Logical NOT ---
 * - "!A + B"             # A is not pressed and B is pressed
 * - "!(A + B)"           # A and B are not pressed simultaneously
 * - "!LT(1s)"            # LT is not pressed for 1 second (i.e., LT.pressed_time < 1 or not pressed)
 *
 * --- Nested Grouping ---
 * - "(A + B) | (X + Y)"  # A+B or X+Y is satisfied
 * - "!(A + B | X)"       # A+B or X is not allowed
 *
 * --- Mixed Directions and Buttons ---
 * - "LT + up.on_pressed" # LT is pressed + up is just pressed
 *
 * --- Multi-level Complex Combinations ---
 * - "((LT(1s) + up) | (RB + X.on_pressed)) + !Y"
 *   # LT long press + up or RB + X trigger, while Y is not pressed
 *
 * --- Axes and Trigger Keys ---
 * - "LX + LY"            # Left joystick exceeds threshold in any direction
 * - "RX(1s) + B"         # Right joystick holds beyond threshold for 1s + B is pressed
 *
 * --- Start/Exit Actions ---
 * - "start.on_pressed"   # Start button is just pressed
 * - "back.on_pressed"    # Back button is just pressed
 * - "!start + !back"     # Neither is pressed (commonly used for "idle state")
 *
 * --- Combined Long Press Actions (Complex Example) ---
 * - "(LT(2s) + RT(2s)) + A"
 *   # LT and RT are both long pressed for more than 2 seconds, and A is pressed
 */
#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cctype>
#include <memory>
#include <algorithm>
#include <yaml-cpp/yaml.h>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

namespace unitree::common::dsl {

// ======================== Lexical Analysis ========================
struct Token {
  enum Kind {
    kIdent, kNumber,
    kPlus, kOr, kNot,
    kLParen, kRParen,
    kDot, kEnd
  } kind;
  std::string text;
};

class Lexer {
 public:
  explicit Lexer(std::string s) : s_(s) {}
  Token Next() {
    SkipWs();
    if (pos_ >= s_.size()) return {Token::kEnd, ""};
    char c = s_[pos_];
    if (std::isalpha(static_cast<unsigned char>(c))) return Ident();
    if (std::isdigit(static_cast<unsigned char>(c))) return Number(); // Only support [1-9]
    ++pos_;
    switch (c) {
      case '+': return {Token::kPlus, "+"};
      case '|': return {Token::kOr, "|"};
      case '!': return {Token::kNot, "!"};
      case '(': return {Token::kLParen, "("};
      case ')': return {Token::kRParen, ")"};
      case '.': return {Token::kDot, "."};
      default:  throw std::runtime_error(std::string("Unexpected char: ") + std::string(1, c) + " near pos=" + std::to_string(pos_-1));
    }
  }
  size_t pos() const { return pos_; }

 private:
  void SkipWs() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
  }
  Token Ident() {
    size_t start = pos_;
    while (pos_ < s_.size() &&
           (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_]=='_'))
      ++pos_;
    return {Token::kIdent, std::string(s_.substr(start, pos_-start))};
  }
  Token Number() {
    size_t start = pos_;
    if (pos_ < s_.size() && s_[pos_] >= '1' && s_[pos_] <= '9') {
      ++pos_;
    } else {
      throw std::runtime_error("Expected a number starting with [1-9] near pos=" + std::to_string(pos_));
    }
    while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
      ++pos_;
    }
    return {Token::kNumber, std::string(s_.substr(start, pos_ - start))};
  }

  const std::string s_;
  size_t pos_{0};
};

// ======================== Abstract Syntax Tree (AST) & Semantics ========================
enum class Field { kPressed, kOnPressed, kOnReleased, kHoldTimeGE };

struct Atom {
  std::string name;         // Key name: "LT" "RB" "up" ...
  Field field{Field::kPressed};
  float hold_seconds{0.f};  // used when field==kHoldTimeGE
};

struct Node {
  enum Kind { kAtom, kNot, kAnd, kOr } kind{kAtom};
  Atom atom;                  // kAtom
  std::unique_ptr<Node> lhs;  // kNot: child is in lhs; kAnd/kOr: left
  std::unique_ptr<Node> rhs;  // kAnd/kOr: right
};

// Utility to convert strings to lowercase (used to make key names case-insensitive)
inline std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
  return s;
}

// Retrieve KeyBase from UnitreeJoystick (case-insensitive)
inline const KeyBase& GetKey(const UnitreeJoystick& joy, std::string_view name_sv) {
  const std::string name = ToLower(std::string{name_sv});
  static const std::unordered_map<std::string, const KeyBase* (*)(const UnitreeJoystick&)> kMap = {
    {"back", [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.back); }},
    {"start",[](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.start); }},
    {"ls",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.LS); }},
    {"rs",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.RS); }},
    {"lb",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.LB); }},
    {"rb",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.RB); }},
    {"a",    [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.A); }},
    {"b",    [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.B); }},
    {"x",    [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.X); }},
    {"y",    [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.Y); }},
    {"up",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.up); }},
    {"down", [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.down); }},
    {"left", [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.left); }},
    {"right",[](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.right); }},
    {"f1",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.F1); }},
    {"f2",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.F2); }},
    {"lx",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.lx); }},
    {"ly",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.ly); }},
    {"rx",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.rx); }},
    {"ry",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.ry); }},
    {"lt",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.LT); }},
    {"rt",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.RT); }},
  };
  auto it = kMap.find(name);
  if (it == kMap.end()) throw std::runtime_error("Unknown key name: " + std::string(name_sv));
  return *it->second(joy);
}

// ======================== Recursive Descent Parser ========================
// Supports: ! unary NOT; + logical AND; | logical OR; () grouping
// Atom syntax: name [ '(' number ['s'|'sec'|'secs'] ')' ] [ '.' (pressed|on_pressed|on_released) ]
class Parser {
 public:
  explicit Parser(std::string expr) : lex_(expr) {
    tok_ = lex_.Next();
  }
  std::unique_ptr<Node> Parse() {
    auto n = ParseOr();
    if (tok_.kind != Token::kEnd) {
      throw std::runtime_error("Trailing tokens near pos=" + std::to_string(lex_.pos()));
    }
    return n;
  }

 private:
  std::unique_ptr<Node> ParseOr() {
    auto left = ParseAnd();
    while (tok_.kind == Token::kOr) {
      Eat(Token::kOr);
      auto right = ParseAnd();
      auto n = std::make_unique<Node>();
      n->kind = Node::kOr;
      n->lhs = std::move(left);
      n->rhs = std::move(right);
      left = std::move(n);
    }
    return left;
  }
  std::unique_ptr<Node> ParseAnd() {
    auto left = ParseUnary();
    while (tok_.kind == Token::kPlus) {
      Eat(Token::kPlus);
      auto right = ParseUnary();
      auto n = std::make_unique<Node>();
      n->kind = Node::kAnd;
      n->lhs = std::move(left);
      n->rhs = std::move(right);
      left = std::move(n);
    }
    return left;
  }
  std::unique_ptr<Node> ParseUnary() {
    if (tok_.kind == Token::kNot) {
      Eat(Token::kNot);
      auto child = ParseUnary();
      auto n = std::make_unique<Node>();
      n->kind = Node::kNot;
      n->lhs = std::move(child);
      return n;
    }
    if (tok_.kind == Token::kLParen) {
      Eat(Token::kLParen);
      auto inside = ParseOr();
      Eat(Token::kRParen);
      return inside;
    }
    return ParseAtom();
  }

  std::unique_ptr<Node> ParseAtom() {
    if (tok_.kind != Token::kIdent) throw std::runtime_error("Expected identifier near pos=" + std::to_string(lex_.pos()));
    Atom a;
    a.name = tok_.text;
    Eat(Token::kIdent);

    // Optional hold duration: name '(' number ['s'|'sec'|'secs'] ')'
    if (tok_.kind == Token::kLParen) {
      Eat(Token::kLParen);
      if (tok_.kind != Token::kNumber) throw std::runtime_error("Expected hold seconds number near pos=" + std::to_string(lex_.pos()));
      a.hold_seconds = std::stof(tok_.text);
      Eat(Token::kNumber);
      // Optional unit
      if (tok_.kind == Token::kIdent) {
        std::string unit = ToLower(tok_.text);
        if (unit == "s" || unit == "sec" || unit == "secs") {
          Eat(Token::kIdent);
        } else {
          // Allow clearer error message for unknown or missing units
          throw std::runtime_error("Unknown time unit '" + tok_.text + "'; use 's'/'sec'");
        }
      }
      Eat(Token::kRParen);
      a.field = Field::kHoldTimeGE;
    }

    // Optional explicit state
    if (tok_.kind == Token::kDot) {
      Eat(Token::kDot);
      if (tok_.kind != Token::kIdent) throw std::runtime_error("Expected state after '.' near pos=" + std::to_string(lex_.pos()));
      const std::string st = tok_.text;
      if (st == "on_pressed")      a.field = Field::kOnPressed;
      else if (st == "on_released") a.field = Field::kOnReleased;
      else if (st == "pressed")     a.field = Field::kPressed;
      else throw std::runtime_error("Unknown field: " + st + " (allowed: pressed|on_pressed|on_released)");
      Eat(Token::kIdent);
    }

    auto n = std::make_unique<Node>();
    n->kind = Node::kAtom;
    n->atom = a;

    return n;
  }

  void Eat(Token::Kind k) {
    if (tok_.kind != k) {
      throw std::runtime_error("Unexpected token near pos=" + std::to_string(lex_.pos()));
    }
    tok_ = lex_.Next();
  }

  Lexer lex_;
  Token tok_{Token::kEnd, ""};
};

// ======================== Compile to Executable Predicate ========================
inline std::function<bool(const UnitreeJoystick&)> Compile(const Node& n) {
  switch (n.kind) {
    case Node::kAtom: {
      Atom a = n.atom;
      return [a](const UnitreeJoystick& joy) -> bool {
        const KeyBase& kb = GetKey(joy, a.name);
        switch (a.field) {
          case Field::kPressed:     return kb.pressed;
          case Field::kOnPressed:   return kb.on_pressed;
          case Field::kOnReleased:  return kb.on_released;
          case Field::kHoldTimeGE:  return kb.pressed && (kb.pressed_time >= a.hold_seconds);
        }
        return false;
      };
    }
    case Node::kNot: {
      auto child = Compile(*n.lhs);
      return [child](const UnitreeJoystick& joy){ return !child(joy); };
    }
    case Node::kAnd: {
      auto l = Compile(*n.lhs);
      auto r = Compile(*n.rhs);
      return [l, r](const UnitreeJoystick& joy){ return l(joy) && r(joy); };
    }
    case Node::kOr: {
      auto l = Compile(*n.lhs);
      auto r = Compile(*n.rhs);
      return [l, r](const UnitreeJoystick& joy){ return l(joy) || r(joy); };
    }
  }
  throw std::runtime_error("Invalid node kind");
}

} // namespace unitree::common::dsl
