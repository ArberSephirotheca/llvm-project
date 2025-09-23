#pragma once

#include <optional>
#include <string>
#include <utility>

namespace simt_hlsl_import {

template <typename T> class Result {
public:
  using value_type = T;

  static Result ok(T value) { return Result(std::move(value)); }
  static Result err(std::string message) { return Result(std::move(message)); }

  bool succeeded() const { return storage.has_value(); }
  explicit operator bool() const { return succeeded(); }

  const std::string &error() const { return message; }

  T &value() { return storage.value(); }
  const T &value() const { return storage.value(); }

  template <typename F>
  auto map(F &&fn) -> Result<std::invoke_result_t<F, T &>> {
    using U = std::invoke_result_t<F, T &>;
    if (!succeeded())
      return Result<U>::err(message);
    return Result<U>::ok(fn(storage.value()));
  }

  template <typename F>
  auto andThen(F &&fn) -> decltype(fn(std::declval<T &>())) {
    using ReturnResult = decltype(fn(std::declval<T &>()));
    if (!succeeded())
      return ReturnResult::err(message);
    return fn(storage.value());
  }

private:
  explicit Result(T value) : storage(std::move(value)) {}
  explicit Result(std::string message)
      : storage(std::nullopt), message(std::move(message)) {}

  std::optional<T> storage;
  std::string message;
};

template <> class Result<void> {
public:
  using value_type = void;

  static Result ok() { return Result(true, ""); }
  static Result err(std::string message) { return Result(false, std::move(message)); }

  bool succeeded() const { return hasValue; }
  explicit operator bool() const { return succeeded(); }
  const std::string &error() const { return message; }

  template <typename F> Result &map(F &&fn) {
    if (succeeded())
      fn();
    return *this;
  }

  template <typename F> auto andThen(F &&fn) -> decltype(fn()) {
    using ReturnResult = decltype(fn());
    if (!succeeded())
      return ReturnResult::err(message);
    return fn();
  }

private:
  Result(bool ok, std::string message)
      : message(std::move(message)), hasValue(ok) {}

  std::string message;
  bool hasValue;
};

} // namespace simt_hlsl_import
