# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

add_library(
  io STATIC
  Io.cpp
  Io.h
  Json.cpp
  Json.h
  JsonAllTypes.h
  JsonBuiltinTypes.h
  JsonEspTypes.cpp
  JsonEspTypes.h
  JsonMagnumTypes.cpp
  JsonMagnumTypes.h
  JsonStlTypes.cpp
  JsonStlTypes.h
  JsonUtils.h
  URDFParser.cpp
  URDFParser.h
)

target_include_directories(io PRIVATE "${DEPS_DIR}/tinyxml2")

target_link_libraries(
  io
  PUBLIC core
  PRIVATE tinyxml2
)
