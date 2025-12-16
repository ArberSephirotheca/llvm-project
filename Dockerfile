# Stage 0: common base with build deps
FROM ubuntu:22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git python3 pkg-config \
      libedit-dev libxml2-dev zlib1g-dev libncurses5-dev \
      curl ca-certificates \
      && rm -rf /var/lib/apt/lists/*

# Stage 1: build LLVM/MLIR from source
ARG LLVM_VERSION=21.1.0
FROM base AS llvm-build
WORKDIR /opt
RUN curl -L "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz" \
      -o llvm-src.tar.xz \
 && tar -xf llvm-src.tar.xz
WORKDIR /opt/llvm-project-${LLVM_VERSION}.src
RUN cmake -G Ninja -S llvm -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DLLVM_ENABLE_ASSERTIONS=OFF
ARG LLVM_JOBS=4
RUN cmake --build build -- -j${LLVM_JOBS} -l${LLVM_JOBS}
RUN cmake --install build --prefix /opt/llvm-install

# Stage 2: build SIMT-Step against the installed LLVM
FROM base AS simt-step
ENV LLVM_PREFIX=/opt/llvm-install
ENV PATH=/opt/llvm-install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/llvm-install/lib:/opt/llvm-install/lib64
COPY --from=llvm-build /opt/llvm-install /opt/llvm-install

WORKDIR /opt/simt-step
COPY . .

RUN cmake -G Ninja -S . -B build \
      -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
      -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir \
      -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build

CMD ["bash"]
