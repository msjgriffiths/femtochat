#!/usr/bin/env bash
# Build the two native adapters for the pinned Reactant environment (Linux).
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$root/../.." && pwd)
build="$root/build"
lib=${1:-}
if [ -z "$lib" ]; then
    lib=$("${JULIA:-julia}" --startup-file=no --project="$repo/environments/reactant" -e \
        'using Reactant; print(joinpath(Reactant.Reactant_jll.artifact_dir,"lib","libReactantExtra.so"))')
fi
lib=$(realpath -e "$lib")
mkdir -p "$build"
cd "$build"

llvm=a6b0af7536ef0ae8383ec729c5fbf23c73243776
enzyme=3e7cc67727eadcdb6cc19bc24292a6fe7aed100c
xla=9862401b5d9444686cd1bc9283991f6b834b58d3

# Revision-keyed caches; a failed extraction is retried on the next invocation.
source_tree() {
    local repository=$1 revision=$2 checksum=${3:-}
    local directory="${repository##*/}-$revision"
    if [ ! -f "$directory/.ready" ]; then
        curl -fL --retry 3 "https://github.com/$repository/archive/$revision.tar.gz" -o "$directory.tar.gz"
        if [ -n "$checksum" ]; then
            printf '%s  %s\n' "$checksum" "$directory.tar.gz" | sha256sum -c
        fi
        mkdir -p "$directory"
        tar -xzf "$directory.tar.gz" --strip-components=1 -C "$directory"
        touch "$directory/.ready"
    fi
}
source_tree llvm/llvm-project "$llvm" dfd250ff52da324d2e93dcded5e00676653e319982fff4f3789c1df3db88316c
source_tree EnzymeAD/Enzyme "$enzyme"
llvm_source="llvm-project-$llvm"
llvm_build="llvm-build-$llvm"
enzyme_source="Enzyme-$enzyme/enzyme/Enzyme"

# Generate matching headers/tablegen, not the LLVM or Reactant libraries.
cmake -G Ninja -S "$llvm_source/llvm" -B "$llvm_build" \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD='' \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DMLIR_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF
cmake --build "$llvm_build" --target mlir-headers llvm-headers -j "${JOBS:-4}"

mkdir -p generated/MLIR/Interfaces generated/MLIR/Dialect
includes=(-I"$llvm_source/mlir/include" -I"$llvm_build/tools/mlir/include"
          -I"$llvm_source/llvm/include" -I"$llvm_build/include"
          -I"$enzyme_source" -I"$enzyme_source/MLIR" -I"$enzyme_source/MLIR/Dialect")
td="$llvm_build/bin/mlir-tblgen"
"$td" "${includes[@]}" -gen-op-interface-decls \
    "$enzyme_source/MLIR/Interfaces/AutoDiffOpInterface.td" -o generated/MLIR/Interfaces/AutoDiffOpInterface.h.inc
"$td" "${includes[@]}" -gen-type-interface-decls \
    "$enzyme_source/MLIR/Interfaces/AutoDiffTypeInterface.td" -o generated/MLIR/Interfaces/AutoDiffTypeInterface.h.inc
"$td" "${includes[@]}" -gen-enum-decls \
    "$enzyme_source/MLIR/Dialect/EnzymeEnums.td" -o generated/MLIR/Dialect/EnzymeEnums.h.inc
for pair in 'op-decls EnzymeOps' 'typedef-decls EnzymeOpsTypes' \
            'attr-interface-decls EnzymeAttributeInterfaces' 'attrdef-decls EnzymeAttributes'; do
    read -r generator output <<< "$pair"
    "$td" "${includes[@]}" "-gen-$generator" --attrdefs-dialect=enzyme --typedefs-dialect=enzyme \
        "$enzyme_source/MLIR/Dialect/EnzymeOps.td" -o "generated/MLIR/Dialect/$output.h.inc"
done

headers="xla-$xla"
mkdir -p "$headers/xla/ffi/api"
for header in c_api.h api.h ffi.h; do
    file="$headers/xla/ffi/api/$header"
    if [ ! -f "$file" ]; then
        curl -fL --retry 3 "https://raw.githubusercontent.com/openxla/xla/$xla/xla/ffi/api/$header" -o "$file.part"
        mv "$file.part" "$file"
    fi
done

"${CXX:-g++}" -std=c++17 -O2 -shared -fPIC -I"$headers" \
    "$root/xla_ffi.cpp" "$lib" -Wl,-rpath,"$(dirname "$lib")" -o libreactant_xla_ffi.so
"${CXX:-g++}" -std=c++17 -O2 -shared -fPIC -DNDEBUG "${includes[@]}" \
    -Igenerated -Igenerated/MLIR "$root/attention_rule.cpp" "$lib" \
    -Wl,-rpath,"$(dirname "$lib")" -o libreactant_attention_rule.so
# Keep existing adapters in place if either compilation fails.
mv libreactant_xla_ffi.so libreactant_attention_rule.so "$root/"
printf 'Reactant adapters built in %s\n' "$root"
