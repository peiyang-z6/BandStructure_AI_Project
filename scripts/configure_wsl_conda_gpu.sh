#!/usr/bin/env bash
# Configure TensorFlow CUDA library discovery for one Conda environment.
# Usage: bash scripts/configure_wsl_conda_gpu.sh "$CONDA_PREFIX"
set -euo pipefail

target_prefix="${1:-${CONDA_PREFIX:-}}"
if [[ -z "$target_prefix" || ! -x "$target_prefix/bin/python" ]]; then
  echo "Usage: $0 /absolute/path/to/conda/env" >&2
  exit 2
fi

python_version="$($target_prefix/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
nvidia_root="$target_prefix/lib/python${python_version}/site-packages/nvidia"
if [[ ! -d "$nvidia_root" ]]; then
  echo "NVIDIA pip libraries not found under: $nvidia_root" >&2
  exit 3
fi

activate_dir="$target_prefix/etc/conda/activate.d"
deactivate_dir="$target_prefix/etc/conda/deactivate.d"
mkdir -p "$activate_dir" "$deactivate_dir"

cat > "$activate_dir/bandstructure_cuda.sh" <<EOF
export BANDSTRUCTURE_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="$nvidia_root/cuda_runtime/lib:$nvidia_root/cuda_nvrtc/lib:$nvidia_root/cublas/lib:$nvidia_root/cudnn/lib:$nvidia_root/cufft/lib:$nvidia_root/curand/lib:$nvidia_root/cusolver/lib:$nvidia_root/cusparse/lib:$nvidia_root/nvjitlink/lib:/usr/lib/wsl/lib:\${LD_LIBRARY_PATH-}"
EOF

cat > "$deactivate_dir/bandstructure_cuda.sh" <<'EOF'
export LD_LIBRARY_PATH="${BANDSTRUCTURE_OLD_LD_LIBRARY_PATH-}"
unset BANDSTRUCTURE_OLD_LD_LIBRARY_PATH
EOF

echo "Configured TensorFlow CUDA library paths for: $target_prefix"
