#!/usr/bin/env bash

set -euo pipefail

CUDA_VERSION_DEFAULT="12.2"     # change if you want a different default
INSTALL_STABLE="false"
INSTALL_TOOLKIT_ONLY="false"
SKIP_DRIVER="false"
NONINTERACTIVE="false"

usage() {
  cat <<'EOF'
Usage:
  sudo bash cuda_reset.sh [options]

Options:
  --install-stable           After removal, install a stable CUDA stack (default CUDA 12.2).
  --cuda-version X.Y         Override CUDA version (e.g. 12.6, 12.4, 12.2).
  --toolkit-only             Install CUDA toolkit only (no meta 'cuda' package).
  --skip-driver              Do not install NVIDIA driver (useful for headless/containers).
  --yes                      Non-interactive (assume "yes" to apt prompts).
  -h, --help                 Show this help.

Examples:
  sudo bash cuda_reset.sh --install-stable
  sudo bash cuda_reset.sh --install-stable --cuda-version 11.8 --skip-driver
  sudo bash cuda_reset.sh --install-stable --toolkit-only --yes
EOF
}

# parse args
CUDA_VERSION="$CUDA_VERSION_DEFAULT"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-stable) INSTALL_STABLE="true"; shift ;;
    --cuda-version) CUDA_VERSION="${2:-}"; shift 2 ;;
    --toolkit-only) INSTALL_TOOLKIT_ONLY="true"; shift ;;
    --skip-driver) SKIP_DRIVER="true"; shift ;;
    --yes) NONINTERACTIVE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

APT_YES=()
[[ "$NONINTERACTIVE" == "true" ]] && APT_YES=(-y)

echo ">>> Starting CUDA/NVIDIA cleanup..."
echo "    Non-interactive: $NONINTERACTIVE"
echo "    Will install stable after cleanup: $INSTALL_STABLE"
if [[ "$INSTALL_STABLE" == "true" ]]; then
  echo "    Target CUDA version: $CUDA_VERSION"
  echo "    Toolkit-only: $INSTALL_TOOLKIT_ONLY   Skip driver: $SKIP_DRIVER"
fi

### ---------- sanity checks ----------
if [[ $EUID -ne 0 ]]; then
  echo "This script needs sudo/root. Re-run with: sudo bash $0 ..." >&2
  exit 1
fi

if ! command -v apt &>/dev/null; then
  echo "This script currently targets apt-based distros (Ubuntu/Debian)." >&2
  exit 1
fi

source /etc/os-release || true
UBU_VER="${VERSION_ID:-}"
if [[ -z "${UBU_VER}" ]]; then
  echo "Could not detect Ubuntu/Debian version from /etc/os-release. Proceeding anyway..."
fi
UBU_TAG="ubuntu${UBU_VER//./}" # e.g., 22.04 -> ubuntu2204, 20.04 -> ubuntu2004

### ---------- removal phase ----------
echo ">>> Purging CUDA & NVIDIA packages..."
apt "${APT_YES[@]}" --purge remove 'cuda*' 'nvidia*' 'libnvidia*' 'nsight-*' || true

echo ">>> Autoremove + clean..."
apt "${APT_YES[@]}" autoremove
apt clean

echo ">>> Checking for leftover dpkg entries..."
dpkg -l | grep -E 'cuda|nvidia' || echo "No leftover packages found."

echo ">>> Removing common directories (safe if absent)..."
rm -rf /usr/local/cuda* \
       /etc/ld.so.conf.d/cuda.conf \
       /etc/apt/preferences.d/cuda-repository-pin-600 \
       /var/lib/cuda* \
       /var/cuda* \
       /opt/nvidia \
       /opt/cuda \
       /usr/share/nvidia || true

echo ">>> Running cuda-uninstaller if present..."
if [[ -x /usr/local/cuda/bin/cuda-uninstaller ]]; then
  /usr/local/cuda/bin/cuda-uninstaller || true
else
  # try any versioned cuda dir
  CAND=(/usr/local/cuda-*/bin/cuda-uninstaller)
  for u in "${CAND[@]}"; do
    [[ -x "$u" ]] && "$u" || true
  done
fi

echo ">>> Scrubbing user-level caches (won't harm if missing)..."
rm -rf /root/.nv ~/.nv || true

echo ">>> Removing CUDA exports from shell rc files..."
for RC in ~/.bashrc ~/.zshrc /root/.bashrc /root/.zshrc; do
  [[ -f "$RC" ]] || continue
  sed -i '/\/usr\/local\/cuda/d;/LD_LIBRARY_PATH.*cuda/d;/CUDA_HOME/d' "$RC" || true
done

echo ">>> Verifying nvidia-smi absence..."
if command -v nvidia-smi &>/dev/null; then
  echo "Warning: nvidia-smi still exists at $(command -v nvidia-smi). That can be fine if the kernel module is still loaded."
fi

### ---------- optional install phase ----------
if [[ "$INSTALL_STABLE" != "true" ]]; then
  echo ">>> Removal complete. Skipping installation (no --install-stable)."
  exit 0
fi

echo ">>> Installing stable NVIDIA stack..."

# 1) Driver (unless skipped)
if [[ "$SKIP_DRIVER" != "true" ]]; then
  echo ">>> Installing recommended NVIDIA driver via ubuntu-drivers..."
  apt "${APT_YES[@]}" update
  if command -v ubuntu-drivers &>/dev/null; then
    ubuntu-drivers autoinstall || true
  else
    echo "ubuntu-drivers not found; installing ubuntu-drivers-common..."
    apt "${APT_YES[@]}" install ubuntu-drivers-common
    ubuntu-drivers autoinstall || true
  fi
else
  echo ">>> Skipping driver installation as requested."
fi

# 2) CUDA repository keyring
echo ">>> Setting up NVIDIA CUDA apt repo for ${UBU_TAG:-your Ubuntu}..."
# Construct the repo URL safely. Fallback to a broadly used keyring if tag unknown.
KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
REPO_URL_BASE="https://developer.download.nvidia.com/compute/cuda/repos"
if [[ -n "$UBU_TAG" ]]; then
  KEY_URL="${REPO_URL_BASE}/${UBU_TAG}/x86_64/${KEYRING_DEB}"
else
  # fallback to ubuntu2204; user can adjust later if needed
  KEY_URL="${REPO_URL_BASE}/ubuntu2204/x86_64/${KEYRING_DEB}"
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
set +e
curl -fsSL "$KEY_URL" -o "$TMPDIR/$KEYRING_DEB"
CURL_RC=$?
set -e
if [[ $CURL_RC -ne 0 ]]; then
  echo "Could not fetch ${KEY_URL}. Trying ubuntu2204 as a fallback..."
  curl -fsSL "${REPO_URL_BASE}/ubuntu2204/x86_64/${KEYRING_DEB}" -o "$TMPDIR/$KEYRING_DEB"
fi

dpkg -i "$TMPDIR/$KEYRING_DEB" || true
apt "${APT_YES[@]}" update

# 3) CUDA toolkit meta
CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUDA_MINOR="${CUDA_VERSION#*.}"
CUDA_MINOR="${CUDA_MINOR%%.*}"  # just in case of x.y.z input
CUDA_META="cuda-${CUDA_MAJOR}-${CUDA_MINOR}"           # e.g., cuda-12-2 (full stack)
CUDA_TOOLKIT="cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"# e.g., cuda-toolkit-12-2

echo ">>> Installing CUDA ${CUDA_VERSION}..."
if [[ "$INSTALL_TOOLKIT_ONLY" == "true" ]]; then
  # toolkit-only
  if apt-cache policy "${CUDA_TOOLKIT}" | grep -q Candidate; then
    apt "${APT_YES[@]}" install "${CUDA_TOOLKIT}"
  else
    echo "Package ${CUDA_TOOLKIT} not found in repo. Falling back to 'cuda-toolkit'."
    apt "${APT_YES[@]}" install cuda-toolkit
  fi
else
  # full meta (includes driver; harmless if driver already present/pinned)
  if apt-cache policy "${CUDA_META}" | grep -q Candidate; then
    apt "${APT_YES[@]}" install "${CUDA_META}"
  else
    echo "Package ${CUDA_META} not found. Falling back to 'cuda'."
    apt "${APT_YES[@]}" install cuda
  fi
fi

# 4) Post-install environment
echo ">>> Wiring up PATH and LD_LIBRARY_PATH in ~/.bashrc..."
USER_RC="${SUDO_USER:+/home/$SUDO_USER}/.bashrc"
[[ -n "$SUDO_USER" && -f "$USER_RC" ]] || USER_RC="${HOME}/.bashrc"

# determine /usr/local/cuda symlink (usually created by meta)
if [[ -d /usr/local/cuda ]]; then
  CUDA_PREFIX="/usr/local/cuda"
else
  # try versioned path
  if [[ -d "/usr/local/cuda-${CUDA_VERSION}" ]]; then
    CUDA_PREFIX="/usr/local/cuda-${CUDA_VERSION}"
  else
    # best-effort generic
    CUDA_PREFIX="/usr/local/cuda"
  fi
fi

ADD_LINES=$(cat <<EOF
# CUDA ${CUDA_VERSION} setup
export PATH=${CUDA_PREFIX}/bin:\$PATH
export LD_LIBRARY_PATH=${CUDA_PREFIX}/lib64:\$LD_LIBRARY_PATH
EOF
)

# append only if not already present
if ! grep -q "${CUDA_PREFIX}/bin" "$USER_RC" 2>/dev/null; then
  echo "${ADD_LINES}" >> "$USER_RC"
fi

echo ">>> Done. Open a new shell or run: source \"$USER_RC\""
echo ">>> Quick sanity checks you can run:"
echo "    nvcc --version    # should show ${CUDA_VERSION} (or close)"
echo "    nvidia-smi        # should show driver + CUDA runtime"

# Friendly reminder about reboot
if [[ "$SKIP_DRIVER" != "true" ]]; then
  echo ">>> A reboot is recommended to load the fresh NVIDIA kernel modules."
fi
