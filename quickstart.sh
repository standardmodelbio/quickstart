#!/bin/bash
set +m

cat <<'EOF'

             █▀╫╫█⌐
          ,▄Φ█▓╫╫█─▄µ
    ▄ΦΦ▄▄▀╙    ╙└   ╙▀Φ▄▓Φ▄⌂
   ▓▌░░╠█             ▓▌░░░█
    ▀▓▓▀╙¬4▄µ ,╓⌂ ,,Φ▀╙▀▓▓▀└   ▗▄▄▖▗▄▄▄▖▗▄▖ ▗▖  ▗▖▗▄▄▄  ▗▄▖ ▗▄▄▄▖ ▗▄▄▄    ▗▖  ▗▖ ▗▄▖ ▗▄▄▄ ▗▄▄▄▖▗▖   
     ▐▌     ▐█▀╫╫█▌     ▐▌     ▐▌    █ ▐▌ ▐▌▐▛▚▖▐▌▐▌  █▐▌ ▐▌▐▌ ▐▌▐▌  █    ▐▛▚▞▜▌▐▌ ▐▌▐▌  █▐▌   ▐▌   
     ▐▌      █▓╫▄█¬     ▐▌     ▝▀▚▖  █ ▐▛▀▜▌▐▌ ▝▜▌▐▌  █▐▛▀▜▌▐▛▀▚▖▐▌  █    ▐▌  ▐▌▐▌ ▐▌▐▌  █▐▛▀▀▘▐▌   
    ▄▓█▄¿      █─      ▄ΦΦ▄⌂   ▗▄▄▞▘ █ ▐▌ ▐▌▐▌  ▐▌▐▙▄▄▀▐▌ ▐▌▐▌ ▐▌▐▙▄▄▀    ▐▌  ▐▌▝▚▄▞▘▐▙▄▄▀▐▙▄▄▖▐▙▄▄▖
   ╢▌╫╫╫█      ▄      ▓▌░░╠█  
    ▀▀▀▀,╙▀▄p ;█p ╓▄Φ▀└▀▀▓▀└
            ▐█▀h╠█▌
             █▄▄▄█'
               └`
EOF

# ── Colors ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
CYAN='\033[0;36m'
BOLD='\033[1m'

print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
print_error()   { echo -e "${RED}[✗]${NC} $1"; }

show_spinner() {
    local pid=$1
    local message=$2
    local spin='⣾⣽⣻⢿⡿⣟⣯⣷'
    local i=0
    local spinlen=8
    while kill -0 "$pid" 2>/dev/null; do
        i=$(( (i+1) % spinlen ))
        printf "\r${BLUE}[${spin:$i:1}]${NC} %s" "$message"
        sleep 0.08
    done
    wait "$pid" 2>/dev/null
    printf "\r${GREEN}[✓]${NC} %s\n" "$message"
}

REPO_URL="https://github.com/standardmodelbio/quickstart.git"
REPO_DIR="quickstart"

# ── Overview ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Lightspeed Quickstart${NC}"
echo ""
echo "This will:"
echo "  • Clone the Standard Model quickstart repo"
echo "  • Install all dependencies via uv (locked & reproducible)"
echo "  • Get you ready to run the demo or use your own data"
echo ""

# ── Prerequisites ───────────────────────────────────────────────────
print_status "Checking prerequisites..."

if ! command -v git &>/dev/null; then
    print_error "git is required but not installed."
    echo -e "  Install it: ${CYAN}https://git-scm.com/downloads${NC}"
    exit 1
fi
print_success "git found"

# Install uv if not present
if ! command -v uv &>/dev/null; then
    print_status "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        print_error "Failed to install uv. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi
print_success "uv found"

# ── Clone ───────────────────────────────────────────────────────────
echo ""
if [ -d "$REPO_DIR" ]; then
    print_warning "'$REPO_DIR' directory already exists."
    echo -n -e "  Overwrite and re-clone? (y/N): "
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$REPO_DIR"
    else
        print_status "Using existing '$REPO_DIR' directory."
    fi
fi

if [ ! -d "$REPO_DIR" ]; then
    ( git clone --depth 1 "$REPO_URL" "$REPO_DIR" > /dev/null 2>&1 ) &
    show_spinner $! "Cloning quickstart repo..."
fi

cd "$REPO_DIR" || { print_error "Failed to enter $REPO_DIR"; exit 1; }

# ── Install dependencies ───────────────────────────────────────────
echo ""
echo -e "${CYAN}══════════════════════════════════════${NC}"
echo -e "${CYAN}       INSTALLING DEPENDENCIES${NC}"
echo -e "${CYAN}══════════════════════════════════════${NC}"
echo ""

( uv sync 2>&1 ) &
show_spinner $! "Installing locked dependencies (this may take a few minutes)..."

# ── Verify ──────────────────────────────────────────────────────────
echo ""
print_status "Verifying installation..."

uv run python -c "
import torch, transformers, smb_utils
print(f'  PyTorch:      {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_success "All dependencies verified"
else
    print_error "Verification failed. Try running 'uv sync' manually in the '$REPO_DIR' directory."
    exit 1
fi

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}══════════════════════════════════════${NC}"
echo -e "${CYAN}            READY TO GO${NC}"
echo -e "${CYAN}══════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Setup complete!${NC} You're in the ${BOLD}quickstart/${NC} directory."
echo ""
echo -e "${BOLD}Run the demo with synthetic data:${NC}"
echo -e "  ${CYAN}cd quickstart${NC}"
echo -e "  ${CYAN}uv run python demo.py${NC}"
echo ""
echo -e "${BOLD}Learn more:${NC}"
echo -e "  Synthetic data example:  ${CYAN}https://docs.standardmodel.bio/example${NC}"
echo -e "  Use your own data:       ${CYAN}https://docs.standardmodel.bio/your-own-data${NC}"
echo ""
