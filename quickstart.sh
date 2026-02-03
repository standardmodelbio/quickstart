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

# Colors and symbols for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
CYAN='\033[0;36m'

# Function to print status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Dots spinner function (UV-style)
show_spinner() {
    local pid=$1
    local message=$2
    local spin='⣾⣽⣻⢿⡿⣟⣯⣷'
    local i=0
    local spinlen=8
    
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) % $spinlen ))
        printf "\r${BLUE}[${spin:$i:1}]${NC} $message"
        sleep 0.08
    done
    wait $pid 2>/dev/null  # Wait silently
    printf "\r${GREEN}[✓]${NC} $message\n"
}


echo ""
echo "This quickstart will:"
echo "  • Create a Python 3.10 virtual environment named 'standard_model'"
echo "  • Install PyTorch with CUDA support"
echo "  • Install HuggingFace libraries (transformers, datasets, accelerate)"
echo -e "  ${GREEN}•${NC} \033[1mInstall the Standard Model huggingface family to local\033[0m"
echo ""

# Check if user already has dependencies in their current environment
print_status "Checking current environment for existing dependencies..."
HAS_TORCH=false
HAS_TRANSFORMERS=false
HAS_DATASETS=false
HAS_ACCELERATE=false
HAS_SMB_UTILS=false

python3 -c "import torch" 2>/dev/null && HAS_TORCH=true
python3 -c "import transformers" 2>/dev/null && HAS_TRANSFORMERS=true
python3 -c "import datasets" 2>/dev/null && HAS_DATASETS=true
python3 -c "import accelerate" 2>/dev/null && HAS_ACCELERATE=true
python3 -c "import smb_biopan_utils" 2>/dev/null && HAS_ACCELERATE=true

print_success "Environment check complete"

if $HAS_TORCH && $HAS_TRANSFORMERS && $HAS_DATASETS && $HAS_ACCELERATE && $HAS_SMB_UTILS; then
    print_success "All dependencies found in current environment."
    echo ""
    echo -n -e "Do you still want to create a new 'standard_model' environment \033[1m[RECOMMENDED]\033[0m? (y/N): "
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_success "Setup skipped. Using current environment."
        exit 0
    fi
else
    echo ""
    print_status "Missing dependencies in current environment:"
    $HAS_TORCH || echo -e "  ${RED}✗${NC} torch"
    $HAS_TRANSFORMERS || echo -e "  ${RED}✗${NC} transformers"
    $HAS_DATASETS || echo -e "  ${RED}✗${NC} datasets"
    $HAS_ACCELERATE || echo -e "  ${RED}✗${NC} accelerate"
    $HAS_SMB_UTILS || echo -e "  ${RED}✗${NC} smb_biopan_utils"
    echo ""
fi

# Check if standard_model already exists
SKIP_ENV_CREATION=false

if [ -d "standard_model" ]; then
    echo -e "${YELLOW}[⚠]${NC} Virtual environment 'standard_model' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        rm -rf standard_model
    else
        print_status "Using existing environment. Checking dependencies..."
        SKIP_ENV_CREATION=true
        
        # Check if key packages are installed in standard_model
        if standard_model/bin/python -c "import torch, transformers, datasets, accelerate, smb_biopan_utils" 2>/dev/null; then
            print_success "All dependencies already installed in 'standard_model'!"
            echo ""
            print_success "Setup complete! Activate with: source standard_model/bin/activate"
            exit 0
        else
            print_warning "Some dependencies missing in 'standard_model'. Installing..."
        fi
    fi
fi

echo ""
echo -e "${CYAN}==================================${NC}"
echo -e "${CYAN}        ENVIRONMENT SETUP${NC}"
echo -e "${CYAN}==================================${NC}"
echo ""

# Only create environment if needed
if [ "$SKIP_ENV_CREATION" = false ]; then
    ( pip install --upgrade pip > /dev/null 2>&1 ) &
    show_spinner $! "Upgrading pip..."

    ( pip install uv > /dev/null 2>&1 ) &
    show_spinner $! "Installing uv package manager..."

    print_status "Creating virtual environment 'standard_model' (Python 3.10)..."
    uv venv standard_model --python 3.10
else
    print_status "Skipping environment creation, using existing 'standard_model'"
fi

echo ""
echo -e "${CYAN}==================================${NC}"
echo -e "${CYAN}     DEPENDENCY INSTALLATION${NC}"
echo -e "${CYAN}==================================${NC}"
echo ""

# Detect platform and CUDA availability
echo -e "\033[1mAll dependencies will be installed in "standard_model" venv\033[0m"
print_status "Detecting platform and GPU capabilities..."

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    print_success "CUDA detected: Version $CUDA_VERSION"
    
    # Install PyTorch with CUDA support using uv
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        ( uv pip install --python standard_model torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 > /dev/null 2>&1 ) &
        show_spinner $! "Installing PyTorch with CUDA 12.x support..."
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        ( uv pip install --python standard_model torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1 ) &
        show_spinner $! "Installing PyTorch with CUDA 11.x support..."
    fi
else
    print_warning "No CUDA detected, installing CPU-only version"
    ( uv pip install --python standard_model torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1 ) &
    show_spinner $! "Installing PyTorch (CPU)..."
fi

# Install HuggingFace with compatible versions
# Pin transformers to 4.46.3 to avoid tie_weights incompatibility with custom models
( uv pip install --python standard_model transformers==4.46.3 > /dev/null 2>&1 ) &
show_spinner $! "Installing HuggingFace transformers..."

( uv pip install --python standard_model datasets > /dev/null 2>&1 ) &
show_spinner $! "Installing HuggingFace datasets..."

( uv pip install --python standard_model accelerate > /dev/null 2>&1 ) &
show_spinner $! "Installing HuggingFace accelerate..."

( uv pip install --python standard_model git+https://github.com/standardmodelbio/smb-biopan-utils.git > /dev/null 2>&1 ) &
show_spinner $! "Installing smb-biopan-utils..."

( uv pip install --python standard_model lifelines > /dev/null 2>&1 ) &
show_spinner $! "Installing lifelines (survival analysis)..."

uv pip install --python standard_model python-dotenv > /dev/null 2>&1

# Verify installations
echo ""
print_status "Verifying installations..."
standard_model/bin/python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
standard_model/bin/python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
print_success "Verification complete"

echo ""
echo -e "${CYAN}==================================${NC}"
echo -e "${CYAN}  STANDARD MODEL HF INSTALLATION${NC}"
echo -e "${CYAN}==================================${NC}"
echo ""

# Install huggingface_hub if not already installed
( uv pip install --python standard_model huggingface_hub > /dev/null 2>&1 ) &
show_spinner $! "Installing huggingface_hub..."

# Download the model
MODEL="standardmodelbio/SMB-v1-1.7B-Structure"
MODEL_NAME=$(basename "$MODEL")

print_status "Checking model accessibility..."

# Check if model is accessible
MODEL_CHECK=$(standard_model/bin/python << PYEOF
from huggingface_hub import model_info
try:
    info = model_info("$MODEL")
    print("accessible")
except Exception as e:
    if "gated" in str(e).lower() or "private" in str(e).lower():
        print("private")
    else:
        print("error")
PYEOF
)

if [ "$MODEL_CHECK" == "private" ]; then
    print_error "Model '$MODEL' is private or gated."
    print_warning "Please authenticate with: huggingface-cli login"
    print_warning "Or request access at: https://huggingface.co/$MODEL"
    exit 1
elif [ "$MODEL_CHECK" == "error" ]; then
    print_error "Failed to access model '$MODEL'. It may not exist."
    exit 1
fi

print_success "Model is accessible"

print_status "Downloading $MODEL_NAME..."
( standard_model/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" > /dev/null 2>&1 ) &
show_spinner $! "Downloading $MODEL_NAME..."
print_success "Model downloaded to HuggingFace cache (~/.cache/huggingface/)"

echo ""
print_success "Setup complete!"
echo ""
echo -e "To get started, activate your environment:"
echo -e "  ${CYAN}source standard_model/bin/activate${NC}"
echo ""
