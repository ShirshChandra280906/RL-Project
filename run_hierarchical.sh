#!/bin/bash
# =============================================================================
# Hierarchical DRL Navigation Training Script
# =============================================================================
# Based on: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"
#
# This script provides easy commands to train the hierarchical navigation system.
#
# Usage:
#   ./run_hierarchical.sh [command] [options]
#
# Commands:
#   train-full     - Run complete two-stage training (MA + SA)
#   train-ma       - Pre-train Motion Agent only
#   train-sa       - Train Subgoal Agent (requires pre-trained MA)
#   test           - Run tests to verify implementation
#   help           - Show this help message
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
DRL_PKG_DIR="$WORKSPACE_DIR/src/turtlebot3_drl/turtlebot3_drl"
HIERARCHICAL_DIR="$DRL_PKG_DIR/hierarchical"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "  Hierarchical DRL Navigation"
    echo "  Based on: Lightweight Motion Planning via Hierarchical RL"
    echo "============================================================"
    echo -e "${NC}"
}

print_help() {
    print_header
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  train-full     Run complete two-stage training (MA + SA)"
    echo "  train-ma       Pre-train Motion Agent only (Stage 1)"
    echo "  train-sa       Train Subgoal Agent with frozen MA (Stage 2)"
    echo "  test           Run tests to verify implementation"
    echo "  test-import    Quick import test"
    echo "  help           Show this help message"
    echo ""
    echo "Options for training:"
    echo "  --output-dir DIR    Output directory for models"
    echo "  --ma-model PATH     MA model path (for train-sa)"
    echo "  --ma-episodes N     Max episodes for MA training (default: 10000)"
    echo "  --sa-episodes N     Episodes for SA training (default: 5000)"
    echo "  --device DEVICE     cuda or cpu (default: auto)"
    echo ""
    echo "Examples:"
    echo "  $0 train-full"
    echo "  $0 train-ma --ma-episodes 5000"
    echo "  $0 train-sa --ma-model ./models/ma_converged.pth"
    echo ""
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}ERROR: python3 not found${NC}"
        exit 1
    fi
    
    # Check PyTorch
    if ! python3 -c "import torch" &> /dev/null; then
        echo -e "${RED}ERROR: PyTorch not installed${NC}"
        echo "Install with: pip install torch"
        exit 1
    fi
    
    # Check NumPy
    if ! python3 -c "import numpy" &> /dev/null; then
        echo -e "${RED}ERROR: NumPy not installed${NC}"
        echo "Install with: pip install numpy"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All dependencies satisfied${NC}"
}

run_import_test() {
    echo -e "${YELLOW}Running import test...${NC}"
    cd "$DRL_PKG_DIR"
    
    python3 << 'EOF'
import sys
sys.path.insert(0, '.')

print("Testing imports...")

# Test config
from hierarchical.config import HierarchicalConfig
config = HierarchicalConfig()
print(f"✓ Config loaded: SA={config.SA_TIME_STEP}s, MA={config.MA_TIME_STEP}s")

# Test planners
from hierarchical.planners import AStarPlanner, WaypointManager
print("✓ Planners imported")

# Test preprocessing
from hierarchical.preprocessing import LidarProcessor, LidarAttention
print("✓ Preprocessing imported")

# Test agents
from hierarchical.agents import SubgoalAgent, MotionAgent
print("✓ Agents imported")

# Test environments
from hierarchical.environments import HierarchicalEnvironment, SceneType
print("✓ Environments imported")

# Test training
from hierarchical.training import HierarchicalTrainer
print("✓ Training imported")

print("\n✓ All imports successful!")
EOF
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Import test passed${NC}"
    else
        echo -e "${RED}✗ Import test failed${NC}"
        exit 1
    fi
}

run_tests() {
    echo -e "${YELLOW}Running comprehensive tests...${NC}"
    cd "$DRL_PKG_DIR"
    
    # Run test files
    if [ -f "$HIERARCHICAL_DIR/tests/test_steps_5_to_6.py" ]; then
        echo "Running preprocessing tests..."
        python3 "$HIERARCHICAL_DIR/tests/test_steps_5_to_6.py"
    fi
    
    if [ -f "$HIERARCHICAL_DIR/tests/test_steps_7_to_8.py" ]; then
        echo "Running agent tests..."
        python3 "$HIERARCHICAL_DIR/tests/test_steps_7_to_8.py"
    fi
    
    echo -e "${GREEN}✓ All tests passed${NC}"
}

train_full() {
    print_header
    check_dependencies
    
    echo -e "${YELLOW}Starting full hierarchical training...${NC}"
    echo "Stage 1: Motion Agent pre-training (until convergence)"
    echo "Stage 2: Subgoal Agent training (with frozen MA)"
    echo ""
    
    cd "$DRL_PKG_DIR"
    python3 -m hierarchical.training.hierarchical_trainer --stage full "$@"
}

train_ma() {
    print_header
    check_dependencies
    
    echo -e "${YELLOW}Starting Motion Agent pre-training...${NC}"
    
    cd "$DRL_PKG_DIR"
    python3 -m hierarchical.training.hierarchical_trainer --stage ma "$@"
}

train_sa() {
    print_header
    check_dependencies
    
    echo -e "${YELLOW}Starting Subgoal Agent training...${NC}"
    
    cd "$DRL_PKG_DIR"
    python3 -m hierarchical.training.hierarchical_trainer --stage sa "$@"
}

# Main script
case "${1:-help}" in
    train-full)
        shift
        train_full "$@"
        ;;
    train-ma)
        shift
        train_ma "$@"
        ;;
    train-sa)
        shift
        train_sa "$@"
        ;;
    test)
        run_tests
        ;;
    test-import)
        run_import_test
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_help
        exit 1
        ;;
esac
