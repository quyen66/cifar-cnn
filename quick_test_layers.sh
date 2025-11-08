#!/bin/bash
# =================================================================
# Quick Parameter Tuning Test Script
# =================================================================
# Chạy NHANH hơn test_all_attacks.sh để test param tuning
# Chỉ test 1 attack scenario với nhiều param combinations
# =================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =================================================================
# CONFIGURATION
# =================================================================
NUM_CLIENTS=40
NUM_ROUNDS=30  # Giảm từ 50 → 30 để nhanh hơn
SAVE_DIR="results/quick_param_test"

# Attack scenario (FIXED)
ATTACK_TYPE="byzantine"
ATTACK_RATIO=0.3
BYZANTINE_TYPE="sign_flip"
BYZANTINE_SCALE=2.0

# Tạo thư mục
mkdir -p "$SAVE_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  QUICK PARAMETER TUNING TEST"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  • Clients: $NUM_CLIENTS"
echo "  • Rounds: $NUM_ROUNDS (faster than full 50)"
echo "  • Attack: $ATTACK_TYPE @ $ATTACK_RATIO ($BYZANTINE_TYPE, scale=$BYZANTINE_SCALE)"
echo "  • Results: $SAVE_DIR"
echo ""
echo "Purpose: Quick validation of parameter configurations"
echo "Note: Use automated scripts for comprehensive tuning"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# =================================================================
# TEST FUNCTIONS
# =================================================================

run_test() {
    local TEST_NUM=$1
    local TEST_NAME=$2
    local EXTRA_PARAMS=$3
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🧪 TEST $TEST_NUM: $TEST_NAME${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Base config
    BASE_CONFIG="num-clients=$NUM_CLIENTS num-server-rounds=$NUM_ROUNDS partition-type=\"iid\" attack-type=\"$ATTACK_TYPE\" attack-ratio=$ATTACK_RATIO byzantine-type=\"$BYZANTINE_TYPE\" byzantine-scale=$BYZANTINE_SCALE enable-defense=true"
    
    # Add test-specific params
    RUN_CONFIG="$BASE_CONFIG $EXTRA_PARAMS save-dir=\"$SAVE_DIR/test_$TEST_NUM\" save-interval=10"
    
    # Run
    flwr run . --run-config "$RUN_CONFIG"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
    else
        echo -e "${RED}✗ Test $TEST_NUM failed${NC}"
    fi
    
    echo ""
    sleep 2
}

# =================================================================
# BASELINE TEST
# =================================================================

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  BASELINE (DEFAULT PARAMETERS)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

run_test 1 "Baseline - Default Params" ""

# =================================================================
# LAYER 1 VARIATIONS
# =================================================================

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  LAYER 1 PARAMETER VARIATIONS${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 2: Stricter DBSCAN
run_test 2 "Layer1 - Stricter DBSCAN (minPts=4, eps=0.4)" \
    "dbscan_min_samples=4 dbscan_eps_multiplier=0.4"

# Test 3: Looser magnitude filter
run_test 3 "Layer1 - Looser MAD (k=5.0)" \
    "mad_k_normal=5.0 mad_k_warmup=7.0"

# Test 4: Stricter voting
run_test 4 "Layer1 - Stricter Voting (threshold=3)" \
    "voting_threshold_normal=3 voting_threshold_warmup=4"

# =================================================================
# LAYER 2 VARIATIONS
# =================================================================

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  LAYER 2 PARAMETER VARIATIONS${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 5: Stricter distance
run_test 5 "Layer2 - Stricter Distance (mult=1.2)" \
    "distance_multiplier=1.2"

# Test 6: Looser cosine
run_test 6 "Layer2 - Looser Cosine (threshold=0.4)" \
    "cosine_threshold=0.4"

# Test 7: Longer warmup
run_test 7 "Layer2 - Longer Warmup (20 rounds)" \
    "layer2_warmup_rounds=20"

# =================================================================
# REPUTATION VARIATIONS
# =================================================================

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  REPUTATION SYSTEM VARIATIONS${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 8: Faster adaptation
run_test 8 "Reputation - Faster Adaptation (alpha=0.5/0.3)" \
    "ema_alpha_increase=0.5 ema_alpha_decrease=0.3"

# Test 9: Harsher penalties
run_test 9 "Reputation - Harsher Penalties (0.3)" \
    "penalty_flagged=0.3 penalty_variance=0.15"

# Test 10: Higher floor lift
run_test 10 "Reputation - Higher Floor Lift (0.5)" \
    "floor_lift_threshold=0.5"

# =================================================================
# SUMMARY
# =================================================================

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ ALL QUICK TESTS COMPLETED!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "📁 Results saved to: $SAVE_DIR"
echo ""
echo "Test Summary:"
echo "  • 1 Baseline test"
echo "  • 3 Layer 1 variations"
echo "  • 3 Layer 2 variations"
echo "  • 3 Reputation variations"
echo "  • Total: 10 tests"
echo ""
echo "Next steps:"
echo ""
echo "1. Quick analysis:"
echo "   python check_attack_results.py $SAVE_DIR"
echo ""
echo "2. For comprehensive tuning, use automated scripts:"
echo "   python param_generator.py           # Generate param combinations"
echo "   python run_param_tests.py           # Run all tests"
echo "   python analyze_param_results.py     # Analyze results"
echo ""
echo "3. Compare with baseline:"
echo "   Compare test_1 (baseline) with others"
echo ""
echo -e "${BLUE}🧹 To clean up:${NC}"
echo -e "${BLUE}  rm -rf $SAVE_DIR${NC}"
echo ""