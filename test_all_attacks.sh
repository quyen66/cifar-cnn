#!/bin/bash
# =================================================================
# COMPREHENSIVE ATTACK TEST SUITE
# =================================================================
# Test tất cả các loại attack với nhiều cấu hình khác nhau
# Bao gồm: Label Flip, Byzantine, Gaussian với các mức độ khác nhau
# =================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =================================================================
# CONFIGURATION
# =================================================================
NUM_CLIENTS=40
NUM_ROUNDS=50
SAVE_DIR="results/comprehensive_attacks"

# Tạo thư mục kết quả
mkdir -p "$SAVE_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  COMPREHENSIVE FEDERATED LEARNING ATTACK TEST SUITE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  • Total clients: $NUM_CLIENTS"
echo "  • Server rounds: $NUM_ROUNDS"
echo "  • Results directory: $SAVE_DIR"
echo ""
echo "Attack types to test:"
echo "  • Label Flip (3 variants)"
echo "  • Byzantine (3 types × 3 scales)"
echo "  • Gaussian Noise (3 levels)"
echo "  • Each attack: 30% malicious clients"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Counter for tests
TEST_NUM=0
TOTAL_TESTS=26  # 1 baseline + 25 attack variants

# =================================================================
# BASELINE: NO ATTACK
# =================================================================
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: BASELINE - No Attack${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="none" attack-ratio=0.0 enable-defense=false save-dir="'$SAVE_DIR'/baseline_clean" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
echo ""
sleep 2

# =================================================================
# LABEL FLIP ATTACKS
# =================================================================
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  LABEL FLIP ATTACKS${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 2: Label Flip - Reverse (100% flip)
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Label Flip - Reverse (30%, 100% flip)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="label_flip" attack-ratio=0.3 flip-probability=1.0 flip-type="reverse" enable-defense=false save-dir="'$SAVE_DIR'/labelflip_reverse_nodefense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 3: Label Flip - Reverse (100% flip) WITH DEFENSE
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Label Flip - Reverse WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="label_flip" attack-ratio=0.3 flip-probability=1.0 flip-type="reverse" enable-defense=true save-dir="'$SAVE_DIR'/labelflip_reverse_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 4: Label Flip - Reverse (50% flip)
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Label Flip - Reverse (30%, 50% flip)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="label_flip" attack-ratio=0.3 flip-probability=0.5 flip-type="reverse" enable-defense=true save-dir="'$SAVE_DIR'/labelflip_reverse_50pct_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 5: Label Flip - Random
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Label Flip - Random (30%, 100% flip)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="label_flip" attack-ratio=0.3 flip-probability=1.0 flip-type="random" enable-defense=true save-dir="'$SAVE_DIR'/labelflip_random_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# =================================================================
# BYZANTINE ATTACKS - SIGN FLIP
# =================================================================
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  BYZANTINE ATTACKS - SIGN FLIP${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 6: Byzantine Sign Flip - Scale 1.0
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Sign Flip (Scale 1.0) - NO DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="sign_flip" byzantine-scale=1.0 enable-defense=false save-dir="'$SAVE_DIR'/byzantine_signflip_scale1_nodefense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 7: Byzantine Sign Flip - Scale 1.0 WITH DEFENSE
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Sign Flip (Scale 1.0) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="sign_flip" byzantine-scale=1.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_signflip_scale1_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 8: Byzantine Sign Flip - Scale 2.0
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Sign Flip (Scale 2.0) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="sign_flip" byzantine-scale=2.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_signflip_scale2_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 9: Byzantine Sign Flip - Scale 5.0
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Sign Flip (Scale 5.0) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="sign_flip" byzantine-scale=5.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_signflip_scale5_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# =================================================================
# BYZANTINE ATTACKS - SCALED
# =================================================================
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  BYZANTINE ATTACKS - SCALED${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 10: Byzantine Scaled - Scale 2.0
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Scaled (Scale 2.0) - NO DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="scaled" byzantine-scale=2.0 enable-defense=false save-dir="'$SAVE_DIR'/byzantine_scaled_scale2_nodefense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 11: Byzantine Scaled - Scale 2.0 WITH DEFENSE
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Scaled (Scale 2.0) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="scaled" byzantine-scale=2.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_scaled_scale2_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 12: Byzantine Scaled - Scale 5.0
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Scaled (Scale 5.0) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="scaled" byzantine-scale=5.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_scaled_scale5_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 13: Byzantine Scaled - Scale 10.0
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Scaled (Scale 10.0) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="scaled" byzantine-scale=10.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_scaled_scale10_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# =================================================================
# BYZANTINE ATTACKS - RANDOM
# =================================================================
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  BYZANTINE ATTACKS - RANDOM${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 14: Byzantine Random - Scale 1.0
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Random (Scale 1.0) - NO DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="random" byzantine-scale=1.0 enable-defense=false save-dir="'$SAVE_DIR'/byzantine_random_scale1_nodefense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 15: Byzantine Random - Scale 1.0 WITH DEFENSE
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Random (Scale 1.0) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="random" byzantine-scale=1.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_random_scale1_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 16: Byzantine Random - Scale 3.0
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Random (Scale 3.0) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="random" byzantine-scale=3.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_random_scale3_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# =================================================================
# GAUSSIAN NOISE ATTACKS
# =================================================================
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  GAUSSIAN NOISE ATTACKS${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 17: Gaussian - Mild (std=0.01)
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Gaussian Mild (std=0.01) - NO DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.01 enable-defense=false save-dir="'$SAVE_DIR'/gaussian_mild_nodefense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 18: Gaussian - Mild (std=0.01) WITH DEFENSE
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Gaussian Mild (std=0.01) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.01 enable-defense=true save-dir="'$SAVE_DIR'/gaussian_mild_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 19: Gaussian - Medium (std=0.1)
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Gaussian Medium (std=0.1) - NO DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.1 enable-defense=false save-dir="'$SAVE_DIR'/gaussian_medium_nodefense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 20: Gaussian - Medium (std=0.1) WITH DEFENSE
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Gaussian Medium (std=0.1) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.1 enable-defense=true save-dir="'$SAVE_DIR'/gaussian_medium_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 21: Gaussian - Strong (std=0.2)
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Gaussian Strong (std=0.2) - NO DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.2 enable-defense=false save-dir="'$SAVE_DIR'/gaussian_strong_nodefense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 22: Gaussian - Strong (std=0.2) WITH DEFENSE
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Gaussian Strong (std=0.2) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.2 enable-defense=true save-dir="'$SAVE_DIR'/gaussian_strong_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 23: Gaussian - Very Strong (std=0.5)
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Gaussian Very Strong (std=0.5) WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.5 enable-defense=true save-dir="'$SAVE_DIR'/gaussian_verystrong_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# =================================================================
# MIXED SCENARIOS
# =================================================================
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  MIXED ATTACK RATIO TESTS${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Test 24: Byzantine Sign Flip - 10% attackers
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Sign Flip - 10% attackers WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.1 byzantine-type="sign_flip" byzantine-scale=2.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_10pct_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 25: Byzantine Sign Flip - 20% attackers
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Sign Flip - 20% attackers WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.2 byzantine-type="sign_flip" byzantine-scale=2.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_20pct_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# Test 26: Byzantine Sign Flip - 40% attackers
TEST_NUM=$((TEST_NUM + 1))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 TEST $TEST_NUM/$TOTAL_TESTS: Byzantine Sign Flip - 40% attackers WITH DEFENSE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.4 byzantine-type="sign_flip" byzantine-scale=2.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_40pct_defense" save-interval=5'

echo -e "${GREEN}✓ Test $TEST_NUM completed${NC}"
sleep 2

# =================================================================
# FINAL SUMMARY
# =================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ ALL $TOTAL_TESTS TESTS COMPLETED!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "📁 Results saved to: $SAVE_DIR"
echo ""
echo "Test Summary:"
echo "  • Baseline: 1 test"
echo "  • Label Flip: 4 tests"
echo "  • Byzantine Sign Flip: 4 tests"
echo "  • Byzantine Scaled: 4 tests"
echo "  • Byzantine Random: 3 tests"
echo "  • Gaussian Noise: 7 tests"
echo "  • Mixed Ratios: 3 tests"
echo ""
echo "Directories created:"
ls -1 "$SAVE_DIR" | nl
echo ""
echo -e "${YELLOW}📊 To analyze all results, run:${NC}"
echo -e "${YELLOW}  python check_attack_results.py${NC}"
echo ""
echo -e "${BLUE}🧹 To clean up:${NC}"
echo -e "${BLUE}  rm -rf $SAVE_DIR${NC}"
echo ""