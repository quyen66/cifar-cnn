#!/bin/bash
# =================================================================
# CONFIGURATION
# =================================================================
NUM_CLIENTS=40
NUM_ROUNDS=50
SAVE_DIR="results/defense_test"


echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  FEDERATED LEARNING DEFENSE TEST SUITE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  • Total clients: $NUM_CLIENTS"
echo "  • Server rounds: $NUM_ROUNDS"
echo "  • Results directory: $SAVE_DIR"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# =================================================================
# TEST 1: Byzantine Attack (30%) - NO DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 1: Byzantine 30% - NO DEFENSE (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="sign_flip" byzantine-scale=2.0 enable-defense=false save-dir="'$SAVE_DIR'/byzantine_nodefense" save-interval=5'

echo ""
echo "✓ Test 1 completed"
echo ""
sleep 2

# =================================================================
# TEST 2: Byzantine Attack (30%) - WITH DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 2: Byzantine 30% - WITH DEFENSE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="byzantine" attack-ratio=0.3 byzantine-type="sign_flip" byzantine-scale=2.0 enable-defense=true save-dir="'$SAVE_DIR'/byzantine_defense" save-interval=5'

echo ""
echo "✓ Test 2 completed"
echo ""
sleep 2

# =================================================================
# TEST 3: Gaussian Noise (30%) - NO DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 3: Gaussian 30% - NO DEFENSE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.2 enable-defense=false save-dir="'$SAVE_DIR'/gaussian_nodefense" save-interval=5'

echo ""
echo "✓ Test 3 completed"
echo ""
sleep 2

# =================================================================
# TEST 4: Gaussian Noise (30%) - WITH DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 4: Gaussian 30% - WITH DEFENSE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config 'num-clients='$NUM_CLIENTS' num-server-rounds='$NUM_ROUNDS' partition-type="iid" attack-type="gaussian" attack-ratio=0.3 noise-std=0.2 enable-defense=true save-dir="'$SAVE_DIR'/gaussian_defense" save-interval=5'

echo ""
echo "✓ Test 4 completed"
echo ""

# =================================================================
# SUMMARY
# =================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL TESTS COMPLETED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Results saved to: $SAVE_DIR"
echo ""
echo "Test folders:"
echo "  1. $SAVE_DIR/byzantine_nodefense"
echo "  2. $SAVE_DIR/byzantine_defense"
echo "  3. $SAVE_DIR/gaussian_nodefense"
echo "  4. $SAVE_DIR/gaussian_defense"
echo ""
echo "Verification:"
ls -lh "$SAVE_DIR"
echo ""
echo "To analyze results, run:"
echo "  python analyze_defense_results.py $SAVE_DIR"
echo ""
echo "To clean up:"
echo "  rm -rf $SAVE_DIR"
echo ""