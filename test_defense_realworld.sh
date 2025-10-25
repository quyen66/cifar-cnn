#!/bin/bash
# =================================================================
# TEST LAYER 1 DEFENSE - REAL WORLD ATTACKS
# =================================================================

echo "========================================================================"
echo "🧪 TESTING LAYER 1 DEFENSE WITH REAL ATTACKS"
echo "========================================================================"
echo ""

# Configuration
NUM_CLIENTS=30  # MUST MATCH pyproject.toml num-supernodes!
NUM_ROUNDS=20  # Short test
SAVE_DIR="results/defense_test"

# =================================================================
# TEST 1: Byzantine Attack (30%) - NO DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 1: Byzantine 30% - NO DEFENSE (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config \
  "num-clients=$NUM_CLIENTS \
   num-server-rounds=$NUM_ROUNDS \
   partition-type=\"iid\" \
   attack-type=\"byzantine\" \
   attack-ratio=0.3 \
   byzantine-type=\"sign_flip\" \
   byzantine-scale=10.0 \
   enable-defense=false \
   save-dir=\"$SAVE_DIR/byzantine_nodefense\" \
   save-interval=5"

echo ""
echo "✓ Test 1 complete"
echo ""

# =================================================================
# TEST 2: Byzantine Attack (30%) - WITH LAYER 1 DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 2: Byzantine 30% - WITH LAYER 1 DEFENSE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config \
  "num-clients=$NUM_CLIENTS \
   num-server-rounds=$NUM_ROUNDS \
   partition-type=\"iid\" \
   attack-type=\"byzantine\" \
   attack-ratio=0.3 \
   byzantine-type=\"sign_flip\" \
   byzantine-scale=10.0 \
   enable-defense=true \
   save-dir=\"$SAVE_DIR/byzantine_defense\" \
   save-interval=5"

echo ""
echo "✓ Test 2 complete"
echo ""

# =================================================================
# TEST 3: Gaussian Attack (30%) - NO DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 3: Gaussian 30% - NO DEFENSE (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config \
  "num-clients=$NUM_CLIENTS \
   num-server-rounds=$NUM_ROUNDS \
   partition-type=\"iid\" \
   attack-type=\"gaussian\" \
   attack-ratio=0.3 \
   noise-std=0.5 \
   enable-defense=false \
   save-dir=\"$SAVE_DIR/gaussian_nodefense\" \
   save-interval=5"

echo ""
echo "✓ Test 3 complete"
echo ""

# =================================================================
# TEST 4: Gaussian Attack (30%) - WITH LAYER 1 DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 4: Gaussian 30% - WITH LAYER 1 DEFENSE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config \
  "num-clients=$NUM_CLIENTS \
   num-server-rounds=$NUM_ROUNDS \
   partition-type=\"iid\" \
   attack-type=\"gaussian\" \
   attack-ratio=0.3 \
   noise-std=0.5 \
   enable-defense=true \
   save-dir=\"$SAVE_DIR/gaussian_defense\" \
   save-interval=5"

echo ""
echo "✓ Test 4 complete"
echo ""

# =================================================================
# TEST 5: Label Flipping (30%) - NO DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 5: Label Flip 30% - NO DEFENSE (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config \
  "num-clients=$NUM_CLIENTS \
   num-server-rounds=$NUM_ROUNDS \
   partition-type=\"iid\" \
   attack-type=\"label_flip\" \
   attack-ratio=0.3 \
   flip-type=\"reverse\" \
   flip-probability=1.0 \
   enable-defense=false \
   save-dir=\"$SAVE_DIR/labelflip_nodefense\" \
   save-interval=5"

echo ""
echo "✓ Test 5 complete"
echo ""

# =================================================================
# TEST 6: Label Flipping (30%) - WITH LAYER 1 DEFENSE
# =================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST 6: Label Flip 30% - WITH LAYER 1 DEFENSE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

flwr run . --run-config \
  "num-clients=$NUM_CLIENTS \
   num-server-rounds=$NUM_ROUNDS \
   partition-type=\"iid\" \
   attack-type=\"label_flip\" \
   attack-ratio=0.3 \
   flip-type=\"reverse\" \
   flip-probability=1.0 \
   enable-defense=true \
   save-dir=\"$SAVE_DIR/labelflip_defense\" \
   save-interval=5"

echo ""
echo "✓ Test 6 complete"
echo ""

# =================================================================
# SUMMARY
# =================================================================
echo "========================================================================"
echo "📊 ALL TESTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: $SAVE_DIR/"
echo ""
echo "Test Summary:"
echo "  ✓ Test 1: Byzantine 30% - No Defense"
echo "  ✓ Test 2: Byzantine 30% - With Defense"
echo "  ✓ Test 3: Gaussian 30% - No Defense"
echo "  ✓ Test 4: Gaussian 30% - With Defense"
echo "  ✓ Test 5: Label Flip 30% - No Defense"
echo "  ✓ Test 6: Label Flip 30% - With Defense"
echo ""
echo "Next steps:"
echo "  1. Check model checkpoints in $SAVE_DIR/"
echo "  2. Run: python analyze_defense_results.py"
echo "  3. Compare accuracy curves: defense vs no defense"
echo ""
echo "========================================================================"