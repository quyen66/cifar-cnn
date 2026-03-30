#!/usr/bin/env python3
"""
MAD k_reject / k_flag Parameter Test Runner
=============================================
Chạy tất cả attack scenarios cho từng (k_reject, k_flag) config.

Features:
  - Resume tự động khi bị ngắt (dựa vào CSV)
  - Checkpoint management (giữ checkpoint mới nhất)
  - Lưu log riêng từng run
  - CSV kết quả đầy đủ

Usage:
    python3 run_mad_param_tests.py mad_params_05_20260323_120000.json
    python3 run_mad_param_tests.py mad_params_05_20260323_120000.json --attacks alie none minmax
    python3 run_mad_param_tests.py mad_params_05_20260323_120000.json --dry-run
"""

import os
import sys
import re
import csv
import json
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TOML_FILE    = "pyproject.toml"
CSV_FILE     = "mad_param_results.csv"
LOGS_DIR     = "mad_param_logs"
BASE_SAVE_DIR = "saved_models"
SAVE_INTERVAL = 10

ATTACK_SCENARIOS = [
    "none",
    "alie",
    "minmax",
    "minsum",
    "label_flip",
    "backdoor",
    "gaussian_noise",
    "random_noise",
    "sign_flip",
    "on_off",
    "slow_poison",
]

CSV_HEADERS = [
    "Config_Name",
    "k_reject", "k_flag", "gap", "profile",
    "Attack_Type",
    "Accuracy", "Precision", "Recall", "F1", "FPR",
    "TP", "FP", "FN", "TN",
    "Timestamp",
]

# ==============================================================================
# JSON LOADER
# ==============================================================================

def load_configs(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        print(f"❌ Config file not found: {json_path}")
        sys.exit(1)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    configs = data.get("suite", {}).get("mad_params", [])
    if not configs:
        print("❌ No 'mad_params' key in suite — did you use generate_mad_params.py?")
        sys.exit(1)
    print(f"✅ Loaded {len(configs)} configs from {json_path}")
    return configs

# ==============================================================================
# CHECKPOINT MANAGEMENT
# ==============================================================================

def _round_from_dirname(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0

def manage_checkpoints(save_dir: str) -> str:
    """Giữ checkpoint mới nhất, xóa cũ. Trả về path file .pth mới nhất."""
    if not os.path.exists(save_dir):
        return ""
    subdirs = [d for d in os.listdir(save_dir)
               if os.path.isdir(os.path.join(save_dir, d)) and d.startswith("model_r")]
    if not subdirs:
        return ""
    subdirs.sort(key=_round_from_dirname, reverse=True)
    latest = subdirs[0]
    if len(subdirs) > 1:
        print(f"   🧹 Cleanup: keeping round {_round_from_dirname(latest)}, "
              f"removing {len(subdirs)-1} old checkpoint(s)...")
        for old in subdirs[1:]:
            try:
                shutil.rmtree(os.path.join(save_dir, old))
            except OSError as e:
                print(f"      ⚠ Could not delete {old}: {e}")
    pth = os.path.join(save_dir, latest, "model.pth")
    return pth if os.path.exists(pth) else ""

# ==============================================================================
# TOML UPDATER
# ==============================================================================

def update_toml(k_reject: float, k_flag: float,
                attack: str, save_dir: str, resume_from: str) -> None:
    with open(TOML_FILE, encoding="utf-8") as f:
        content = f.read()

    # MAD params
    content = re.sub(r"mad-k-reject\s*=\s*[\d\.]+",
                     f"mad-k-reject = {k_reject}", content)
    content = re.sub(r"mad-k-flag\s*=\s*[\d\.]+",
                     f"mad-k-flag = {k_flag}", content)

    # Attack
    content = re.sub(r'attack-type\s*=\s*".*?"',
                     f'attack-type = "{attack}"', content)
    ratio = 0.0 if attack == "none" else 0.3
    content = re.sub(r"attack-ratio\s*=\s*[\d\.]+",
                     f"attack-ratio = {ratio}", content)

    # Paths
    save_clean = save_dir.replace(os.sep, "/")
    content = re.sub(r'save-dir\s*=\s*".*?"',
                     f'save-dir = "{save_clean}"', content)

    if resume_from:
        res_clean = resume_from.replace(os.sep, "/")
        content = re.sub(r'resume-from\s*=\s*".*?"',
                         f'resume-from = "{res_clean}"', content)
    else:
        content = re.sub(r'resume-from\s*=\s*".*?"',
                         'resume-from = ""', content)

    content = re.sub(r"save-interval\s*=\s*\d+",
                     f"save-interval = {SAVE_INTERVAL}", content)

    with open(TOML_FILE, "w", encoding="utf-8") as f:
        f.write(content)

# ==============================================================================
# TOML READER (for display)
# ==============================================================================

def _toml_val(content: str, key: str, default: str = "?") -> str:
    m = re.search(fr"^\s*{re.escape(key)}\s*=\s*(.+?)(?:\s*#.*)?\s*$",
                  content, re.MULTILINE)
    if m:
        return m.group(1).strip().strip('"\'')
    return default

def print_config_summary(cfg_name: str, k_reject: float, k_flag: float,
                         attack: str) -> None:
    with open(TOML_FILE, encoding="utf-8") as f:
        c = f.read()

    print("\n" + "━" * 68)
    print(f"📋  {cfg_name}  |  attack={attack}")
    print("━" * 68)
    print(f"  MAD:     k_reject={k_reject}  k_flag={k_flag}  "
          f"(gap={round(k_reject-k_flag,1)})")
    print(f"  Rounds:  {_toml_val(c,'num-server-rounds')}  "
          f"Clients: {_toml_val(c,'num-clients')}")
    print(f"  Attack:  {_toml_val(c,'attack-type')}  "
          f"ratio={_toml_val(c,'attack-ratio')}")
    print(f"  Save:    {_toml_val(c,'save-dir')}")
    resume = _toml_val(c, "resume-from")
    print(f"  Resume:  {'YES → ' + os.path.basename(resume) if resume else 'NO (fresh start)'}")
    print("━" * 68)

# ==============================================================================
# METRICS PARSER
# ==============================================================================

def parse_metrics(output: str) -> Dict:
    m: Dict = {
        "Accuracy": 0.0,
        "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "FPR": 0.0,
        "TP": 0, "FP": 0, "FN": 0, "TN": 0,
    }
    # Accuracy — take last match
    acc = re.findall(r"\[AGGREGATED\] Accuracy:\s*([\d\.]+)", output)
    if acc:
        m["Accuracy"] = float(acc[-1])
    # Confusion matrix
    cm = re.findall(r"TP=(\d+),\s*FP=(\d+),\s*FN=(\d+),\s*TN=(\d+)", output)
    if cm:
        tp, fp, fn, tn = cm[-1]
        m.update({"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)})
    # Precision / Recall / F1 / FPR
    adv = re.findall(
        r"Precision=([\d\.]+),\s*Recall=([\d\.]+),\s*F1=([\d\.]+),\s*FPR=([\d\.]+)",
        output)
    if adv:
        pr, rc, f1, fpr = adv[-1]
        m.update({"Precision": float(pr), "Recall": float(rc),
                  "F1": float(f1), "FPR": float(fpr)})
    return m

# ==============================================================================
# LOG SAVER
# ==============================================================================

def save_log(cfg_name: str, attack: str, content: str) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"{cfg_name}_{attack}.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"--- LOG START: {datetime.now()} ---\n")
        f.write(f"Config: {cfg_name}\nAttack: {attack}\n")
        f.write("-" * 50 + "\n")
        f.write(content)
        f.write("\n--- LOG END ---\n")

# ==============================================================================
# RESUME STATE
# ==============================================================================

def load_completed(csv_path: str) -> Tuple[set, Dict]:
    """
    Returns:
        completed: set of (cfg_name, attack) already done
        results:   dict[cfg_name][attack] = Accuracy
    """
    completed: set = set()
    results:   Dict = {}
    if not os.path.isfile(csv_path):
        return completed, results
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row.get("Config_Name", "")
            atk = row.get("Attack_Type", "")
            try:
                acc = float(row.get("Accuracy", 0))
            except ValueError:
                acc = 0.0
            if cfg and atk and acc > 0:
                completed.add((cfg, atk))
                results.setdefault(cfg, {})[atk] = acc
    return completed, results

def append_csv(row: List) -> None:
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def ensure_csv_header() -> None:
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

# ==============================================================================
# SIMULATION RUNNER
# ==============================================================================

def run_simulation(cfg_name: str, attack: str, save_dir: str,
                   dry_run: bool = False) -> Dict:
    print(f"\n  ▶ RUN: {cfg_name}  |  attack={attack}")

    if dry_run:
        print("    [DRY RUN — skipping flwr run]")
        return {"Accuracy": 0.0}

    full_output = ""
    try:
        proc = subprocess.Popen(
            ["flwr", "run", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            full_output += line
            sys.stdout.flush()
        proc.wait()
    except KeyboardInterrupt:
        print("\n  ⏸  Interrupted — saving partial log and exiting cleanly.")
        save_log(cfg_name, attack, full_output + "\n[INTERRUPTED]")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        return {"Accuracy": 0.0}

    # Cleanup old checkpoints
    latest = manage_checkpoints(save_dir)
    if latest:
        print(f"  💾 Checkpoint kept: {os.path.basename(os.path.dirname(latest))}")

    save_log(cfg_name, attack, full_output)

    metrics = parse_metrics(full_output)
    if metrics["Accuracy"] > 0:
        print(f"  ✅ Acc={metrics['Accuracy']:.4f}  "
              f"F1={metrics['F1']:.4f}  FPR={metrics['FPR']:.4f}  "
              f"TP={metrics['TP']}  FP={metrics['FP']}")
    else:
        print("  ⚠  No accuracy found — check log.")
    return metrics

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run MAD k_reject/k_flag grid tests across all attack scenarios")
    parser.add_argument("json_file",
                        help="JSON file from generate_mad_params.py")
    parser.add_argument("--attacks", nargs="+", default=None,
                        metavar="ATTACK",
                        help="Subset of attacks to run (default: all 11)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config summaries without running flwr")
    args = parser.parse_args()

    attacks = args.attacks if args.attacks else ATTACK_SCENARIOS

    # Validate attack names
    invalid = [a for a in attacks if a not in ATTACK_SCENARIOS]
    if invalid:
        print(f"❌ Unknown attack(s): {invalid}")
        print(f"   Valid: {ATTACK_SCENARIOS}")
        sys.exit(1)

    configs = load_configs(args.json_file)
    ensure_csv_header()
    completed, _ = load_completed(CSV_FILE)

    total_runs   = len(configs) * len(attacks)
    skip_count   = sum(1 for c in configs for a in attacks
                       if (f"MADcfg{c['config_id']}_{c['profile']}", a) in completed)
    remain_count = total_runs - skip_count

    print(f"\n📊 Plan: {len(configs)} configs × {len(attacks)} attacks = {total_runs} runs")
    if skip_count:
        print(f"   ⏭  Already done (CSV): {skip_count}  →  remaining: {remain_count}")

    if args.dry_run:
        print("\n[DRY RUN MODE — showing first 5 configs only]\n")

    done_this_session = 0

    for cfg in configs:
        cid     = cfg["config_id"]
        k_rej   = cfg["k_reject"]
        k_flg   = cfg["k_flag"]
        profile = cfg["profile"]
        gap     = cfg["gap"]

        cfg_name = f"MADcfg{cid}_{profile}"

        for attack in attacks:
            run_key = (cfg_name, attack)

            # Resume check
            if run_key in completed:
                continue

            # Paths
            save_dir = os.path.join(BASE_SAVE_DIR, cfg_name, attack)
            os.makedirs(save_dir, exist_ok=True)

            # Resume from latest checkpoint if exists
            latest_ckpt = manage_checkpoints(save_dir)

            # Update TOML
            update_toml(k_rej, k_flg, attack, save_dir, latest_ckpt)

            # Print summary
            print_config_summary(cfg_name, k_rej, k_flg, attack)

            # Run
            metrics = run_simulation(cfg_name, attack, save_dir,
                                     dry_run=args.dry_run)

            # Save to CSV (only if Accuracy > 0 or dry_run skipped)
            if metrics["Accuracy"] > 0:
                row = [
                    cfg_name,
                    k_rej, k_flg, gap, profile,
                    attack,
                    metrics["Accuracy"],
                    metrics["Precision"], metrics["Recall"],
                    metrics["F1"], metrics["FPR"],
                    metrics["TP"], metrics["FP"],
                    metrics["FN"], metrics["TN"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ]
                append_csv(row)
                completed.add(run_key)
                done_this_session += 1
                print(f"  📝 Saved to {CSV_FILE}")

            if args.dry_run and done_this_session >= 5:
                print("\n[DRY RUN] Showed 5 configs. Exiting.")
                return

            time.sleep(1)

    print(f"\n🏆 Done. Ran {done_this_session} new experiments this session.")
    print(f"   Results: {CSV_FILE}")
    print(f"   Logs:    {LOGS_DIR}/")


if __name__ == "__main__":
    main()