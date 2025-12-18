import os
import subprocess
import re
import csv
import time
import statistics
import sys
import json
import glob
import shutil
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

JSON_CONFIG_FILE = "h_weights_dense_01_20251213_111350.json"
TOML_FILE = "pyproject.toml"
CSV_FILE = "optimization_results.csv"
LOGS_DIR = "optimization_logs" 
BASE_SAVE_DIR = "saved_models"
SAVE_INTERVAL = 10 

ATTACK_SCENARIOS = [
    "none",            
    "random_noise",    
    "gaussian_noise",  
    "sign_flip",       
    "label_flip",      
    "backdoor",        
    "alie",            
    "minmax"           
]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_configs_from_json(json_path):
    if not os.path.exists(json_path):
        print(f"❌ Error: Config file '{json_path}' not found!")
        sys.exit(1)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        configs = data.get("suite", {}).get("h_weights", [])
        if not configs:
            print("❌ Error: No 'h_weights' found in JSON 'suite'!")
            sys.exit(1)
        print(f"✅ Loaded {len(configs)} configurations from {json_path}")
        return configs
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        sys.exit(1)

def get_round_from_dirname(dirname):
    """Lấy số round từ tên thư mục (vd: 'model_r20' -> 20)."""
    numbers = re.findall(r'\d+', dirname)
    if numbers:
        return int(numbers[-1])
    return 0

def manage_checkpoints(save_dir):
    """Xử lý dọn dẹp checkpoint, giữ lại file mới nhất."""
    if not os.path.exists(save_dir):
        return ""
    
    subdirs = [d for d in os.listdir(save_dir) 
               if os.path.isdir(os.path.join(save_dir, d)) and d.startswith("model_r")]
    
    if not subdirs:
        return ""
    
    sorted_dirs = sorted(subdirs, key=get_round_from_dirname, reverse=True)
    latest_dir = sorted_dirs[0]
    latest_round = get_round_from_dirname(latest_dir)
    latest_model_path = os.path.join(save_dir, latest_dir, "model.pth")
    
    if len(sorted_dirs) > 1:
        print(f"   🧹 Cleanup: Keeping round {latest_round}, deleting {len(sorted_dirs)-1} old checkpoints...")
        for old_dir in sorted_dirs[1:]:
            full_old_path = os.path.join(save_dir, old_dir)
            try:
                shutil.rmtree(full_old_path)
            except OSError as e:
                print(f"      Warning: Could not delete {full_old_path}: {e}")
                
    if os.path.exists(latest_model_path):
        return latest_model_path
    
    return ""

def update_toml_config(weight_cv, weight_sim, weight_cluster, attack_type, save_dir, resume_from):
    """Cập nhật toml với đường dẫn chuẩn hóa."""
    try:
        with open(TOML_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        content = re.sub(r'weight-cv\s*=\s*[\d\.]+', f'weight-cv = {weight_cv}', content)
        content = re.sub(r'weight-sim\s*=\s*[\d\.]+', f'weight-sim = {weight_sim}', content)
        content = re.sub(r'weight-cluster\s*=\s*[\d\.]+', f'weight-cluster = {weight_cluster}', content)

        content = re.sub(r'attack-type\s*=\s*".*"', f'attack-type = "{attack_type}"', content)
        ratio = 0.0 if attack_type == "none" else 0.3
        content = re.sub(r'attack-ratio\s*=\s*[\d\.]+', f'attack-ratio = {ratio}', content)
        
        if save_dir:
            save_dir_clean = save_dir.replace(os.sep, '/')
            content = re.sub(r'save-dir\s*=\s*".*"', f'save-dir = "{save_dir_clean}"', content)
        
        if resume_from:
            resume_from_clean = resume_from.replace(os.sep, '/')
            content = re.sub(r'resume-from\s*=\s*".*"', f'resume-from = "{resume_from_clean}"', content)
        else:
            content = re.sub(r'resume-from\s*=\s*".*"', f'resume-from = ""', content)
        
        if 'save-interval' in content:
            content = re.sub(r'save-interval\s*=\s*\d+', f'save-interval = {SAVE_INTERVAL}', content)

        with open(TOML_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except Exception as e:
        print(f"❌ Error updating TOML: {e}")
        sys.exit(1)

def get_toml_config_value(content, key, default="?"):
    """Helper đọc giá trị từ file TOML (cho việc hiển thị)."""
    match = re.search(fr'^\s*{key}\s*=\s*(.+?)\s*(?:#.*)?$', content, re.MULTILINE)
    if match:
        val = match.group(1).strip('"\'')
        return val
    return default

def print_full_config(cfg_name, attack_type):
    """In toàn bộ cấu hình hiện tại để dễ theo dõi (ĐÃ KHÔI PHỤC)."""
    try:
        with open(TOML_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("\n" + "━"*70)
        print(f"📋 CONFIGURATION SUMMARY: {cfg_name}")
        print("━"*70)
        
        rounds = get_toml_config_value(content, "num-server-rounds")
        clients = get_toml_config_value(content, "num-clients")
        part_type = get_toml_config_value(content, "partition-type")
        alpha = get_toml_config_value(content, "alpha")
        
        print(f"🌍 GLOBAL:  Rounds={rounds} | Clients={clients}")
        print(f"📊 DATA:    Partition={part_type.upper()} | Alpha={alpha}")
        
        atk_type = get_toml_config_value(content, "attack-type")
        atk_ratio = get_toml_config_value(content, "attack-ratio")
        print(f"⚔️  ATTACK:  Type={atk_type} | Ratio={atk_ratio}")
        
        w_cv = get_toml_config_value(content, "weight-cv")
        w_sim = get_toml_config_value(content, "weight-sim")
        w_clus = get_toml_config_value(content, "weight-cluster")
        print(f"🛡️  DEFENSE: H-Weights [CV={w_cv}, Sim={w_sim}, Cluster={w_clus}]")
        
        save_dir = get_toml_config_value(content, "save-dir")
        resume = get_toml_config_value(content, "resume-from")
        save_int = get_toml_config_value(content, "save-interval")
        
        resume_status = f"🔄 YES ({os.path.basename(resume)})" if resume else "🆕 NO"
        print(f"💾 SYSTEM:  SaveInterval={save_int} | Resume={resume_status}")
        print(f"📂 DIR:     {save_dir}")
        
        print("-" * 70)
        print(f"🚀 CMD:     flwr run .")
        print("━"*70 + "\n")
        
    except Exception as e:
        print(f"⚠️ Error reading config for display: {e}")

def parse_full_metrics(log_output):
    """
    Trích xuất toàn bộ các chỉ số quan trọng từ Log.
    """
    metrics = {
        "Accuracy": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
        "FPR": 0.0,
        "TP": 0, "FP": 0, "FN": 0, "TN": 0
    }
    
    # Accuracy
    acc_matches = re.findall(r"\[AGGREGATED\] Accuracy:\s*([\d\.]+)", log_output)
    if acc_matches:
        metrics["Accuracy"] = float(acc_matches[-1])
        
    # Confusion Matrix
    cm_matches = re.findall(r"TP=(\d+),\s*FP=(\d+),\s*FN=(\d+),\s*TN=(\d+)", log_output)
    if cm_matches:
        tp, fp, fn, tn = cm_matches[-1]
        metrics.update({
            "TP": int(tp), "FP": int(fp), 
            "FN": int(fn), "TN": int(tn)
        })
        
    # Advanced Metrics
    adv_matches = re.findall(r"Precision=([\d\.]+),\s*Recall=([\d\.]+),\s*F1=([\d\.]+),\s*FPR=([\d\.]+)", log_output)
    if adv_matches:
        prec, rec, f1, fpr = adv_matches[-1]
        metrics.update({
            "Precision": float(prec), 
            "Recall": float(rec), 
            "F1": float(f1), 
            "FPR": float(fpr)
        })
        
    return metrics

def save_run_log(config_name, attack_type, log_content):
    """Lưu toàn bộ log chạy ra file riêng để debug sau này."""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)
        
    filename = f"{config_name}_{attack_type}.log"
    filepath = os.path.join(LOGS_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"--- LOG START: {datetime.now()} ---\n")
            f.write(f"Config: {config_name}\n")
            f.write(f"Attack: {attack_type}\n")
            f.write("-" * 50 + "\n")
            f.write(log_content)
            f.write("\n" + "-" * 50 + "\n")
            f.write("--- LOG END ---\n")
    except Exception as e:
        print(f"   ⚠️ Error saving log file: {e}")

def get_existing_data():
    completed_runs = set()
    existing_results = {}
    if not os.path.isfile(CSV_FILE):
        return completed_runs, existing_results
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cfg = row.get('Config_Name')
                atk = row.get('Attack_Type')
                try:
                    acc = float(row.get('Accuracy', 0.0))
                except ValueError:
                    acc = 0.0
                
                # Chỉ đánh dấu là hoàn thành nếu có Accuracy > 0
                if cfg and atk and acc > 0:
                    completed_runs.add((cfg, atk))
                    if cfg not in existing_results:
                        existing_results[cfg] = {}
                    existing_results[cfg][atk] = acc
    except Exception:
        pass
    return completed_runs, existing_results

def run_simulation(config_name, attack_type, save_dir):
    # IN LẠI THÔNG TIN RUN (ĐÃ KHÔI PHỤC)
    print(f"\n   ▶ STARTING RUN: {config_name} | Attack: {attack_type}")
    print(f"     Storage: {save_dir}")
    print("   " + "-"*50)
    
    full_output = ""
    try:
        process = subprocess.Popen(
            ["flwr", "run", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        for line in process.stdout:
            print(line, end='')
            full_output += line
            sys.stdout.flush()
        process.wait()
        
        print("   " + "-"*50)
        
        # Cleanup & Save Log
        latest = manage_checkpoints(save_dir)
        if latest:
            dir_name = os.path.basename(os.path.dirname(latest))
            print(f"   💾 Storage Optimized: Kept {dir_name}")
            
        save_run_log(config_name, attack_type, full_output)

        metrics = parse_full_metrics(full_output)
        
        if metrics["Accuracy"] > 0:
            print(f"   ✅ Finished. Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f} | FPR: {metrics['FPR']:.4f}")
        else:
            print(f"   ⚠️ Finished but no accuracy found (Check logs).")
            
        return metrics
        
    except Exception as e:
        print(f"\n   ❌ CRITICAL ERROR: {e}")
        return {"Accuracy": 0.0}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("🚀 STARTING OPTIMIZATION (FULL DISPLAY + METRICS)")
    weight_configs = load_configs_from_json(JSON_CONFIG_FILE)
    
    # CSV Headers (Đầy đủ)
    headers = [
        "Config_Name", 
        "Weight_CV", "Weight_Sim", "Weight_Cluster", 
        "Attack_Type", 
        "Accuracy", "Precision", "Recall", "F1", "FPR",
        "TP", "FP", "FN", "TN",
        "Timestamp"
    ]
    
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    else:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            if "F1" not in header_line:
                print("⚠️  WARNING: Existing CSV lacks new metric columns.")
                print("   Recommendation: Delete 'optimization_results.csv' to avoid format errors.")
                time.sleep(3)

    completed_runs, existing_results = get_existing_data()

    for config_item in weight_configs:
        cid = config_item.get('config_id', 'Unknown')
        profile = config_item.get('profile', 'custom')
        cfg_name = f"Cfg{cid}_{profile}"
        
        w_cv = config_item.get('weight_cv', 0.33)
        w_sim = config_item.get('weight_sim', 0.33)
        w_cluster = config_item.get('weight_cluster', 0.33)

        for attack in ATTACK_SCENARIOS:
            if (cfg_name, attack) in completed_runs:
                old_acc = existing_results.get(cfg_name, {}).get(attack, 0.0)
                # Skip in lặng hoặc print ngắn gọn
                continue
            
            # Setup
            current_save_dir = os.path.join(BASE_SAVE_DIR, cfg_name, attack)
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir, exist_ok=True)
            
            # Resume Check
            latest_checkpoint = manage_checkpoints(current_save_dir)
            latest_path_for_toml = latest_checkpoint if latest_checkpoint else ""
            
            # Update Config
            update_toml_config(w_cv, w_sim, w_cluster, attack, current_save_dir, latest_path_for_toml)
            
            # Print Info (Cái bạn cần)
            print_full_config(cfg_name, attack)
            
            # Run
            metrics = run_simulation(cfg_name, attack, current_save_dir)
            
            # Save Results
            if metrics["Accuracy"] > 0:
                row_data = [
                    cfg_name, 
                    w_cv, w_sim, w_cluster, 
                    attack, 
                    metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1"], metrics["FPR"],
                    metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
                
                with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                print(f"   📝 Saved metrics to CSV")
            
            time.sleep(1)

    print("\n🏆 ALL TASKS COMPLETED.")

if __name__ == "__main__":
    main()