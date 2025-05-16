import os
import glob
import shutil
import subprocess
import concurrent.futures
from pathlib import Path
import uuid
import logging
from tqdm import tqdm


INPUT_DIR = ""
OUTPUT_DIR = ""
TEMP_DIR = "/tmp/yosys_parallel"
MAX_WORKERS = 80
TIMEOUT_SECONDS = 900

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='yosys_parallel.log')



YOSYS_TEMPLATE = """
# yosys优化脚本
read_verilog -sv {input_file}
synth -top {top_module}
dfflibmap -liberty nangate.lib
abc -liberty nangate.lib
clean -purge
write_verilog {output_file}
stat
"""

def get_top_module_name(file_path):
    
    try:
        
        return Path(file_path).stem
    except Exception as e:
        logging.error(f"cannot extract module name from {file_path}: {e}")
        return Path(file_path).stem  

def process_file(file_path):
    
    try:
        
        unique_id = str(uuid.uuid4())
        work_dir = os.path.join(TEMP_DIR, f"worker_{unique_id}")
        os.makedirs(work_dir, exist_ok=True)
        
        
        rel_path = os.path.relpath(file_path, INPUT_DIR)
        output_file = os.path.join(OUTPUT_DIR, rel_path)
        log_file = output_file.replace(".v", "_yosys.log")
        
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        
        top_module = get_top_module_name(file_path)
        script_path = os.path.join(work_dir, f"{unique_id}.ys")
        with open(script_path, "w") as f:
            f.write(YOSYS_TEMPLATE.format(
                input_file=file_path,
                top_module=top_module,
                output_file=output_file
            ))
        
        
        result = subprocess.run(
            ["yosys", "-Q", script_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )
        
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
        
        
        success = result.returncode == 0 and os.path.exists(output_file)
        
        
        try:
            shutil.rmtree(work_dir)
        except Exception as e:
            logging.warning(f"failed to clean up working directory: {e}")
        
        if not success:
            logging.error(f"failed to process file: {file_path}, exit code: {result.returncode}")
            return False
            
        return True
    
    except Exception as e:
        logging.error(f"error processing file {file_path}: {str(e)}")
        return False

def main():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    
    verilog_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.v"), recursive=True)
    total_files = len(verilog_files)
    logging.info(f"found {total_files} Verilog files")
    print(f"found {total_files} Verilog files")
    
    
    processed_count = 0
    success_count = 0
    failure_count = 0
    
    with tqdm(total=total_files) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            future_to_file = {executor.submit(process_file, file): file for file in verilog_files}
            
            
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    success = future.result()
                    processed_count += 1
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception as e:
                    logging.error(f"error processing file {file}: {str(e)}")
                    processed_count += 1
                    failure_count += 1
                
                pbar.update(1)
                if processed_count % 100 == 0 or processed_count == total_files:
                    logging.info(f"progress: {processed_count}/{total_files}, success: {success_count}, failure: {failure_count}")
    
    
    print(f"processed {total_files} files, success: {success_count}, failure: {failure_count}")
    logging.info(f"processed {total_files} files, success: {success_count}, failure: {failure_count}")
    
    
    try:
        shutil.rmtree(TEMP_DIR)
        logging.info(f"cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        logging.warning(f"failed to clean up main temporary directory: {e}")

if __name__ == "__main__":
    main()