import json
import re
import os
import subprocess
from tqdm import tqdm
import time
def flatten_design(input_folder, output_file):

    import os
    import subprocess
    

    yosys_commands = []

    for fname in sorted(os.listdir(input_folder)):
        if fname.endswith('.v'):
            fpath = os.path.join(input_folder, fname)
            yosys_commands.append(f"read_verilog -sv {fpath}")
    

    yosys_commands.extend([
        "hierarchy -check -top top",
        "proc",
        "flatten",
        f"write_verilog {output_file}"
    ])

    script = "; ".join(yosys_commands)

    try:

        output_dir = os.path.dirname(output_file)

        log_file = os.path.join(output_dir, "yosys_flatten.log")

        with open(log_file, "w") as log:
            subprocess.run(["yosys", "-p", script], check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"Yosys log saved to: {log_file}")

        clean_output(output_file)
    except subprocess.CalledProcessError as e:
        print("Yosys execution error:", e)

def clean_output(file):

    import re
    import os
    
    tmp_file = file + ".tmp"
    os.system(f"cp {file} {tmp_file}")
    os.remove(file)
    
    with open(tmp_file, "r") as f:
        lines = f.readlines()
        with open(file, "w") as out:
            for line in lines:
                line = re.sub(r'\(\*(.*)\*\)', '', line)
                if line.strip():
                    out.write(line)
                    
    os.remove(tmp_file)
def process_all_designs():

    with open('design_names.json', 'r') as f:
        designs = json.load(f)
    

    output_dir = ''
    os.makedirs(output_dir, exist_ok=True)
    

    for i, name in enumerate(designs):
        print(f"processing design: {name}")
        
        
        input_folder = f'new_{name}_cone'
        output_file = os.path.join(output_dir, f'test_{i}.v')
        
        try:
            
            flatten_design(input_folder, output_file)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            content = content.replace('clk', 'clock')
            
            with open(output_file, 'w') as f:
                f.write(content)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            
            content = re.sub(r'module\s+top\s*\(', f'module test_{i}(', content, count=1)
            
            with open(output_file, 'w') as f:
                f.write(content)
                
            print(f"design {name} processed")
        except Exception as e:
            print(f"error processing design {name}: {str(e)}")
            continue


def process_folder(folder_path, output_dir):

    import os
    import glob
    os.makedirs(output_dir, exist_ok=True)
    test_folders = glob.glob(os.path.join(folder_path, "test_*"))
    

    test_folders.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    
    print(f"found {len(test_folders)} test folders in {folder_path}")
    
    q = 0

    for test_folder in tqdm(test_folders):
        test_name = os.path.basename(test_folder)
        print(f"processing test: {test_name}")
        input_folder = test_folder
        output_file = os.path.join(output_dir, f'test_{q}.v')
        try:
            
            flatten_design(input_folder, output_file)
            # time.sleep(100)
            with open(output_file, 'r') as f:
                content = f.read()
            
            content = content.replace('clk', 'clock')
            
            with open(output_file, 'w') as f:
                f.write(content)
            with open(output_file, 'r') as f:
                content = f.read()
           
            content = re.sub(r'module\s+top\s*\(', f'module test_{q}(', content, count=1)

            with open(output_file, 'w') as f:
                f.write(content)
            print(f"test {test_name} processed")
            q += 1
        except Exception as e:
            print(f"error processing test {test_name}: {str(e)}")
            continue

process_folder('', '')
