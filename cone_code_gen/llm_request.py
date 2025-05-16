
import os
import re
import time
import requests
import json
from verilog_checker import VerilogChecker

def generate_prompt_with_ref_code(input_widths, output_width, input_names, output_name, ref_func, ref_code, depth, extra_info=None):

    input_count = len(input_widths)
    input_widths_str = ', '.join(str(w) for w in input_widths)
    input_names_str = ', '.join(str(n) for n in input_names)
    
    template = (
        f"You are tasked with generating a Verilog module.\n\n"
        f"**Inputs:**\n"
        f"- You are given {input_count} input signals named: {input_names_str}.\n"
        f"- Their respective bit widths are: {input_widths_str}.\n"
        f"- There is also a 1-bit input clock signal named 'clock'.\n\n"
        f"**Output:**\n"
        f"- The single output signal is named '{output_name}'.\n"
        f"- Its bit width is {output_width}, remember to declare the output port!!! You always forget to declare the output port!!!\n\n"
        f"**Module Requirements:**\n"
        f"1.  **Module Name:** The module must be named '{output_name}'.\n"
        f"2.  **Output:** The output signal '{output_name}' port must be the *only* register in the design. All other logic must be purely combinational!!!\n"
        f"3.  **Output Assignment:** Assign the value to the output '{output_name}' using a non-blocking assignment (`<=`) exclusively within an `always @(posedge clock) begin ... end` block. The other wires should be assigned in the combinational logic (not in the always block).\n"
        f"4.  **Logic Depth:** The module must have {min(depth, 7)} levels of logic depth. Do not skip any levels. Layer 1 depends on the inputs, layer 2 depends on layer 1 (and potentially inputs), and so on, up to the final layer which computes the value for '{output_name}'.\n"
        f"5.  **Intermediate Wires:**\n"
        f"    - All intermediate signals must be declared as `wire`.\n"
        f"    - Name intermediate wires strictly using the format: `layerN_wireK_widthM`.\n"
        f"        - `N`: The logic layer index (starting from 1).\n"
        f"        - `K`: The sequential index of the wire within that layer (starting from 1).\n"
        f"        - `M`: The bit width of the wire.\n"
        f"    - Example: `wire [7:0] layer1_wire1_width8;`, `wire [15:0] layer2_wire1_width16;`\n"
        f"6.  **Layer-by-Layer Generation:** Generate the combinational logic strictly layer by layer, starting from layer 1.\n"
        f"7.  **Logic Depth Definition:** A wire assigned in layer `i` must depend on at least one wire from layer `i-1`. It can also depend on wires from layers 1 to `i-1` and the primary inputs.\n"
        f"8.  **Bit Width Matching:** Pay close attention to bit widths during assignments\n"
        f"    - Avoid mismatches: Ensure the left-hand side (LHS) and right-hand side (RHS) have the same width.\n"
        f"    - Avoid redundancy: Do not assign a narrow signal to a wider wire directly (e.g., `assign wide_wire = narrow_signal;` where `width(wide_wire) > width(narrow_signal)`).\n"
        "    - **Crucially, do not use constant padding** to match widths (e.g., avoid `assign wide_wire = {{31'b0, single_bit_signal}};`). Use operations involving other signals.\n"
        f"9.  **Signal Usage:** Ensure *every* declared signal (inputs and intermediate wires) is used on the right-hand side (RHS) of at least one assignment (`=` or `<=`). Avoid unused signals.\n"
        "10. **No Constant Bit Assignments:** Do not assign constant values directly to specific bits or parts of a wire (e.g., avoid `assign my_wire[3:0] = 4'b0;` or `assign my_wire = {constant, signal};`). All assignments should derive from input signals or previously computed wire values.\n\n"
        f"11. Do not assign a existing wire or input directly to another wire (e.g., avoid `assign new_wire = existing_wire;` directly, otherwise this assignment is redundant).\n\n"
        f"**Reference Material:**\n"
        f"Reference Function:\n"
        f"```verilog\n{ref_func}\n```\n\n"
        f"Mimic the *types*, *quantities*, and *proportions* of operators found in the following reference code snippet, but adapt them to the specific requirements of this task (especially regarding bit widths and constant usage).\n\n"
        f"Reference Code Snippet:\n"
        f"```verilog\n{ref_code}\n```\n\n"
        f"Additional Information for Reference:\n"
        f"{extra_info}\n\n"
        f"You should follow the (+-*/%) arithmetic operators usage as stated above\n"
        f"**Output Format:**\n"
        f"Place the complete and pure Verilog module code between the markers `##code##` and `##end code##`.\n"
        f"```verilog\n"
        f"##code##\n"
        f"// Your Verilog code here\n"
        f"##end code##\n"
        f"```"
        f"**always do meaningful operations, not obvious simplifications in the code**\n"
        f"do not make the code too long, no more than 4 wires in each layer. and no multiple assignments for the same wire\n"
        f"directly give me the code, do not add any other text in your answer, here is the code:\n"
    )
    return template


def extract_code(text):

    try:
        start = text.index('##code##') + len('##code##')
        end = text.index('##end code##')
        code = text[start:end].strip()
        return extract_module_code(code)
    except ValueError:
        return ''

def extract_module_code(text):

    try:
        module_start = text.find('module')
        if module_start == -1:
            return text
        endmodule_end = text.rfind('endmodule') + len('endmodule')
        if endmodule_end == len('endmodule') - 1:
            return text[module_start:]
        return text[module_start:endmodule_end]
    except Exception:
        return text

def generate_code_with_openrouter(api_key, prompt, model="deepseek/deepseek-coder", temperature=0.6):

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    data = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'provider': {
            'sort': 'throughput'
        },
        'temperature': temperature
    }

    try:
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', 
                                headers=headers, 
                                json=data)
        response.raise_for_status()  
        result = response.json()
        print(f"result:\n{result}")

        if 'finish_reason' in result['choices'][0] and result['choices'][0]['finish_reason'] == 'length':
            print("warning: response is truncated")
            
            truncated_content = result['choices'][0]['message']['content']
            timestamp = int(time.time())
            output_file = f"./truncated_response_{timestamp}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(truncated_content)
            print(f"truncated response saved to: {output_file}")
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"API request error: {str(e)}")
        return None

def generate_code_for_node_with_ref(api_key, g, node, ref_func, ref_code, depth, output_dir, extra_info=None):


    parent_widths = []
    parent_names = []
    for parent in g.predecessors(node):
        parent_widths.append(g.nodes[parent].get('width', 1))
        parent_names.append(g.nodes[parent].get('name', parent))
    

    output_width = g.nodes[node].get('width', 1)
    output_name = g.nodes[node].get('name', node)
    
    max_attempts = 3 
    for attempt in range(max_attempts):
        try:
            prompt = generate_prompt_with_ref_code(
                parent_widths, 
                output_width, 
                parent_names, 
                output_name, 
                ref_func,
                ref_code, 
                depth,
                extra_info
            )
            print(f"prompt:\n{prompt}")
            
            response = generate_code_with_openrouter(api_key, prompt, model="google/gemini-2.5-flash-preview", temperature=0.7)
            if not response:
                print(f"failed to generate code for node {node}, attempt {attempt+1}/{max_attempts}")
                time.sleep(1)
                continue
            
            
            code = extract_code(response)
            if code == '':
                print(f"node {node} generated code is empty, attempt {attempt+1}/{max_attempts}")
                time.sleep(1)
                continue
            
            print(f"generated code for node {node}:\n", code)
            
            valid, err_msg = VerilogChecker.check_syntax(code)
           
            has_always = "always" in code
            has_nonblocking = "<=" in code
            has_clock = "clock" in code
            has_output = "output" in code
            if not has_always or not has_nonblocking or not has_clock or not has_output:
                print(f"warning: node {node} generated code is missing necessary keywords")
                print(f"  - always keyword exists: {has_always}")
                print(f"  - non-blocking assignment (<=) exists: {has_nonblocking}")
                print(f"  - clock keyword exists: {has_clock}")
                print(f"  - output keyword exists: {has_output}")
                valid = False
            
            if valid:
                
                file_path = os.path.join(output_dir, f"{output_name}.v")
                with open(file_path, 'w') as f:
                    f.write(code)
                return code
            else:
                print(f"node {node} generated code is not syntactically correct, attempt {attempt+1}/{max_attempts}")
            
            time.sleep(1)
        except Exception as e:
            print(f"error generating code for node {node}: {str(e)}")
            time.sleep(1)
