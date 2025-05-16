import os
import re
import subprocess
import tempfile
from threading import Timer
class VerilogChecker:
    
    
    @staticmethod
    def check_syntax(verilog_code):
        
        with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(verilog_code.encode('utf-8'))
        
        try:
            
            cmd = ["yosys", "-p", "read_verilog -sv " + temp_filename]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            
            
            if result.returncode != 0 or b"ERROR" in result.stderr:
                return False, result.stderr.decode('utf-8')
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "验证超时"
        except Exception as e:
            return False, str(e)
        finally:
            
            os.unlink(temp_filename)
    
    @staticmethod
    def check_syntax_pyverilog(verilog_code):
        
        with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(verilog_code.encode('utf-8'))
        
        try:
            from pyverilog.vparser.parser import VerilogParser
            parser = VerilogParser()
            
            try:
                ast = parser.parse([temp_filename])
                return True, ""
            except Exception as e:
                return False, str(e)
        except ImportError:
            return False, "未安装pyverilog库"
        except Exception as e:
            return False, str(e)
        finally:
            
            os.unlink(temp_filename)
    
    @staticmethod
    def check_syntax_iverilog(verilog_code: str, timeout: int = 1) -> tuple[bool, str]:
        
        timed_out = False
        p = None 

        def kill_process():
            
            nonlocal timed_out, p
            
            if p and p.poll() is None:
                try:
                    p.kill()
                    timed_out = True
                    
                except OSError:
                    
                    
                    pass 

        try:
            
            
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=True, encoding='utf-8') as tmp_file:
                tmp_file.write(verilog_code)
                
                tmp_file.flush()

                
                
                
                cmd = f"iverilog -t null -g2012 {tmp_file.name}"

                
                p = subprocess.Popen(
                    cmd,
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE  
                )

                
                timer = Timer(timeout, kill_process)
                try:
                    timer.start()
                    
                    
                    stdout, stderr = p.communicate()
                finally:
                    
                    timer.cancel()

                
                stderr_str = stderr.decode("utf-8", errors='ignore')

                
                if timed_out:
                    
                    return False, f"检查超时 ({timeout} 秒)"

                
                
                if p.returncode != 0:
                    
                    error_message = stderr_str if stderr_str else "未知编译错误 (无 stderr 输出)"
                    
                    if "syntax error" in error_message.lower() or ": error:" in error_message.lower():
                        return False, f"语法错误:\n{error_message}"
                    else:
                        return False, f"编译错误:\n{error_message}"
                else:
                    
                    
                    
                    return True, ""

        except FileNotFoundError:
            
            return False, "错误: 'iverilog' 命令未找到。请确保 Icarus Verilog 已安装并且在系统 PATH 环境变量中。"
        except Exception as e:
            
            return False, f"执行检查时发生意外错误: {e}"

    @staticmethod
    def extract_module_name(verilog_code):
        
        pattern = r"module\s+([a-zA-Z0-9_]+)"
        match = re.search(pattern, verilog_code)
        if match:
            return match.group(1)
        return None
    
    @staticmethod
    def are_functionally_equivalent(verilog_code1, verilog_code2, top_module_name=None):
        
        
        temp_dir = tempfile.mkdtemp()
        try:
            
            if not top_module_name:
                top_module_name = VerilogChecker.extract_module_name(verilog_code2)
                if not top_module_name:
                    return False, "无法确定顶层模块名"
            
            
            rtl_fsm_dir = os.path.join(temp_dir, "rtl_fsm")
            os.makedirs(rtl_fsm_dir, exist_ok=True)
            
            
            ref_path = os.path.join(rtl_fsm_dir, "design_ref.v")
            imp_path = os.path.join(rtl_fsm_dir, "design_imp.v")
            
            with open(ref_path, 'w') as f:
                f.write(verilog_code2)
            
            with open(imp_path, 'w') as f:
                f.write(verilog_code1)
            
            
            eqc_path = os.path.join(temp_dir, "eqc.ys")
            with open(eqc_path, 'w') as f:
                
                eqc_template = 
                eqc_content = eqc_template.replace("MODULE_NAME", top_module_name)
                f.write(eqc_content)
            
            
            os.chdir(temp_dir)  
            result = subprocess.run(["yosys", "eqc.ys"], 
                                  capture_output=True,
                                  text=True,
                                  timeout=20)  
            
            
            if "Equivalence successfully proven!" in result.stdout:
                return True, "验证成功"
            else:
                return False, "验证失败"
        
        except Exception as e:
            return False, f"验证过程出错: {str(e)}"
        
        finally:
            
            import shutil
            shutil.rmtree(temp_dir)
