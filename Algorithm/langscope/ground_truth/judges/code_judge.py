"""
Code Completion Ground Truth Judge.

Evaluates code completion with syntax validation, test execution, and BLEU.
Supports sandboxed execution via Docker for security.
"""

import subprocess
import tempfile
import os
import shutil
import logging
from typing import Dict, List, Optional, Any
from langscope.ground_truth.judge import GroundTruthJudge, GroundTruthScore
from langscope.ground_truth.metrics import validate_syntax, bleu_score, exact_match

logger = logging.getLogger(__name__)


class CodeExecutionSandbox:
    """
    Sandboxed code execution environment using Docker.
    
    Provides secure execution of untrusted code with:
    - Memory limits
    - CPU limits  
    - Network isolation
    - Execution timeouts
    """
    
    # Default Docker image for Python execution
    PYTHON_IMAGE = "python:3.11-slim"
    
    def __init__(
        self,
        timeout: int = 10,
        memory_limit: str = "128m",
        cpu_limit: float = 0.5,
        use_docker: bool = True
    ):
        """
        Initialize sandbox.
        
        Args:
            timeout: Execution timeout in seconds
            memory_limit: Docker memory limit (e.g., "128m")
            cpu_limit: CPU limit (0.5 = half a core)
            use_docker: Whether to use Docker (falls back to subprocess if False)
        """
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.use_docker = use_docker and self._docker_available()
    
    def _docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def run_python(
        self,
        code: str,
        test_input: str = ""
    ) -> Dict[str, Any]:
        """
        Execute Python code in sandbox.
        
        Args:
            code: Python code to execute
            test_input: Input to provide to the code
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "return_code": int,
                "timed_out": bool
            }
        """
        if self.use_docker:
            return self._run_in_docker(code, test_input)
        else:
            return self._run_in_subprocess(code, test_input)
    
    def _run_in_docker(
        self,
        code: str,
        test_input: str = ""
    ) -> Dict[str, Any]:
        """Execute code in Docker container."""
        try:
            # Create temp directory for code
            with tempfile.TemporaryDirectory() as tmpdir:
                code_file = os.path.join(tmpdir, "code.py")
                
                # Write code to file
                with open(code_file, "w") as f:
                    f.write(code)
                
                # Build docker command
                docker_cmd = [
                    "docker", "run",
                    "--rm",  # Remove container after execution
                    "--network=none",  # No network access
                    f"--memory={self.memory_limit}",
                    f"--cpus={self.cpu_limit}",
                    "--read-only",  # Read-only filesystem
                    "--tmpfs=/tmp:rw,size=10m",  # Small writable /tmp
                    "-v", f"{tmpdir}:/code:ro",  # Mount code read-only
                    "-w", "/code",
                    self.PYTHON_IMAGE,
                    "python", "code.py"
                ]
                
                # Execute
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    input=test_input
                )
                
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode,
                    "timed_out": False
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Execution timed out",
                "return_code": -1,
                "timed_out": True
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "return_code": -1,
                "timed_out": False
            }
    
    def _run_in_subprocess(
        self,
        code: str,
        test_input: str = ""
    ) -> Dict[str, Any]:
        """
        Execute code in subprocess (less secure, for development only).
        
        WARNING: This should only be used when Docker is not available.
        Not recommended for production with untrusted code.
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                temp_path = f.name
            
            try:
                result = subprocess.run(
                    ["python", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    input=test_input
                )
                
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode,
                    "timed_out": False
                }
            finally:
                os.unlink(temp_path)
                
        except subprocess.TimeoutExpired:
            if 'temp_path' in locals():
                os.unlink(temp_path)
            return {
                "success": False,
                "output": "",
                "error": "Execution timed out",
                "return_code": -1,
                "timed_out": True
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "return_code": -1,
                "timed_out": False
            }


class CodeCompletionJudge(GroundTruthJudge):
    """
    Specialized judge for code completion evaluation.
    
    Metrics:
    - syntax_valid: Code has valid syntax
    - tests_pass: Percentage of test cases passing
    - exact_match: Code matches reference exactly
    - bleu: BLEU score against reference
    
    Uses sandboxed execution for running test cases securely.
    """
    
    def __init__(
        self,
        execution_timeout: int = 10,
        allow_execution: bool = True,
        use_docker: bool = True,
        memory_limit: str = "128m",
        **kwargs
    ):
        """
        Initialize code judge.
        
        Args:
            execution_timeout: Timeout for test execution (seconds)
            allow_execution: Whether to allow code execution
            use_docker: Whether to use Docker for sandboxing
            memory_limit: Docker memory limit
        """
        super().__init__(domain="code_completion", **kwargs)
        self.execution_timeout = execution_timeout
        self.allow_execution = allow_execution
        
        # Initialize sandbox
        self.sandbox = CodeExecutionSandbox(
            timeout=execution_timeout,
            memory_limit=memory_limit,
            use_docker=use_docker
        )
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute code completion metrics."""
        # Get expected code
        if isinstance(ground_truth, dict):
            expected = ground_truth.get("expected_code", str(ground_truth))
        else:
            expected = str(ground_truth)
        
        language = sample.get("language", "python")
        test_cases = sample.get("test_cases", [])
        
        # Syntax validation
        syntax_valid = validate_syntax(response, language)
        
        # Exact match
        exact = exact_match(
            self._normalize_code(response),
            self._normalize_code(expected)
        )
        
        # BLEU score
        bleu = bleu_score(response, [expected])
        
        metrics = {
            "syntax_valid": syntax_valid,
            "exact_match": exact,
            "bleu": bleu,
        }
        
        # Run tests if allowed and syntax is valid
        if self.allow_execution and syntax_valid == 1.0 and test_cases:
            tests_pass = self._run_tests(response, test_cases, language)
            metrics["tests_pass"] = tests_pass
        else:
            # Estimate based on other metrics
            metrics["tests_pass"] = (exact + syntax_valid + bleu) / 3
        
        return metrics
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove comments
        lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _run_tests(
        self,
        code: str,
        test_cases: List[Dict],
        language: str
    ) -> float:
        """
        Run test cases against generated code using sandboxed execution.
        
        Uses Docker-based sandboxing for security with:
        - Memory limits
        - CPU limits
        - Network isolation
        - Execution timeouts
        
        Args:
            code: Generated code
            test_cases: List of {input, expected} or {function_call, expected} dicts
            language: Programming language
        
        Returns:
            Percentage of tests passing (0.0-1.0)
        """
        if not self.allow_execution:
            logger.debug("Code execution disabled, returning 0.5")
            return 0.5  # Unknown
        
        if language.lower() not in ("python", "py", "python3"):
            logger.debug(f"Language {language} not supported, returning 0.5")
            return 0.5  # Only Python supported currently
        
        if not test_cases:
            return 0.0
        
        passed = 0
        total = len(test_cases)
        
        for i, test in enumerate(test_cases):
            try:
                result = self._run_single_test(code, test)
                if result:
                    passed += 1
            except Exception as e:
                logger.warning(f"Test {i+1}/{total} failed with error: {e}")
                continue
        
        return passed / total if total > 0 else 0.0
    
    def _run_single_test(
        self,
        code: str,
        test: Dict
    ) -> bool:
        """
        Run a single test case.
        
        Supports multiple test formats:
        1. {"input": "func(1, 2)", "expected": "3"}  - Function call style
        2. {"input": "1 2", "expected": "3", "stdin": True}  - Stdin style
        3. {"assert": "func(1, 2) == 3"}  - Assert style
        
        Args:
            code: Generated code
            test: Test case dict
        
        Returns:
            True if test passed
        """
        # Determine test format and build test script
        if "assert" in test:
            # Assert-style test
            assertion = test["assert"]
            test_script = f"""
{code}

# Test assertion
try:
    assert {assertion}
    print("PASS")
except AssertionError:
    print("FAIL")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            expected_output = "PASS"
            stdin_input = ""
            
        elif test.get("stdin"):
            # Stdin-style test
            test_script = code
            stdin_input = test.get("input", "")
            expected_output = test.get("expected", "")
            
        else:
            # Function call style (default)
            test_input = test.get("input", test.get("function_call", ""))
            expected = test.get("expected", "")
            
            test_script = f"""
{code}

# Test execution
try:
    result = {test_input}
    print(result)
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            stdin_input = ""
            expected_output = str(expected)
        
        # Execute in sandbox
        result = self.sandbox.run_python(test_script, stdin_input)
        
        if result["timed_out"]:
            logger.warning("Test timed out")
            return False
        
        if not result["success"] and "ERROR:" not in result["output"]:
            logger.debug(f"Test execution failed: {result['error']}")
            return False
        
        # Compare output
        actual_output = result["output"].strip()
        expected_output = expected_output.strip()
        
        # Handle different comparison modes
        comparison_mode = test.get("comparison", "exact")
        
        if comparison_mode == "exact":
            return actual_output == expected_output
        elif comparison_mode == "contains":
            return expected_output in actual_output
        elif comparison_mode == "startswith":
            return actual_output.startswith(expected_output)
        elif comparison_mode == "numeric":
            # Numeric comparison with tolerance
            try:
                actual_val = float(actual_output)
                expected_val = float(expected_output)
                tolerance = test.get("tolerance", 1e-6)
                return abs(actual_val - expected_val) <= tolerance
            except ValueError:
                return False
        else:
            return actual_output == expected_output
    
    def get_language_breakdown(
        self,
        match_results: List[Dict],
        model_id: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Get metrics breakdown by programming language.
        
        Args:
            match_results: List of match results
            model_id: Model identifier
        
        Returns:
            {language: {avg_tests_pass, avg_bleu, sample_count}}
        """
        stats: Dict[str, Dict[str, Any]] = {}
        
        for result in match_results:
            scores = result.get("scores", {}).get(model_id, {})
            metadata = result.get("sample_metadata", {})
            
            language = metadata.get("language", "python")
            metrics = scores.get("metrics", {})
            
            if language not in stats:
                stats[language] = {
                    "tests_pass_sum": 0.0,
                    "bleu_sum": 0.0,
                    "syntax_valid_sum": 0.0,
                    "count": 0,
                }
            
            stats[language]["tests_pass_sum"] += metrics.get("tests_pass", 0.0)
            stats[language]["bleu_sum"] += metrics.get("bleu", 0.0)
            stats[language]["syntax_valid_sum"] += metrics.get("syntax_valid", 0.0)
            stats[language]["count"] += 1
        
        # Compute averages
        breakdown = {}
        for lang, data in stats.items():
            count = data["count"]
            breakdown[lang] = {
                "avg_tests_pass": data["tests_pass_sum"] / count if count > 0 else 0.0,
                "avg_bleu": data["bleu_sum"] / count if count > 0 else 0.0,
                "avg_syntax_valid": data["syntax_valid_sum"] / count if count > 0 else 0.0,
                "sample_count": count,
            }
        
        return breakdown

