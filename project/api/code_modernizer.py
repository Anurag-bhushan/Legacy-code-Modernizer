import ast
import astor
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import google.generativeai as genai  # Changed from openai to google.generativeai
import re
import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib
import subprocess

@dataclass
class ModernizationResult:
    """Stores the results of code modernization"""
    original_code: str
    modernized_code: str
    changes_made: List[str]
    success: bool
    language: str
    frameworks_detected: List[str]
    error_message: Optional[str] = None

@dataclass
class BatchModernizationResult:
    """Stores results from batch processing multiple files"""
    successful_files: Dict[str, ModernizationResult]
    failed_files: Dict[str, str]  # filename -> error message
    total_files: int
    success_rate: float

class EnhancedAIModernizer:
    def __init__(self, api_key: str):
        """Initialize the Enhanced AI Code Modernizer"""
        # Set up Gemini client with API key
        genai.configure(api_key=api_key)
        self.language_prompts = self._initialize_language_prompts()
        self.framework_patterns = self._initialize_framework_patterns()
        self.language_extensions = self._initialize_language_extensions()
        
    def _initialize_language_prompts(self) -> Dict[str, str]:
        """Initialize comprehensive language-specific modernization prompts"""
        return {
            "python": """
            You are an expert Python modernizer. Analyze the provided code and:
            1. Update to Python 3.9+ features including:
               - Type hints with generics
               - Pattern matching
               - Dictionary unions
               - New string methods
            2. Convert to modern frameworks/libraries
            3. Implement current best practices
            4. Add comprehensive documentation
            5. Preserve exact functionality
            6. Add error handling and logging
            
            List all changes as comments at the end.
            """,
            
            "javascript": """
            You are an expert JavaScript modernizer. Transform the code to:
            1. Use latest ECMAScript features:
               - Optional chaining
               - Nullish coalescing
               - Private class fields
               - Top-level await
            2. Convert to TypeScript where beneficial
            3. Use modern framework patterns
            4. Implement current best practices
            5. Add proper error handling
            6. Include comprehensive JSDoc
            
            List all changes as comments at the end.
            """,
            
            "java": """
            You are an expert Java modernizer. Update the code to:
            1. Use latest Java features:
               - Records
               - Pattern matching
               - Text blocks
               - Switch expressions
            2. Apply modern framework patterns
            3. Use current best practices
            4. Add comprehensive JavaDoc
            5. Implement proper exception handling
            6. Use modern APIs
            
            List all changes as comments at the end.
            """,
            
            "csharp": """
            You are an expert C# modernizer. Transform the code to:
            1. Use latest C# features:
               - Record types
               - Pattern matching
               - Top-level statements
               - File-scoped namespaces
            2. Apply .NET Core best practices
            3. Use modern framework patterns
            4. Add comprehensive documentation
            5. Implement proper error handling
            6. Use current APIs
            
            List all changes as comments at the end.
            """
        }

    def _initialize_framework_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize patterns to detect frameworks and libraries"""
        return {
            "python": {
                "django": [r"from django", r"django\."],
                "flask": [r"from flask", r"Flask\("],
                "fastapi": [r"from fastapi", r"FastAPI\("],
                "pandas": [r"import pandas", r"pd\."],
                "tensorflow": [r"import tensorflow", r"tf\."],
                "pytorch": [r"import torch", r"torch\."]
            },
            "javascript": {
                "react": [r"import React", r"React\.", r"useState"],
                "vue": [r"import Vue", r"new Vue"],
                "angular": [r"@Component", r"@Injectable"],
                "express": [r"express\(\)", r"app\.get\("],
                "next": [r"from 'next'", r"NextPage"]
            },
            "java": {
                "spring": [r"@SpringBootApplication", r"@Autowired"],
                "hibernate": [r"@Entity", r"@Table"],
                "junit": [r"@Test", r"Assert\."],
                "jackson": [r"@JsonProperty", r"ObjectMapper"]
            },
            "csharp": {
                "aspnet": [r"Microsoft\.AspNetCore", r"IActionResult"],
                "ef": [r"Microsoft\.EntityFrameworkCore", r"DbContext"],
                "xamarin": [r"Xamarin\.", r"ContentPage"],
                "unity": [r"UnityEngine", r"MonoBehaviour"]
            }
        }

    def _initialize_language_extensions(self) -> Dict[str, Set[str]]:
        """Initialize file extensions for supported languages"""
        return {
            "python": {".py", ".pyw", ".pyx"},
            "javascript": {".js", ".jsx", ".ts", ".tsx"},
            "java": {".java"},
            "csharp": {".cs"},
            "ruby": {".rb"},
            "php": {".php"},
            "go": {".go"},
            "rust": {".rs"}
        }

    def detect_language_and_frameworks(self, code: str, file_path: Optional[str] = None) -> Tuple[str, List[str]]:
        """Detect programming language and frameworks used in the code"""
        # Try to detect from file extension first
        if file_path:
            ext = Path(file_path).suffix.lower()
            for lang, extensions in self.language_extensions.items():
                if ext in extensions:
                    language = lang
                    break
        else:
            language = self._detect_language_from_content(code)

        # Detect frameworks
        frameworks = []
        if language in self.framework_patterns:
            for framework, patterns in self.framework_patterns[language].items():
                if any(re.search(pattern, code) for pattern in patterns):
                    frameworks.append(framework)

        return language, frameworks

    def _detect_language_from_content(self, code: str) -> str:
        """Detect programming language from code content"""
        patterns = {
            "python": (r"def\s+\w+\s*\(", r"import\s+\w+", r"from\s+\w+\s+import"),
            "javascript": (r"function\s+\w+\s*\(", r"const\s+\w+\s*=", r"let\s+\w+\s*="),
            "java": (r"public\s+class", r"private\s+void", r"System\.out\.println"),
            "csharp": (r"namespace\s+\w+", r"public\s+class", r"using\s+System"),
            "ruby": (r"def\s+\w+\s*", r"require\s+'", r"module\s+\w+"),
            "php": (r"<\?php", r"function\s+\w+\s*\(", r"namespace\s+\w+"),
            "go": (r"package\s+main", r"func\s+\w+\s*\(", r"import\s+\("),
            "rust": (r"fn\s+main", r"use\s+std", r"pub\s+struct")
        }

        scores = {lang: 0 for lang in patterns}
        for lang, patterns_list in patterns.items():
            for pattern in patterns_list:
                if re.search(pattern, code):
                    scores[lang] += 1

        return max(scores.items(), key=lambda x: x[1])[0]

    def validate_code(self, code: str, language: str) -> Tuple[bool, Optional[str]]:
        """Validate code syntax for various languages"""
        validators = {
            "python": self._validate_python,
            "javascript": self._validate_javascript,
            "java": self._validate_java,
            "csharp": self._validate_csharp
        }

        validator = validators.get(language)
        if validator:
            return validator(code)
        return True, None  # Skip validation for unsupported languages

    def _validate_python(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code"""
        try:
            ast.parse(code)
            return True, None
        except Exception as e:
            return False, str(e)

    def _validate_javascript(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate JavaScript code using Node.js"""
        try:
            with open('temp.js', 'w') as f:
                f.write(code)
            result = subprocess.run(['node', '--check', 'temp.js'], 
                                 capture_output=True, text=True)
            os.remove('temp.js')
            if result.returncode == 0:
                return True, None
            return False, result.stderr
        except Exception as e:
            return False, str(e)

    def _validate_java(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Java code"""
        try:
            # Simple syntax check - not comprehensive
            if "class" not in code:
                return False, "No class definition found"
            return True, None
        except Exception as e:
            return False, str(e)

    def _validate_csharp(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate C# code"""
        try:
            # Simple syntax check - not comprehensive
            if "namespace" not in code and "class" not in code:
                return False, "No namespace or class definition found"
            return True, None
        except Exception as e:
            return False, str(e)

    def modernize_code(self, code: str, language: Optional[str] = None) -> ModernizationResult:
        """Modernize single file of code using AI"""
        try:
            # Detect language and frameworks
            detected_language, frameworks = self.detect_language_and_frameworks(code)
            language = language or detected_language

            # Validate input code
            is_valid, error = self.validate_code(code, language)
            if not is_valid:
                return ModernizationResult(
                    original_code=code,
                    modernized_code=code,
                    changes_made=[],
                    success=False,
                    language=language,
                    frameworks_detected=frameworks,
                    error_message=error
                )

            # Get language-specific prompt
            base_prompt = self.language_prompts.get(language, self.language_prompts["python"])
            
            # Add framework-specific instructions
            framework_instructions = self._get_framework_instructions(frameworks, language)
            prompt = f"{base_prompt}\n{framework_instructions}"

            # Call Gemini API for modernization instead of OpenAI
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

            response = model.generate_content(
                [prompt, f"Code to modernize:\n```{language}\n{code}\n```"]
            )
            
            # Extract modernized code and changes from Gemini response
            ai_response = response.text
            modernized_code, changes = self._parse_ai_response(ai_response)
            
            # Validate output code
            is_valid, error = self.validate_code(modernized_code, language)
            if not is_valid:
                return ModernizationResult(
                    original_code=code,
                    modernized_code=code,
                    changes_made=[f"AI output validation failed: {error}"],
                    success=False,
                    language=language,
                    frameworks_detected=frameworks,
                    error_message=error
                )

            return ModernizationResult(
                original_code=code,
                modernized_code=modernized_code,
                changes_made=changes,
                success=True,
                language=language,
                frameworks_detected=frameworks
            )

        except Exception as e:
            return ModernizationResult(
                original_code=code,
                modernized_code=code,
                changes_made=[],
                success=False,
                language=language or "unknown",
                frameworks_detected=[],
                error_message=f"Modernization error: {str(e)}"
            )

    def _parse_ai_response(self, response: str) -> Tuple[str, List[str]]:
        """Parse the AI response to extract code and changes"""
        # Extract code block
        code_pattern = r"```(?:\w+)?\n([\s\S]+?)\n```"
        code_match = re.search(code_pattern, response)
        
        # Default values
        modernized_code = ""
        changes = []
        
        if code_match:
            modernized_code = code_match.group(1)
            
            # Look for changes section - various patterns to match different formats
            changes_patterns = [
                r"# Changes made:([\s\S]+)$",
                r"# Changes:([\s\S]+)$",
                r"Changes made:([\s\S]+)$",
                r"Changes:([\s\S]+)$"
            ]
            
            for pattern in changes_patterns:
                changes_match = re.search(pattern, response)
                if changes_match:
                    # Extract and clean up changes
                    changes_text = changes_match.group(1)
                    changes = [line.strip().lstrip('-').strip() 
                               for line in changes_text.split('\n') 
                               if line.strip() and not line.strip().startswith('#')]
                    break
            
            # If no changes section found, use AI response after code block
            if not changes:
                post_code = response.split("```")[-1].strip()
                if post_code:
                    # Split by lines and clean up
                    changes = [line.strip().lstrip('-').strip() 
                               for line in post_code.split('\n') 
                               if line.strip() and not line.strip().startswith('#')]
        else:
            # If no code block found, use the entire response
            modernized_code = response
        
        return modernized_code, changes

    def _get_framework_instructions(self, frameworks: List[str], language: str) -> str:
        """Get framework-specific modernization instructions"""
        instructions = []
        framework_instructions = {
            "python": {
                "django": "Update to latest Django patterns and best practices",
                "flask": "Use latest Flask features and patterns",
                "fastapi": "Implement modern FastAPI patterns",
                "pandas": "Use latest Pandas features and best practices",
                "tensorflow": "Update to TensorFlow 2.x patterns",
                "pytorch": "Implement PyTorch best practices"
            },
            "javascript": {
                "react": "Use React hooks and modern patterns",
                "vue": "Update to Vue 3 composition API",
                "angular": "Implement latest Angular practices",
                "express": "Use modern Express.js patterns",
                "next": "Implement Next.js best practices"
            }
            # Add more frameworks as needed
        }

        if language in framework_instructions:
            for framework in frameworks:
                if framework in framework_instructions[language]:
                    instructions.append(framework_instructions[language][framework])

        return "\n".join(instructions)

    def batch_modernize(self, directory: str, output_directory: str = None, 
                       max_workers: int = 5) -> BatchModernizationResult:
        """Modernize multiple files in a directory"""
        if not output_directory:
            output_directory = directory + "_modernized"

        os.makedirs(output_directory, exist_ok=True)
        
        # Collect all files to process
        files_to_process = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                output_path = os.path.join(output_directory, relative_path)
                
                # Check if file extension is supported
                if any(file.endswith(ext) for exts in self.language_extensions.values() for ext in exts):
                    files_to_process.append((file_path, output_path))

        successful_files = {}
        failed_files = {}
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, src, dst): src 
                for src, dst in files_to_process
            }
            
            for future in future_to_file:
                src_file = future_to_file[future]
                try:
                    result = future.result()
                    if result.success:
                        successful_files[src_file] = result
                    else:
                        failed_files[src_file] = result.error_message
                except Exception as e:
                    failed_files[src_file] = str(e)

        # Calculate statistics
        total_files = len(files_to_process)
        success_rate = len(successful_files) / total_files if total_files > 0 else 0
        return BatchModernizationResult(
            successful_files=successful_files,
            failed_files=failed_files,
            total_files=total_files,
            success_rate=success_rate
        )

    def _process_single_file(self, src_path: str, dst_path: str) -> ModernizationResult:
        """Process a single file for batch modernization"""
        try:
            # Read source file
            with open(src_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # Modernize code
            result = self.modernize_code(code)

            if result.success:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Save modernized code
                with open(dst_path, 'w', encoding='utf-8') as f:
                    f.write(result.modernized_code)
                    f.write("\n\n# Modernization Changes:\n")
                    for change in result.changes_made:
                        f.write(f"# - {change}\n")

            return result

        except Exception as e:
            return ModernizationResult(
                original_code="",
                modernized_code="",
                changes_made=[],
                success=False,
                language="unknown",
                frameworks_detected=[],
                error_message=f"File processing error: {str(e)}"
            )

    def generate_modernization_report(self, batch_result: BatchModernizationResult, 
                                    output_path: str) -> None:
        """Generate a detailed report of the modernization process"""
        report = {
            "summary": {
                "total_files": batch_result.total_files,
                "successful_files": len(batch_result.successful_files),
                "failed_files": len(batch_result.failed_files),
                "success_rate": batch_result.success_rate
            },
            "successful_files": {},
            "failed_files": {}
        }

        # Add details for successful files
        for file_path, result in batch_result.successful_files.items():
            report["successful_files"][file_path] = {
                "language": result.language,
                "frameworks_detected": result.frameworks_detected,
                "changes_made": result.changes_made
            }

        # Add details for failed files
        for file_path, error in batch_result.failed_files.items():
            report["failed_files"][file_path] = {
                "error_message": error
            }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    def analyze_codebase(self, directory: str) -> Dict:
        """Analyze a codebase for modernization opportunities"""
        analysis = {
            "languages": {},
            "frameworks": {},
            "files_by_type": {},
            "modernization_opportunities": []
        }

        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Detect language and frameworks
                    language, frameworks = self.detect_language_and_frameworks(code, file_path)
                    
                    # Update statistics
                    analysis["languages"][language] = analysis["languages"].get(language, 0) + 1
                    for framework in frameworks:
                        analysis["frameworks"][framework] = analysis["frameworks"].get(framework, 0) + 1
                    
                    ext = Path(file).suffix
                    analysis["files_by_type"][ext] = analysis["files_by_type"].get(ext, 0) + 1
                    
                    # Identify modernization opportunities
                    opportunities = self._identify_modernization_opportunities(code, language, frameworks)
                    if opportunities:
                        analysis["modernization_opportunities"].append({
                            "file": file_path,
                            "opportunities": opportunities
                        })
                        
                except Exception as e:
                    print(f"Error analyzing {file_path}: {str(e)}")

        return analysis

    def _identify_modernization_opportunities(self, code: str, language: str, 
                                           frameworks: List[str]) -> List[str]:
        """Identify specific modernization opportunities in code"""
        opportunities = []
        
        # Language-specific patterns to check
        patterns = {
            "python": {
                "old_style_string_format": (r'%[sd]', "Use f-strings instead of %-formatting"),
                "no_type_hints": (r'def \w+\([^:]+\)', "Add type hints to function parameters"),
                "old_style_exception": (r'except:\s*$', "Use specific exception handling"),
                "print_statement": (r'print [^(]', "Use print() function"),
            },
            "javascript": {
                "var_usage": (r'\bvar\b', "Replace 'var' with 'const' or 'let'"),
                "old_function_syntax": (r'function\s+\w+', "Use arrow functions where appropriate"),
                "callback_hell": (r'}\)[.;]', "Consider using async/await"),
                "jquery_usage": (r'\$\(', "Consider using modern DOM APIs"),
            }
        }
        
        if language in patterns:
            for pattern_name, (pattern, suggestion) in patterns[language].items():
                if re.search(pattern, code):
                    opportunities.append(suggestion)
        
        return opportunities

def example_usage():
    """Example usage of the Enhanced AI Modernizer"""
    # Initialize modernizer with Gemini API key
    modernizer = EnhancedAIModernizer("your-gemini-api-key")
    
    # Example 1: Modernize a single file
    with open("legacy_code.py", "r") as f:
        code = f.read()
    result = modernizer.modernize_code(code)
    if result.success:
        print(f"Modernized code with {len(result.changes_made)} improvements")
    
    # Example 2: Batch modernize a directory
    batch_result = modernizer.batch_modernize(
        directory="legacy_project",
        output_directory="modern_project",
        max_workers=5
    )
    
    # Generate report
    modernizer.generate_modernization_report(
        batch_result,
        "modernization_report.json"
    )
    
    # Example 3: Analyze codebase
    analysis = modernizer.analyze_codebase("legacy_project")
    print(f"Found {len(analysis['modernization_opportunities'])} opportunities for modernization")

if __name__ == "__main__":
    example_usage()