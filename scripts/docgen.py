#!/usr/bin/env python3
# docgen.py
"""
Python Code Documentation Generator
Erstellt automatisch kompakte Markdown-Dokumentation aus Python-Dateien
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import argparse
from datetime import datetime
import re

class PythonDocumentationExtractor:
    """Extrahiert strukturierte Informationen aus Python-Code"""
    
    def __init__(self):
        self.current_file = None
        
    def extract_docstring(self, node) -> Optional[str]:
        """Extrahiert Docstring aus einem AST-Node"""
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and 
            node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value.strip()
        return None
    
    def get_function_signature(self, func_node: ast.FunctionDef) -> str:
        """Erstellt Funktionssignatur aus AST-Node"""
        args = []
        
        # Normale Argumente
        for arg in func_node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # *args
        if func_node.args.vararg:
            vararg_str = f"*{func_node.args.vararg.arg}"
            if func_node.args.vararg.annotation:
                vararg_str += f": {ast.unparse(func_node.args.vararg.annotation)}"
            args.append(vararg_str)
        
        # **kwargs
        if func_node.args.kwarg:
            kwarg_str = f"**{func_node.args.kwarg.arg}"
            if func_node.args.kwarg.annotation:
                kwarg_str += f": {ast.unparse(func_node.args.kwarg.annotation)}"
            args.append(kwarg_str)
        
        # Defaults
        defaults = func_node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            num_args = len(func_node.args.args)
            for i, default in enumerate(defaults):
                arg_index = num_args - num_defaults + i
                if arg_index < len(args):
                    try:
                        default_value = ast.unparse(default)
                        args[arg_index] += f" = {default_value}"
                    except:
                        args[arg_index] += " = <default>"
        
        signature = f"{func_node.name}({', '.join(args)})"
        
        # Rückgabe-Typ
        if func_node.returns:
            try:
                return_type = ast.unparse(func_node.returns)
                signature += f" -> {return_type}"
            except:
                signature += " -> <return_type>"
        
        return signature
    
    def get_function_description(self, func_node: ast.FunctionDef) -> str:
        """Erstellt kurze Beschreibung der Funktionalität"""
        docstring = self.extract_docstring(func_node)
        
        if docstring:
            # Erste Zeile des Docstrings
            first_line = docstring.split('\n')[0].strip()
            if first_line:
                return first_line
        
        # Fallback: Analysiere erste bedeutsame Statements
        meaningful_lines = []
        for node in func_node.body[:3]:  # Erste 3 Statements
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # Skip docstrings
            try:
                line = ast.unparse(node).strip()
                if line and not line.startswith('#'):
                    meaningful_lines.append(line)
            except:
                continue
        
        if meaningful_lines:
            return f"Führt aus: {meaningful_lines[0][:50]}..."
        
        return "Keine Beschreibung verfügbar"
    
    def extract_imports(self, tree: ast.Module) -> Dict[str, List[str]]:
        """Extrahiert alle Imports"""
        imports = {
            'standard': [],
            'third_party': [],
            'local': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    import_str = f"import {module_name}"
                    if alias.asname:
                        import_str += f" as {alias.asname}"
                    
                    # Kategorisierung
                    if module_name.startswith('app.') or module_name.startswith('.'):
                        imports['local'].append(import_str)
                    elif module_name in ['os', 'sys', 'json', 'datetime', 'argparse', 're', 'pathlib']:
                        imports['standard'].append(import_str)
                    else:
                        imports['third_party'].append(import_str)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module
                    items = [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                            for alias in node.names]
                    import_str = f"from {module_name} import {', '.join(items)}"
                    
                    # Kategorisierung
                    if module_name.startswith('app.') or node.level > 0:
                        imports['local'].append(import_str)
                    elif module_name in ['os', 'sys', 'json', 'datetime', 'argparse', 're', 'pathlib', 'typing']:
                        imports['standard'].append(import_str)
                    else:
                        imports['third_party'].append(import_str)
        
        return imports
    
    def extract_classes(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extrahiert Klassen-Informationen"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'bases': [ast.unparse(base) for base in node.bases] if node.bases else [],
                    'docstring': self.extract_docstring(node),
                    'methods': []
                }
                
                # Methoden extrahieren
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'signature': self.get_function_signature(item),
                            'description': self.get_function_description(item),
                            'is_private': item.name.startswith('_'),
                            'is_property': any(isinstance(dec, ast.Name) and dec.id == 'property' 
                                             for dec in item.decorator_list)
                        }
                        class_info['methods'].append(method_info)
                
                classes.append(class_info)
        
        return classes
    
    def extract_functions(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extrahiert Top-Level Funktionen"""
        functions = []
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'signature': self.get_function_signature(node),
                    'description': self.get_function_description(node),
                    'is_private': node.name.startswith('_'),
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list] if node.decorator_list else []
                }
                functions.append(func_info)
        
        return functions
    
    def extract_constants(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extrahiert Top-Level Konstanten und Variablen"""
        constants = []
        
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        const_info = {
                            'name': target.id,
                            'type': 'variable',
                            'is_constant': target.id.isupper()
                        }
                        try:
                            const_info['value'] = ast.unparse(node.value)[:50]
                        except:
                            const_info['value'] = '<complex_value>'
                        constants.append(const_info)
        
        return constants
    
    def analyze_file(self, filepath: Path) -> Dict[str, Any]:
        """Analysiert eine Python-Datei und extrahiert alle Informationen"""
        self.current_file = filepath
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'filepath': str(filepath),
                'filename': filepath.name,
                'module_docstring': self.extract_docstring(tree),
                'imports': self.extract_imports(tree),
                'constants': self.extract_constants(tree),
                'functions': self.extract_functions(tree),
                'classes': self.extract_classes(tree),
                'line_count': len(content.splitlines()),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            return {
                'filepath': str(filepath),
                'filename': filepath.name,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }

class MarkdownGenerator:
    """Generiert Markdown-Dokumentation aus extrahierten Informationen"""
    
    def generate_documentation(self, analysis: Dict[str, Any]) -> str:
        """Generiert komplette Markdown-Dokumentation"""
        md = []
        
        # Header
        filename = analysis['filename']
        md.append(f"# {filename}")
        md.append("")
        
        # Fehlerbehandlung
        if 'error' in analysis:
            md.append(f"❌ **Fehler bei der Analyse:** {analysis['error']}")
            return "\n".join(md)
        
        # Basis-Informationen
        md.append("## 📋 Übersicht")
        md.append("")
        md.append(f"- **Datei:** `{analysis['filename']}`")
        md.append(f"- **Zeilen:** {analysis['line_count']:,}")
        md.append(f"- **Analysiert:** {analysis['analysis_timestamp'][:19]}")
        md.append("")
        
        # Modul-Docstring
        if analysis.get('module_docstring'):
            md.append("### Beschreibung")
            md.append("")
            md.append(f"```")
            md.append(analysis['module_docstring'])
            md.append(f"```")
            md.append("")
        
        # Imports
        self._add_imports_section(md, analysis['imports'])
        
        # Konstanten
        if analysis['constants']:
            self._add_constants_section(md, analysis['constants'])
        
        # Funktionen
        if analysis['functions']:
            self._add_functions_section(md, analysis['functions'])
        
        # Klassen
        if analysis['classes']:
            self._add_classes_section(md, analysis['classes'])
        
        return "\n".join(md)
    
    def _add_imports_section(self, md: List[str], imports: Dict[str, List[str]]):
        """Fügt Imports-Sektion hinzu"""
        md.append("## 📦 Imports")
        md.append("")
        
        if imports['standard']:
            md.append("### Standard Library")
            for imp in sorted(imports['standard']):
                md.append(f"- `{imp}`")
            md.append("")
        
        if imports['third_party']:
            md.append("### Third Party")
            for imp in sorted(imports['third_party']):
                md.append(f"- `{imp}`")
            md.append("")
        
        if imports['local']:
            md.append("### Local/App")
            for imp in sorted(imports['local']):
                md.append(f"- `{imp}`")
            md.append("")
    
    def _add_constants_section(self, md: List[str], constants: List[Dict[str, Any]]):
        """Fügt Konstanten-Sektion hinzu"""
        md.append("## 🔧 Konstanten & Variablen")
        md.append("")
        
        for const in constants:
            icon = "🔒" if const['is_constant'] else "📝"
            md.append(f"- {icon} **`{const['name']}`** = `{const['value']}`")
        md.append("")
    
    def _add_functions_section(self, md: List[str], functions: List[Dict[str, Any]]):
        """Fügt Funktionen-Sektion hinzu"""
        md.append("## ⚙️ Funktionen")
        md.append("")
        
        # Öffentliche Funktionen zuerst
        public_funcs = [f for f in functions if not f['is_private']]
        private_funcs = [f for f in functions if f['is_private']]
        
        if public_funcs:
            md.append("### Öffentliche Funktionen")
            md.append("")
            for func in public_funcs:
                self._add_function_detail(md, func)
        
        if private_funcs:
            md.append("### Private Funktionen")
            md.append("")
            for func in private_funcs:
                self._add_function_detail(md, func)
    
    def _add_function_detail(self, md: List[str], func: Dict[str, Any]):
        """Fügt Details einer Funktion hinzu"""
        decorators = ""
        if func['decorators']:
            decorators = f" {', '.join([f'@{d}' for d in func['decorators']])}"
        
        md.append(f"#### `{func['signature']}`{decorators}")
        md.append("")
        md.append(f"**Beschreibung:** {func['description']}")
        md.append("")
    
    def _add_classes_section(self, md: List[str], classes: List[Dict[str, Any]]):
        """Fügt Klassen-Sektion hinzu"""
        md.append("## 🏗️ Klassen")
        md.append("")
        
        for cls in classes:
            md.append(f"### `{cls['name']}`")
            if cls['bases']:
                md.append(f"**Erbt von:** `{', '.join(cls['bases'])}`")
            md.append("")
            
            if cls['docstring']:
                md.append(f"**Beschreibung:** {cls['docstring'].split('.')[0]}")
                md.append("")
            
            if cls['methods']:
                md.append("#### Methoden")
                md.append("")
                
                # Kategorisiere Methoden
                public_methods = [m for m in cls['methods'] if not m['is_private']]
                private_methods = [m for m in cls['methods'] if m['is_private']]
                
                for method in public_methods:
                    icon = "🔧" if method['is_property'] else "⚙️"
                    md.append(f"- {icon} **`{method['signature']}`**")
                    md.append(f"  - {method['description']}")
                    md.append("")
                
                if private_methods and len(private_methods) <= 3:  # Nur wenige private Methoden zeigen
                    md.append("**Private Methoden:**")
                    for method in private_methods:
                        md.append(f"- `{method['name']}()` - {method['description'][:30]}...")
                    md.append("")

def process_directory(source_dir: Path, output_dir: Path, file_pattern: str = "*.py"):
    """Verarbeitet alle Python-Dateien in einem Verzeichnis"""
    
    extractor = PythonDocumentationExtractor()
    generator = MarkdownGenerator()
    
    # Output-Verzeichnis erstellen
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Alle Python-Dateien finden
    python_files = list(source_dir.glob(file_pattern))
    
    if not python_files:
        print(f"❌ Keine Python-Dateien in {source_dir} gefunden")
        return
    
    print(f"📁 Verarbeite {len(python_files)} Dateien aus {source_dir}")
    print(f"📝 Ausgabe-Verzeichnis: {output_dir}")
    print()
    
    processed = 0
    errors = 0
    
    for py_file in python_files:
        try:
            print(f"🔍 Analysiere: {py_file.name}")
            
            # Analysiere Datei
            analysis = extractor.analyze_file(py_file)
            
            # Generiere Dokumentation
            documentation = generator.generate_documentation(analysis)
            
            # Speichere Markdown-Datei
            output_file = output_dir / f"{py_file.name}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(documentation)
            
            # Statistiken
            original_size = py_file.stat().st_size
            doc_size = output_file.stat().st_size
            reduction = (1 - doc_size / original_size) * 100 if original_size > 0 else 0
            
            print(f"  ✅ {py_file.name} -> {output_file.name}")
            print(f"     Original: {original_size:,} Bytes | Docs: {doc_size:,} Bytes | Reduktion: {reduction:.1f}%")
            
            processed += 1
            
        except Exception as e:
            print(f"  ❌ Fehler bei {py_file.name}: {str(e)}")
            errors += 1
    
    print()
    print(f"🎯 **Zusammenfassung:**")
    print(f"  - Verarbeitet: {processed}")
    print(f"  - Fehler: {errors}")
    print(f"  - Dokumentation gespeichert in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Python Code Documentation Generator - Erstellt kompakte Markdown-Dokumentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
BEISPIELE:
  python doc_generator.py                           # Verarbeitet ./app -> ./descr
  python doc_generator.py --source ./src           # Andere Quell-Verzeichnis
  python doc_generator.py --output ./documentation # Andere Ausgabe-Verzeichnis
  python doc_generator.py --pattern "*.py"         # Nur bestimmte Dateien
        """
    )
    
    parser.add_argument('--source', '-s', type=str, default='app',
                        help='Quell-Verzeichnis mit Python-Dateien (Standard: app)')
    parser.add_argument('--output', '-o', type=str, default='descr',
                        help='Ausgabe-Verzeichnis für Dokumentation (Standard: descr)')
    parser.add_argument('--pattern', '-p', type=str, default='*.py',
                        help='Datei-Pattern für Python-Dateien (Standard: *.py)')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_dir.exists():
        print(f"❌ Quell-Verzeichnis nicht gefunden: {source_dir}")
        sys.exit(1)
    
    print("🐍 Python Code Documentation Generator")
    print("=" * 50)
    
    try:
        process_directory(source_dir, output_dir, args.pattern)
        print("\n🎉 Dokumentation erfolgreich erstellt!")
        
    except KeyboardInterrupt:
        print("\n👋 Abgebrochen durch Benutzer")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()