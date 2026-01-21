"""
Tests for JavaScript documentation theme functionality.
This test ensures the modernized JavaScript still works correctly.
"""

import pytest
import os
import re
from pathlib import Path


def test_searchtools_js_modernization():
    """Test that searchtools.js has been properly modernized."""
    js_file_path = Path(__file__).parent.parent.parent / "doc" / "themes" / "scikit-learn-modern" / "static" / "js" / "searchtools.js"
    
    assert js_file_path.exists(), f"JavaScript file not found at {js_file_path}"
    
    with open(js_file_path, 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    # Test that no var declarations remain (should be replaced with const/let)
    var_pattern = r'\bvar\s+'
    var_matches = re.findall(var_pattern, js_content)
    assert len(var_matches) == 0, f"Found {len(var_matches)} 'var' declarations that should be replaced"
    
    # Test that no loose equality operators remain (should be ===)
    loose_eq_pattern = r'(?<!===?)==(?!=)'
    loose_eq_matches = re.findall(loose_eq_pattern, js_content)
    assert len(loose_eq_matches) == 0, f"Found {len(loose_eq_matches)} loose equality operators that should be ==="
    
    # Test that no incorrect triple equals remain (should be === not ====)
    triple_eq_pattern = r'===='
    triple_eq_matches = re.findall(triple_eq_pattern, js_content)
    assert len(triple_eq_matches) == 0, f"Found {len(triple_eq_matches)} incorrect triple equals that should be ==="
    
    # Test that no incorrect not equals remain (should be !== not !===)
    incorrect_not_eq_pattern = r'!==='
    incorrect_not_eq_matches = re.findall(incorrect_not_eq_pattern, js_content)
    assert len(incorrect_not_eq_matches) == 0, f"Found {len(incorrect_not_eq_matches)} incorrect not equals that should be !=="
    
    # Test that modern const/let are used
    const_let_pattern = r'\b(const|let)\s+'
    const_let_matches = re.findall(const_let_pattern, js_content)
    assert len(const_let_matches) > 0, "No const/let declarations found - modernization may be incomplete"
    
    # Test that strict equality is used
    strict_eq_pattern = r'==='
    strict_eq_matches = re.findall(strict_eq_pattern, js_content)
    assert len(strict_eq_matches) > 0, "No strict equality operators found - modernization may be incomplete"


def test_searchtools_js_functionality_preserved():
    """Test that essential JavaScript functionality is preserved."""
    js_file_path = Path(__file__).parent.parent.parent / "doc" / "themes" / "scikit-learn-modern" / "static" / "js" / "searchtools.js"
    
    with open(js_file_path, 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    # Test that key objects and functions are still present
    essential_elements = [
        'Scorer',           # Scoring object
        'Search',           # Main search object  
        'splitQuery',       # Query splitting function
        'htmlToText',      # HTML to text conversion
        'performSearch',     # Search performance
        'query',           # Query execution
        'performObjectSearch', # Object search
        'performTermsSearch'   # Terms search
    ]
    
    for element in essential_elements:
        assert element in js_content, f"Essential element '{element}' missing from JavaScript file"


def test_searchtools_js_syntax_validity():
    """Test that the JavaScript file has valid syntax."""
    js_file_path = Path(__file__).parent.parent.parent / "doc" / "themes" / "scikit-learn-modern" / "static" / "js" / "searchtools.js"
    
    with open(js_file_path, 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    # Basic syntax checks
    open_braces = js_content.count('{')
    close_braces = js_content.count('}')
    assert open_braces == close_braces, f"Mismatched braces: {open_braces} open, {close_braces} close"
    
    open_parens = js_content.count('(')
    close_parens = js_content.count(')')
    assert open_parens == close_parens, f"Mismatched parentheses: {open_parens} open, {close_parens} close"
    
    open_brackets = js_content.count('[')
    close_brackets = js_content.count(']')
    assert open_brackets == close_brackets, f"Mismatched brackets: {open_brackets} open, {close_brackets} close"


if __name__ == "__main__":
    pytest.main([__file__])
