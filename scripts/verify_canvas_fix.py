#!/usr/bin/env python3
"""
Verification script for GUI Canvas Image Fix (v4.1)

Tests:
1. _REINIT_CANVAS_JS is now directly executable (no function wrapper)
2. All required components are present
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Go up to project root
sys.path.insert(0, str(ROOT))

def verify_fix():
    """Verify the fix is correctly applied."""
    gui_file = ROOT / "scripts" / "gui_workbench.py"
    content = gui_file.read_text(encoding="utf-8")
    
    # Extract _REINIT_CANVAS_JS definition
    start = content.find("_REINIT_CANVAS_JS = ")
    if start == -1:
        print("❌ ERROR: Cannot find _REINIT_CANVAS_JS definition")
        return False
    
    # Find the string start and end
    quote_start = content.find('"""', start)
    quote_end = content.find('"""', quote_start + 3)
    
    if quote_start == -1 or quote_end == -1:
        print("❌ ERROR: Cannot parse _REINIT_CANVAS_JS string")
        return False
    
    js_code = content[quote_start + 3:quote_end].strip()
    
    print("=" * 70)
    print("Current _REINIT_CANVAS_JS content:")
    print("=" * 70)
    print(js_code)
    print("=" * 70)
    
    # Verify fixes
    starts_with_function = js_code.strip().startswith("function()")
    starts_with_settimeout = js_code.strip().startswith("setTimeout")
    has_setTimeout = "setTimeout" in js_code
    has_wb_reset = "window._wb_reset" in js_code
    has_delay = "200" in js_code
    has_type_check = 'typeof window._wb_reset' in js_code
    
    print("\nVerification Results:")
    print("-" * 70)
    
    all_passed = True
    
    if starts_with_function:
        print("❌ BROKEN: Code starts with 'function()' wrapper (not directly executable)")
        all_passed = False
    elif starts_with_settimeout:
        print("✓ FIXED: Code starts with 'setTimeout' (directly executable)")
    else:
        print("⚠ WARNING: Code doesn't start with expected keyword")
        all_passed = False
    
    if has_setTimeout:
        print("✓ REQUIRED: Has setTimeout")
    else:
        print("❌ REQUIRED: Missing setTimeout")
        all_passed = False
    
    if has_wb_reset:
        print("✓ REQUIRED: Calls window._wb_reset()")
    else:
        print("❌ REQUIRED: Missing window._wb_reset() call")
        all_passed = False
    
    if has_delay:
        print("✓ REQUIRED: Has 200ms delay")
    else:
        print("❌ REQUIRED: Missing 200ms delay")
        all_passed = False
    
    if has_type_check:
        print("✓ REQUIRED: Type check before calling")
    else:
        print("❌ REQUIRED: Missing type check")
        all_passed = False
    
    print("-" * 70)
    
    # Verify other key components
    print("\nOther component checks:")
    print("-" * 70)
    
    component_checks = {
        "loadImageFromSource() function exists": "function loadImageFromSource()" in content,
        "window._wb_reset() function exposed": "window._wb_reset = function()" in content,
        "Canvas data-img-src attribute in HTML": 'data-img-src=' in content,
        "_build_canvas_page() uses data_url": "img_attr = f' data-img-src=" in content,
        "parse_file() generates data URLs": "data_url = f\"data:{mime" in content,
        ".then(js=_REINIT_CANVAS_JS) in file_input.change()": ".then(\n            js=_REINIT_CANVAS_JS" in content,
    }
    
    for check_name, condition in component_checks.items():
        status = "✓" if condition else "❌"
        print(f"{status} {check_name}")
        if not condition:
            all_passed = False
    
    print("-" * 70)
    
    if all_passed:
        print("\n✓ SUCCESS: All verification checks passed!")
        print("\nThe fix should resolve the black canvas issue:")
        print("1. When PNG is uploaded, parse_file() generates data URL")
        print("2. Canvas HTML includes data-img-src attribute")
        print("3. After HTML updates, .then(js=_REINIT_CANVAS_JS) EXECUTES the code")
        print("4. window._wb_reset() is called after 200ms")
        print("5. loadImageFromSource() reads data-img-src and loads the image")
        print("6. redraw() displays the image on canvas ✓")
        return True
    else:
        print("\n❌ FAILURE: Some checks failed. Please review above.")
        return False

if __name__ == "__main__":
    success = verify_fix()
    sys.exit(0 if success else 1)
