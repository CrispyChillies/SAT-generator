#!/usr/bin/env python3
"""
Test the graph answer verification function to ensure it correctly identifies
the right answer based on the graph data.
"""
import sys
sys.path.insert(0, '.')

from generate_question_langchain import _verify_graph_correct_answer

def test_verify_smallest():
    """Test finding smallest value"""
    question = "For what model year is the percent of cars for sale the smallest?"
    choices = ["2010", "2011", "2013", "2014"]
    x_values = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    y_values = [15, 14, 12, 8, 4, 9, 10, 10, 11, 12]
    
    # LLM might incorrectly say A (2010)
    llm_answer = "A"
    
    # Should return D (2014, which has y_value 4, the smallest)
    correct = _verify_graph_correct_answer(question, choices, x_values, y_values, llm_answer)
    
    print(f"Test 'smallest':")
    print(f"  Question: {question}")
    print(f"  Choices: {choices}")
    print(f"  X values: {x_values[:5]}... (showing first 5)")
    print(f"  Y values: {y_values[:5]}... (showing first 5)")
    print(f"  LLM answered: {llm_answer} ({choices[0]})")
    print(f"  Verified answer: {correct} ({choices[ord(correct) - ord('A')]})")
    print(f"  ✓ PASS" if correct == "D" else f"  ✗ FAIL")
    print()
    
    return correct == "D"


def test_verify_largest():
    """Test finding largest value"""
    question = "In which year was the percentage the highest?"
    choices = ["2015", "2016", "2017", "2018"]
    x_values = [2015, 2016, 2017, 2018]
    y_values = [20, 35, 28, 30]
    
    # LLM might incorrectly say A (2015)
    llm_answer = "A"
    
    # Should return B (2016, which has y_value 35, the largest)
    correct = _verify_graph_correct_answer(question, choices, x_values, y_values, llm_answer)
    
    print(f"Test 'largest':")
    print(f"  Question: {question}")
    print(f"  Choices: {choices}")
    print(f"  X values: {x_values}")
    print(f"  Y values: {y_values}")
    print(f"  LLM answered: {llm_answer} ({choices[0]})")
    print(f"  Verified answer: {correct} ({choices[ord(correct) - ord('A')]})")
    print(f"  ✓ PASS" if correct == "B" else f"  ✗ FAIL")
    print()
    
    return correct == "B"


def test_verify_already_correct():
    """Test when LLM answer is already correct"""
    question = "For what year is the value the lowest?"
    choices = ["2010", "2011", "2012", "2013"]
    x_values = [2010, 2011, 2012, 2013]
    y_values = [10, 5, 8, 12]
    
    # LLM correctly says B
    llm_answer = "B"
    
    # Should return B (2011, which has y_value 5, the smallest)
    correct = _verify_graph_correct_answer(question, choices, x_values, y_values, llm_answer)
    
    print(f"Test 'already correct':")
    print(f"  Question: {question}")
    print(f"  Choices: {choices}")
    print(f"  X values: {x_values}")
    print(f"  Y values: {y_values}")
    print(f"  LLM answered: {llm_answer} ({choices[1]})")
    print(f"  Verified answer: {correct} ({choices[ord(correct) - ord('A')]})")
    print(f"  ✓ PASS (no change needed)" if correct == "B" else f"  ✗ FAIL")
    print()
    
    return correct == "B"


def main():
    print("=" * 70)
    print("Testing Graph Answer Verification")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("Test smallest", test_verify_smallest()))
    results.append(("Test largest", test_verify_largest()))
    results.append(("Test already correct", test_verify_already_correct()))
    
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
