#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced GIL Bypass Extension

This test suite explores various methodologies for achieving true concurrency
in CPython through careful circumvention of the Global Interpreter Lock's
protective mechanisms. The tests range from "merely inadvisable" to 
"fundamentally incompatible with the continued existence of your process."

The theoretical foundation for this work rests on the observation that the GIL
is, ultimately, just a mutex. And mutexes, as any systems programmer knows,
are more of a polite suggestion than an immutable law of physics.

WARNING: The following code may cause your Python interpreter to experience
what computer scientists euphemistically call "undefined behavior" and what
everyone else calls "crashing." Run in a disposable environment, preferably
one that you were planning to rebuild anyway.
"""

import time
import threading
import gc
import sys
import os
import ctypes
import mmap
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import dis

import gilfree as gilf

# Global data structures for demonstrating race conditions
shared_counter = defaultdict(int)
shared_list = []
shared_dict = {}
shared_deque = deque()

# More sophisticated shared state for advanced corruption
class SharedDataStructure:
    """
    A deliberately non-thread-safe data structure designed to
    exhibit interesting failure modes under concurrent access.
    """
    def __init__(self):
        self.data = {}
        self.metadata = {'access_count': 0, 'modification_count': 0}
        self.internal_list = []
        self.reference_chain = [self]  # Circular reference for GC fun
        
    def update(self, key, value, thread_id):
        """Update operation designed to maximize race condition potential."""
        # Multiple operations that can be interleaved
        old_value = self.data.get(key, 0)
        time.sleep(0.0001)  # Deliberate yield point
        self.data[key] = old_value + value
        self.metadata['modification_count'] += 1
        self.internal_list.append(f"thread_{thread_id}_key_{key}")
        
        # Reference count manipulation for extra chaos
        temp_ref = self.reference_chain[0]
        self.reference_chain.append(temp_ref)
        
    def read_complex(self, thread_id):
        """Complex read operation that touches multiple data structures."""
        self.metadata['access_count'] += 1
        total = sum(self.data.values())
        list_len = len(self.internal_list)
        return {'total': total, 'list_len': list_len, 'thread_id': thread_id}

# Global instance for maximum chaos
global_shared_data = SharedDataStructure()

def cpu_intensive_with_gil_detection(thread_id, iterations=1000000, use_gil_bypass=False):
    """
    A CPU-intensive task that performs various operations designed to
    detect whether the GIL is properly protecting us. Includes deliberate
    race conditions, reference count manipulation, and memory allocation
    patterns that should be impossible under normal threading.
    """
    
    local_sum = 0
    local_allocations = []
    gil_bypass_depth = 0
    
    # Check if we're in a GIL bypass context
    if use_gil_bypass:
        gil_bypass_depth = gilf.get_gil_bypass_depth()
    
    for i in range(iterations):
        # Mathematical computation to keep the CPU busy
        local_sum += i * i + (i % 7) * (i % 11)
        
        # Periodic operations that should reveal race conditions
        if i % 1000 == 0:
            # Memory allocation and immediate deallocation
            temp_list = [j for j in range(100)]
            local_allocations.append(len(temp_list))
            
            # Shared data structure manipulation
            shared_counter[thread_id] += 1
            shared_list.append(f"thread_{thread_id}_iteration_{i}")
            shared_dict[f"thread_{thread_id}_key_{i}"] = local_sum
            
            # Complex shared data structure operation
            global_shared_data.update(thread_id, i, thread_id)
            
            # Deliberate yield to increase interleaving probability
            if use_gil_bypass:
                gilf.force_memory_barrier()
            else:
                time.sleep(0.00001)
        
        # Periodic garbage collection triggers for reference counting chaos
        if i % 10000 == 0:
            # Force some garbage collection activity
            temp_objects = [object() for _ in range(50)]
            del temp_objects
            
            # Check our GIL bypass depth periodically
            if use_gil_bypass:
                current_depth = gilf.get_gil_bypass_depth()
                if current_depth != gil_bypass_depth:
                    # This would indicate nested GIL bypass operations
                    shared_dict[f"depth_change_{thread_id}"] = current_depth
    
    # Final complex operation
    result = global_shared_data.read_complex(thread_id)
    result['local_sum'] = local_sum
    result['allocations'] = len(local_allocations)
    
    return result

def memory_stress_test(thread_id, size_mb=10):
    """
    Stress test memory allocation and deallocation patterns under
    concurrent execution. This should reveal interesting behaviors
    in Python's memory management when the GIL isn't protecting us.
    """
    
    allocated_blocks = []
    total_allocated = 0
    
    # Allocate and deallocate memory in patterns designed to stress
    # the memory allocator
    for i in range(100):
        # Variable-sized allocations
        size = (i * 1024 + thread_id * 512) % (size_mb * 1024 * 1024 // 100)
        block = bytearray(size)
        
        # Fill with recognizable pattern
        pattern = (thread_id * 256 + i) % 256
        for j in range(0, len(block), 256):
            block[j:j+1] = bytes([pattern])
        
        allocated_blocks.append(block)
        total_allocated += size
        
        # Periodically deallocate some blocks
        if i % 10 == 0 and allocated_blocks:
            # Deallocate in reverse order to fragment memory
            deallocated = allocated_blocks.pop()
            total_allocated -= len(deallocated)
            del deallocated
        
        # Shared state updates during memory operations
        shared_counter[f"memory_{thread_id}"] += size
        
        # Force memory barrier if available
        try:
            gilf.force_memory_barrier()
        except:
            pass
    
    return {'thread_id': thread_id, 'final_allocated': total_allocated, 
            'blocks_remaining': len(allocated_blocks)}

def demonstrate_advanced_gil_bypass():
    """
    Demonstrate the GIL-free functionality with comprehensive
    monitoring of the various failure modes we expect to encounter.
    """
    print("=" * 80)
    print("GIL-FREE THREADING DEMONSTRATION")
    print("Utilizing thread state interpolation with memory barriers")
    print("=" * 80)
    
    # Clear all shared state
    shared_counter.clear()
    shared_list.clear()
    shared_dict.clear()
    shared_deque.clear()
    global_shared_data.__init__()
    
    # Force garbage collection to start with clean state
    gc.collect()
    
    num_threads = 4
    threads = []
    start_time = time.time()
    
    # Configure threads with various advanced options
    thread_configs = [
        {'memory_barriers': True, 'signal_handlers': True, 'stack_switching': False},
        {'memory_barriers': True, 'signal_handlers': False, 'stack_switching': False},
        {'memory_barriers': False, 'signal_handlers': True, 'stack_switching': False},
        {'memory_barriers': False, 'signal_handlers': False, 'stack_switching': False},
    ]
    
    print(f"Creating {num_threads} GIL-free threads with varying configurations...")
    
    for i in range(num_threads):
        config = thread_configs[i % len(thread_configs)]
        
        # Create thread with mixed workload
        def mixed_workload(tid=i):
            cpu_result = cpu_intensive_with_gil_detection(tid, 200000, True)
            memory_result = memory_stress_test(tid, 5)
            return {'cpu': cpu_result, 'memory': memory_result}
        
        thread = gilf.GILFreeThread(target=mixed_workload)
        thread.configure(**config)
        
        threads.append(thread)
        
        print(f"  Thread {i}: {config}")
        thread.start()
        
        # Brief delay to stagger thread starts
        time.sleep(0.01)
    
    print("\nThreads started. Monitoring for signs of concurrent execution...")
    
    # Monitor execution while threads run
    monitor_start = time.time()
    last_counter_state = dict(shared_counter)
    
    while any(not thread.get_stats()['thread_finished'] for thread in threads):
        time.sleep(0.1)
        current_time = time.time() - monitor_start
        
        # Check for signs of true concurrency
        current_counter_state = dict(shared_counter)
        if current_counter_state != last_counter_state:
            changes = sum(current_counter_state.values()) - sum(last_counter_state.values())
            print(f"  [{current_time:.1f}s] Detected {changes} shared state modifications")
            last_counter_state = current_counter_state.copy()
        
        # Safety timeout
        if current_time > 30:
            print("  Timeout reached. Some threads may have achieved heat death of the universe.")
            break
    
    print("\nCollecting results...")
    
    # Wait for all threads to complete and collect results
    results = []
    for i, thread in enumerate(threads):
        try:
            result = thread.join()
            stats = thread.get_stats()
            results.append({'result': result, 'stats': stats, 'thread_id': i})
            
            status = "CRASHED" if stats['thread_crashed'] else "COMPLETED"
            print(f"  Thread {i}: {status}")
            
        except Exception as e:
            print(f"  Thread {i}: EXCEPTION - {e}")
            results.append({'result': None, 'stats': None, 'thread_id': i, 'exception': str(e)})
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nExecution completed in {execution_time:.3f} seconds")
    
    # Analyze results for signs of successful GIL bypass
    print("\n" + "=" * 60)
    print("CONCURRENCY ANALYSIS")
    print("=" * 60)
    
    # Check shared data structure consistency
    total_modifications = global_shared_data.metadata['modification_count']
    expected_modifications = num_threads * 200  # Rough estimate
    
    print(f"Global data structure modifications: {total_modifications}")
    print(f"Expected approximately: {expected_modifications}")
    
    if total_modifications > expected_modifications * 0.8:
        print("✓ High modification count suggests successful concurrent execution")
    else:
        print("✗ Low modification count may indicate serialized execution")
    
    # Check for evidence of race conditions
    shared_list_len = len(shared_list)
    shared_dict_len = len(shared_dict)
    counter_total = sum(shared_counter.values())
    
    print(f"\nShared data structures:")
    print(f"  List entries: {shared_list_len}")
    print(f"  Dictionary entries: {shared_dict_len}")
    print(f"  Counter total: {counter_total}")
    
    # Look for signs of memory corruption or inconsistency
    inconsistencies = 0
    
    # Check for duplicate entries in shared list (sign of race condition)
    if len(set(shared_list)) != len(shared_list):
        inconsistencies += 1
        print("⚠ Duplicate entries detected in shared list (race condition)")
    
    # Check for missing counter entries
    expected_threads = set(range(num_threads))
    actual_threads = set(k for k in shared_counter.keys() if isinstance(k, int))
    if actual_threads != expected_threads:
        inconsistencies += 1
        print(f"⚠ Thread counter mismatch: expected {expected_threads}, got {actual_threads}")
    
    print(f"\nInconsistencies detected: {inconsistencies}")
    
    if inconsistencies > 0:
        print("✓ Data inconsistencies suggest successful GIL bypass")
        print("  (Race conditions are evidence of true concurrency)")
    else:
        print("? No obvious inconsistencies detected")
        print("  (Either the GIL protected us, or we got lucky)")
    
    # Performance analysis
    if execution_time < 2.0:  # Rough threshold
        print(f"\n✓ Fast execution time ({execution_time:.3f}s) suggests parallel processing")
    else:
        print(f"\n? Execution time ({execution_time:.3f}s) inconclusive")
    
    return results

def demonstrate_bytecode_execution():
    """
    Demonstrate direct bytecode execution without GIL protection.
    This is where we really start to push the boundaries of what
    should be possible in a reasonable world.
    """
    print("\n" + "=" * 80)
    print("DIRECT BYTECODE EXECUTION WITHOUT GIL")
    print("Performing interpreter surgery with a rusty scalpel")
    print("=" * 80)
    
    # Create some bytecode to execute that doesn't rely on external variables
    source_code = """
# Simple computation that doesn't require external dependencies
result = 0
temp_dict = {}
for i in range(100000):
    result += i * i
    if i % 10000 == 0:
        temp_dict[f'checkpoint_{i}'] = result
result
"""
    
    # Compile to bytecode
    try:
        code_obj = compile(source_code, '<gil_bypass_test>', 'eval')
    except SyntaxError as e:
        # If eval fails, try exec mode
        print(f"Eval compilation failed: {e}")
        print("Trying exec mode...")
        source_code = """
result = 0
temp_dict = {}
for i in range(100000):
    result += i * i
    if i % 10000 == 0:
        temp_dict[f'checkpoint_{i}'] = result
"""
        try:
            code_obj = compile(source_code, '<gil_bypass_test>', 'exec')
        except SyntaxError as e2:
            print(f"Exec compilation also failed: {e2}")
            print("Skipping bytecode execution test.")
            return
    
    print("Bytecode to execute:")
    dis.dis(code_obj)
    
    print("\nExecuting bytecode without GIL protection...")
    
    try:
        # Clear shared state
        shared_counter.clear()
        
        # Prepare execution environment
        exec_globals = {
            'shared_counter': shared_counter,
            'range': range,  # Ensure built-ins are available
        }
        exec_locals = {}
        
        # Execute without GIL
        start_time = time.time()
        if code_obj.co_flags & 0x20:  # CO_GENERATOR flag
            # Handle generator code
            result = eval(code_obj, exec_globals, exec_locals)
        else:
            # Handle regular code
            if source_code.strip().endswith('result'):
                # This is an expression, use eval
                result = gilf.execute_bytecode_nogil(code_obj, exec_globals, exec_locals)
            else:
                # This is a statement, use exec and extract result
                gilf.execute_bytecode_nogil(code_obj, exec_globals, exec_locals)
                result = exec_locals.get('result', 'No result variable found')
        
        end_time = time.time()
        
        print(f"Execution completed in {(end_time - start_time) * 1000:.2f}ms")
        print(f"Result: {result}")
        print(f"Shared counter state: {dict(shared_counter)}")
        
        if result is not None:
            print("✓ Bytecode execution succeeded without GIL protection")
        else:
            print("✗ Bytecode execution returned None (possible failure)")
            
    except Exception as e:
        print(f"✗ Bytecode execution failed: {e}")
        print("  This is not entirely unexpected")

def comparative_performance_analysis():
    """
    Compare performance between normal threading, advanced GIL bypass,
    and sub-interpreter approaches. This gives us quantitative data
    on just how much the GIL is actually protecting us from ourselves.
    """
    print("\n" + "=" * 80)
    print("COMPARATIVE PERFORMANCE ANALYSIS")
    print("Quantifying the performance cost of thread safety")
    print("=" * 80)
    
    test_iterations = 500000
    num_threads = 4
    
    def cpu_workload(thread_id):
        return cpu_intensive_with_gil_detection(thread_id, test_iterations, False)
    
    def gil_bypass_workload(thread_id):
        return cpu_intensive_with_gil_detection(thread_id, test_iterations, True)
    
    # Test 1: Normal Python threading
    print("1. Normal Python threading (GIL-protected)...")
    shared_counter.clear()
    shared_list.clear()
    global_shared_data.__init__()
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(cpu_workload, i) for i in range(num_threads)]
        normal_results = [f.result() for f in futures]
    normal_time = time.time() - start_time
    
    normal_counter_state = dict(shared_counter)
    normal_list_len = len(shared_list)
    
    print(f"   Time: {normal_time:.3f}s")
    print(f"   Shared modifications: {sum(normal_counter_state.values())}")
    print(f"   List entries: {normal_list_len}")
    
    # Test 2: GIL-free threading
    print("\n2. GIL-free threading (thread state interpolation)...")
    shared_counter.clear()
    shared_list.clear()
    global_shared_data.__init__()
    
    start_time = time.time()
    bypass_threads = []
    
    for i in range(num_threads):
        thread = gilf.GILFreeThread(
            target=gil_bypass_workload, args=(i,)
        )
        thread.configure(memory_barriers=True, signal_handlers=True)
        bypass_threads.append(thread)
        thread.start()
    
    bypass_results = []
    for thread in bypass_threads:
        try:
            result = thread.join()
            bypass_results.append(result)
        except Exception as e:
            bypass_results.append(None)
            print(f"   Thread crashed: {e}")
    
    bypass_time = time.time() - start_time
    
    bypass_counter_state = dict(shared_counter)
    bypass_list_len = len(shared_list)
    
    print(f"   Time: {bypass_time:.3f}s")
    print(f"   Shared modifications: {sum(bypass_counter_state.values())}")
    print(f"   List entries: {bypass_list_len}")
    
    # Performance comparison
    print("\n" + "-" * 60)
    print("PERFORMANCE COMPARISON")
    print("-" * 60)
    
    if bypass_time > 0:
        speedup = normal_time / bypass_time
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✓ Significant speedup indicates successful parallel execution")
        elif speedup > 1.1:
            print("? Modest speedup may indicate partial parallel execution")
        else:
            print("✗ No significant speedup detected")
    
    # Data integrity comparison
    print(f"\nData integrity comparison:")
    print(f"Normal threading - Counter: {sum(normal_counter_state.values())}, List: {normal_list_len}")
    print(f"GIL-free - Counter: {sum(bypass_counter_state.values())}, List: {bypass_list_len}")
    
    # Calculate expected values
    expected_counter = num_threads * (test_iterations // 1000)
    expected_list = num_threads * (test_iterations // 1000)
    
    print(f"Expected values - Counter: {expected_counter}, List: {expected_list}")
    
    normal_accuracy = abs(sum(normal_counter_state.values()) - expected_counter) / expected_counter
    bypass_accuracy = abs(sum(bypass_counter_state.values()) - expected_counter) / expected_counter
    
    print(f"Normal threading accuracy: {(1 - normal_accuracy) * 100:.1f}%")
    print(f"GIL-free accuracy: {(1 - bypass_accuracy) * 100:.1f}%")
    
    if bypass_accuracy > normal_accuracy * 2:
        print("⚠ GIL-free shows significant data corruption")
        print("  (This is evidence of successful concurrency)")
    else:
        print("? Data corruption levels similar to normal threading")

def stress_test_reference_counting():
    """
    Stress test Python's reference counting mechanism under concurrent
    access. This is designed to demonstrate why the GIL exists in the
    first place by showing what happens when we ignore it.
    """
    print("\n" + "=" * 80)
    print("REFERENCE COUNTING STRESS TEST")
    print("Demonstrating why Guido was right about the GIL")
    print("=" * 80)
    
    # Create objects with complex reference patterns
    class CircularReference:
        def __init__(self, value):
            self.value = value
            self.references = [self]
            self.back_refs = []
            
        def add_reference(self, other):
            self.references.append(other)
            other.back_refs.append(self)
            
        def remove_reference(self, other):
            if other in self.references:
                self.references.remove(other)
            if self in other.back_refs:
                other.back_refs.remove(self)
    
    # Shared pool of objects for maximum chaos
    object_pool = [CircularReference(i) for i in range(100)]
    
    # Create circular references
    for i, obj in enumerate(object_pool):
        obj.add_reference(object_pool[(i + 1) % len(object_pool)])
        obj.add_reference(object_pool[(i + 50) % len(object_pool)])
    
    def reference_manipulation_workload(thread_id, iterations=50000):
        """Manipulate object references in patterns designed to stress the reference counter."""
        local_objects = []
        
        for i in range(iterations):
            # Create new objects and add to pool
            new_obj = CircularReference(f"thread_{thread_id}_obj_{i}")
            local_objects.append(new_obj)
            
            # Randomly connect to global pool
            if i % 100 == 0 and object_pool:
                target_idx = (thread_id * i) % len(object_pool)
                target_obj = object_pool[target_idx]
                
                new_obj.add_reference(target_obj)
                target_obj.add_reference(new_obj)
                
                # Occasionally break references
                if i % 500 == 0:
                    new_obj.remove_reference(target_obj)
            
            # Periodic cleanup to trigger reference counting
            if i % 1000 == 0:
                # Remove some local objects
                if local_objects:
                    removed = local_objects.pop(0)
                    # Clear all references
                    for ref in removed.references[:]:
                        removed.remove_reference(ref)
                    del removed
                
                # Force garbage collection
                gc.collect()
        
        return {'thread_id': thread_id, 'objects_created': iterations, 'objects_remaining': len(local_objects)}
    
    print("Starting reference counting stress test...")
    print("Creating complex object graphs with circular references...")
    
    num_threads = 4
    threads = []
    
    start_time = time.time()
    
    for i in range(num_threads):
        thread = gilf.GILFreeThread(
            target=reference_manipulation_workload,
            args=(i, 20000)
        )
        thread.configure(memory_barriers=True, signal_handlers=True)
        threads.append(thread)
        thread.start()
    
    # Monitor garbage collection activity
    gc_start_collections = gc.get_count()
    
    results = []
    for i, thread in enumerate(threads):
        try:
            result = thread.join()
            results.append(result)
            print(f"  Thread {i}: Created {result['objects_created']} objects")
        except Exception as e:
            print(f"  Thread {i}: CRASHED - {e}")
            results.append({'exception': str(e)})
    
    end_time = time.time()
    gc_end_collections = gc.get_count()
    
    print(f"\nStress test completed in {end_time - start_time:.3f} seconds")
    
    # Check for signs of reference counting corruption
    remaining_objects = len(object_pool)
    print(f"Object pool size after test: {remaining_objects}")
    
    # Check garbage collection statistics
    gc_diff = tuple(end - start for start, end in zip(gc_start_collections, gc_end_collections))
    print(f"Garbage collection activity: {gc_diff}")
    
    # Check for memory leaks or corruption
    print("\nPerforming post-test object integrity check...")
    integrity_failures = 0
    
    for i, obj in enumerate(object_pool[:10]):  # Check first 10 objects
        try:
            # Test basic operations
            ref_count = len(obj.references)
            back_ref_count = len(obj.back_refs)
            value = str(obj.value)
            
            # Look for obviously corrupted state
            if ref_count > 1000 or back_ref_count > 1000:
                integrity_failures += 1
                print(f"  Object {i}: Suspicious reference count ({ref_count}, {back_ref_count})")
                
        except Exception as e:
            integrity_failures += 1
            print(f"  Object {i}: Integrity check failed - {e}")
    
    if integrity_failures > 0:
        print(f"\n⚠ {integrity_failures} objects show signs of corruption")
        print("  Reference counting race conditions detected")
    else:
        print("\n? No obvious object corruption detected")
        print("  Either the GIL protected us, or we got very lucky")

if __name__ == "__main__":
    print("GIL-Free Extension - Comprehensive Test Suite")
    print("=" * 80)
    print("This test suite explores the boundaries of what should be possible")
    print("in a programming language that values memory safety.")
    print()
    print("WARNING: The following tests quibble with Python's thread")
    print("safety model and may cause crashes, data corruption, or undefined")
    print("behavior. Run only in a disposable environment.")
    print("=" * 80)
    
    import warnings
    warnings.warn(
        "This test suite performs operations that would make the Python "
        "development team question your life choices.", 
        RuntimeWarning, stacklevel=2
    )
    
    try:
        # Test 1: GIL-free demonstration
        demonstrate_advanced_gil_bypass()
        
        # Brief pause to let any crashes manifest
        time.sleep(1)
        
        # Test 2: Direct bytecode execution
        demonstrate_bytecode_execution()
        
        time.sleep(1)
        
        # Test 3: Performance comparison
        comparative_performance_analysis()
        
        time.sleep(1)
        
        # Test 4: Reference counting stress test
        stress_test_reference_counting()
        
        print("\n" + "=" * 80)
        print("TEST SUITE COMPLETION")
        print("=" * 80)
        print("If you're reading this message, your Python interpreter has")
        print("survived an extended assault on its fundamental assumptions")
        print("about thread safety. This is either a testament to the")
        print("robustness of CPython's implementation, or evidence that")
        print("our GIL bypass techniques need refinement.")
        print()
        print("Check your data integrity carefully. Race conditions have")
        print("a tendency to manifest in subtle and delayed ways.")
        
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user.")
        print("This is probably the most sensible outcome.")
        
    except Exception as e:
        print(f"\nTest suite terminated with exception: {e}")
        print("This is consistent with our theoretical predictions.")
        
    finally:
        # Attempt cleanup
        try:
            gc.collect()
            print("\nFinal garbage collection completed.")
        except:
            print("\nFinal garbage collection failed. This is concerning.")
