#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <structmember.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>

/*
 * Advanced GIL Circumvention via Thread State Interpolation
 * 
 * This extension demonstrates several methodologies for achieving true
 * concurrency in CPython by carefully avoiding the Global Interpreter Lock's
 * protective mechanisms. The approaches range from "merely inadvisable" to
 * "fundamentally incompatible with the continued existence of your process."
 * 
 * The GIL exists because Guido van Rossum and the CPython developers have
 * a quaint attachment to concepts like "memory safety" and "deterministic
 * program execution." This extension respectfully disagrees.
 * 
 * Technical Background:
 * The GIL is implemented as a simple mutex around the interpreter core loop.
 * It protects reference counting operations, object allocation, and most
 * importantly, the thread state switching mechanism. By carefully manipulating
 * thread states and selectively ignoring synchronization primitives, we can
 * achieve what computer scientists call "true parallelism" and what systems
 * engineers call "a really interesting way to crash."
 */

/* Thread-local storage for our interpreter state shenanigans */
static __thread PyThreadState *tls_thread_state = NULL;
static __thread int tls_gil_bypass_depth = 0;

typedef struct {
    PyObject_HEAD
    pthread_t thread_id;
    PyObject *callable;
    PyObject *args;
    PyObject *kwargs;
    PyObject *result;
    PyObject *exception;
    volatile int thread_started;
    volatile int thread_finished;
    volatile int thread_crashed;
    PyInterpreterState *interp_state;
    PyThreadState *original_tstate;
    
    /* Advanced features for the truly adventurous */
    int use_memory_barriers;
    int use_signal_handlers;
    int enable_stack_switching;
    void *alternate_stack;
    size_t stack_size;
} GILFreeThread;

typedef struct {
    GILFreeThread *thread_obj;
    PyThreadState *thread_state;
    void *stack_base;
} ThreadData;

/*
 * Signal handler for when things go predictably wrong.
 * In a properly designed system, this would never execute.
 * In our system, it's basically the primary error handling mechanism.
 */
static void catastrophic_failure_handler(int sig) {
    const char msg[] = "GIL bypass extension: Thread experienced unexpected "
                       "rapid unscheduled disassembly. This was, statistically speaking, "
                       "quite likely to occur.\n";
    /* Using write() because we can't trust printf in a signal handler */
    write(STDERR_FILENO, msg, sizeof(msg) - 1);
    
    /* Clean exit is for programs that haven't violated the fundamental
     * assumptions of their runtime environment */
    _exit(127);
}

/*
 * Memory barrier implementation for architectures that support it.
 * On x86/x64, this compiles to an MFENCE instruction.
 * On ARM, it's a DMB.
 * On architectures that don't support memory barriers, it compiles
 * to a polite suggestion that the CPU maybe consider flushing its
 * write buffers when convenient.
 */
static inline void memory_barrier(void) {
    __asm__ __volatile__ ("" ::: "memory");
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__ ("mfence" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__ ("dmb sy" ::: "memory");
#else
    /* For other architectures, we'll just hope for the best.
     * This is consistent with our overall design philosophy. */
    __sync_synchronize();
#endif
}

/*
 * Stack switching mechanism for the truly committed.
 * This allows each thread to run with its own stack space,
 * which would be useful if we weren't already violating
 * every other assumption about thread safety.
 */
static int setup_alternate_stack(GILFreeThread *self) {
    self->stack_size = 1024 * 1024; /* 1MB should be enough for anyone */
    
    self->alternate_stack = mmap(NULL, self->stack_size,
                                PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS,
                                -1, 0);
    
    if (self->alternate_stack == MAP_FAILED) {
        return -1;
    }
    
    /* Set up a guard page to catch stack overflows.
     * This is our one concession to safety. */
    if (mprotect(self->alternate_stack, getpagesize(), PROT_NONE) != 0) {
        munmap(self->alternate_stack, self->stack_size);
        return -1;
    }
    
    return 0;
}

/*
 * The main thread execution function.
 * This is where we perform the actual GIL bypass through a technique
 * I like to call "thread state interpolation" - we create a new thread
 * state and simply... start using it. Without asking permission.
 * It's the software equivalent of adverse possession.
 */
static void* thread_runner(void *arg) {
    ThreadData *data = (ThreadData*)arg;
    GILFreeThread *self = data->thread_obj;
    PyThreadState *tstate = data->thread_state;
    
    /* Install our signal handler because optimism is not a debugging strategy */
    signal(SIGSEGV, catastrophic_failure_handler);
    signal(SIGBUS, catastrophic_failure_handler);
    signal(SIGFPE, catastrophic_failure_handler);
    
    /* Switch to alternate stack if requested */
    if (self->enable_stack_switching && self->alternate_stack) {
        stack_t ss;
        ss.ss_sp = (char*)self->alternate_stack + getpagesize();
        ss.ss_size = self->stack_size - getpagesize();
        ss.ss_flags = 0;
        
        if (sigaltstack(&ss, NULL) == 0) {
            /* We're now running on our own stack. This would be impressive
             * if it weren't also completely unnecessary. */
        }
    }
    
    /* Store thread state in thread-local storage for easy access */
    tls_thread_state = tstate;
    tls_gil_bypass_depth = 1;
    
    /* Insert memory barrier to ensure all CPUs agree on reality */
    if (self->use_memory_barriers) {
        memory_barrier();
    }
    
    /*
     * Here's the critical moment: we're about to switch to our thread state
     * without acquiring the GIL. This is roughly equivalent to performing
     * surgery without washing your hands - technically possible, but the
     * infection rate is quite high.
     */
    PyThreadState *old_tstate = PyThreadState_Swap(tstate);
    
    /* Another memory barrier because paranoia is the only rational response
     * to what we're attempting */
    if (self->use_memory_barriers) {
        memory_barrier();
    }
    
    /*
     * We are now in the quantum superposition of being both protected
     * and unprotected by the GIL. Schrödinger's thread state, if you will.
     * Multiple threads can now execute Python code simultaneously.
     * What could possibly go wrong?
     */
    
    PyObject *result = NULL;
    
    /* Increment our bypass depth counter for nested mayhem */
    tls_gil_bypass_depth++;
    
    /* Execute the Python callable in our lawless wasteland */
    if (self->kwargs && PyDict_Size(self->kwargs) > 0) {
        result = PyObject_Call(self->callable, self->args, self->kwargs);
    } else {
        result = PyObject_CallObject(self->callable, self->args);
    }
    
    /* Check if we've discovered any exciting new failure modes */
    if (PyErr_Occurred()) {
        self->exception = PyErr_GetRaisedException();
        self->thread_crashed = 1;
    } else {
        self->result = result;
    }
    
    tls_gil_bypass_depth--;
    
    /* Final memory barrier before we restore the natural order */
    if (self->use_memory_barriers) {
        memory_barrier();
    }
    
    /* Restore the previous thread state, completing our brief
     * excursion into concurrent Python execution */
    PyThreadState_Swap(old_tstate);
    
    /* Mark ourselves as finished rather than crashed, which is
     * honestly more optimistic than realistic */
    self->thread_finished = 1;
    
    /* Clean up our thread data. If we've made it this far,
     * we've probably violated several laws of computer science
     * but at least we're cleaning up after ourselves. */
    free(data);
    
    return NULL;
}

static PyObject* GILFreeThread_start(GILFreeThread *self, PyObject *args) {
    if (self->thread_started) {
        PyErr_SetString(PyExc_RuntimeError, 
            "Thread already started. Multiple simultaneous GIL bypasses "
            "would be excessive even by our standards.");
        return NULL;
    }
    
    /*
     * Create a new thread state for this thread.
     * In a sane world, this would require GIL protection.
     * We live in a different world now.
     */
    PyThreadState *new_tstate = PyThreadState_New(self->interp_state);
    if (!new_tstate) {
        PyErr_SetString(PyExc_RuntimeError, 
            "Failed to create thread state. Even Python is trying to protect you.");
        return NULL;
    }
    
    /* Set up alternate stack if the user has requested maximum chaos */
    if (self->enable_stack_switching) {
        if (setup_alternate_stack(self) != 0) {
            PyThreadState_Delete(new_tstate);
            PyErr_SetString(PyExc_RuntimeError, 
                "Failed to set up alternate stack. This is probably for the best.");
            return NULL;
        }
    }
    
    /* Prepare our thread data structure */
    ThreadData *thread_data = malloc(sizeof(ThreadData));
    if (!thread_data) {
        if (self->alternate_stack) {
            munmap(self->alternate_stack, self->stack_size);
        }
        PyThreadState_Delete(new_tstate);
        return PyErr_NoMemory();
    }
    
    thread_data->thread_obj = self;
    thread_data->thread_state = new_tstate;
    thread_data->stack_base = self->alternate_stack;
    
    /*
     * Create the pthread with enhanced attributes for maximum instability.
     * We're requesting a detached thread with a custom stack size,
     * because if you're going to violate thread safety, you might as
     * well do it with style.
     */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    
    if (self->enable_stack_switching && self->alternate_stack) {
        pthread_attr_setstack(&attr, 
                             (char*)self->alternate_stack + getpagesize(),
                             self->stack_size - getpagesize());
    }
    
    int result = pthread_create(&self->thread_id, &attr, 
                               thread_runner, thread_data);
    
    pthread_attr_destroy(&attr);
    
    if (result != 0) {
        free(thread_data);
        if (self->alternate_stack) {
            munmap(self->alternate_stack, self->stack_size);
        }
        PyThreadState_Delete(new_tstate);
        PyErr_SetString(PyExc_RuntimeError, 
            "Failed to create pthread. The OS is also trying to protect you.");
        return NULL;
    }
    
    self->thread_started = 1;
    Py_RETURN_NONE;
}

static PyObject* GILFreeThread_join(GILFreeThread *self, PyObject *args) {
    if (!self->thread_started) {
        PyErr_SetString(PyExc_RuntimeError, 
            "Thread not started. You can't join a thread that doesn't exist, "
            "even in our post-GIL anarchist collective.");
        return NULL;
    }
    
    /* Release GIL before joining - this is actually the correct thing to do */
    Py_BEGIN_ALLOW_THREADS
    pthread_join(self->thread_id, NULL);
    Py_END_ALLOW_THREADS
    
    /* Check if our thread experienced any of the predicted failure modes */
    if (self->thread_crashed) {
        if (self->exception) {
            PyErr_SetObject(PyExc_RuntimeError, self->exception);
        } else {
            PyErr_SetString(PyExc_RuntimeError, 
                "Thread crashed in a way that defied our ability to record the details.");
        }
        return NULL;
    }
    
    /* Return the result, assuming we managed to compute one */
    if (self->result) {
        PyObject *result = self->result;
        Py_INCREF(result);
        return result;
    }
    
    Py_RETURN_NONE;
}

static PyObject* GILFreeThread_configure(GILFreeThread *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"memory_barriers", "signal_handlers", "stack_switching", NULL};
    
    PyObject *memory_barriers = NULL;
    PyObject *signal_handlers = NULL;
    PyObject *stack_switching = NULL;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO", kwlist,
                                     &memory_barriers, &signal_handlers, &stack_switching)) {
        return NULL;
    }
    
    if (memory_barriers) {
        self->use_memory_barriers = PyObject_IsTrue(memory_barriers);
    }
    
    if (signal_handlers) {
        self->use_signal_handlers = PyObject_IsTrue(signal_handlers);
    }
    
    if (stack_switching) {
        self->enable_stack_switching = PyObject_IsTrue(stack_switching);
    }
    
    Py_RETURN_NONE;
}

static PyObject* GILFreeThread_get_stats(GILFreeThread *self, PyObject *args) {
    PyObject *stats = PyDict_New();
    
    PyDict_SetItemString(stats, "thread_started", 
                        self->thread_started ? Py_True : Py_False);
    PyDict_SetItemString(stats, "thread_finished", 
                        self->thread_finished ? Py_True : Py_False);
    PyDict_SetItemString(stats, "thread_crashed", 
                        self->thread_crashed ? Py_True : Py_False);
    PyDict_SetItemString(stats, "use_memory_barriers", 
                        self->use_memory_barriers ? Py_True : Py_False);
    PyDict_SetItemString(stats, "enable_stack_switching", 
                        self->enable_stack_switching ? Py_True : Py_False);
    
    if (self->alternate_stack) {
        PyDict_SetItemString(stats, "stack_size", 
                           PyLong_FromSize_t(self->stack_size));
    }
    
    return stats;
}

static PyMethodDef GILFreeThread_methods[] = {
    {"start", (PyCFunction)GILFreeThread_start, METH_NOARGS,
     "Start the thread execution with GIL-free techniques"},
    {"join", (PyCFunction)GILFreeThread_join, METH_NOARGS,
     "Wait for thread completion and return result"},
    {"configure", (PyCFunction)GILFreeThread_configure, METH_VARARGS | METH_KEYWORDS,
     "Configure threading options"},
    {"get_stats", (PyCFunction)GILFreeThread_get_stats, METH_NOARGS,
     "Get threading statistics and configuration"},
    {NULL}
};

static int GILFreeThread_init(GILFreeThread *self, PyObject *args, PyObject *kwds) {
    PyObject *callable = NULL;
    PyObject *thread_args = NULL;
    PyObject *thread_kwargs = NULL;
    
    static char *kwlist[] = {"target", "args", "kwargs", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist,
                                     &callable, &thread_args, &thread_kwargs)) {
        return -1;
    }
    
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, 
            "target must be callable. We may have abandoned thread safety, "
            "but we still maintain basic type checking standards.");
        return -1;
    }
    
    Py_INCREF(callable);
    self->callable = callable;
    
    if (thread_args) {
        Py_INCREF(thread_args);
        self->args = thread_args;
    } else {
        self->args = PyTuple_New(0);
    }
    
    if (thread_kwargs) {
        Py_INCREF(thread_kwargs);
        self->kwargs = thread_kwargs;
    } else {
        self->kwargs = PyDict_New();
    }
    
    /* Initialize our various state tracking variables */
    self->result = NULL;
    self->exception = NULL;
    self->thread_started = 0;
    self->thread_finished = 0;
    self->thread_crashed = 0;
    
    /* Default configuration: maximum chaos */
    self->use_memory_barriers = 1;
    self->use_signal_handlers = 1;
    self->enable_stack_switching = 0; /* Even we have some limits */
    self->alternate_stack = NULL;
    self->stack_size = 0;
    
    /* Store the current interpreter state for later shenanigans */
    self->interp_state = PyThreadState_Get()->interp;
    self->original_tstate = PyThreadState_Get();
    
    return 0;
}

static void GILFreeThread_dealloc(GILFreeThread *self) {
    if (self->thread_started && !self->thread_finished) {
        /* If the thread is still running, we'll wait for it.
         * This is probably futile, but politeness costs nothing. */
        pthread_join(self->thread_id, NULL);
    }
    
    if (self->alternate_stack) {
        munmap(self->alternate_stack, self->stack_size);
    }
    
    Py_XDECREF(self->callable);
    Py_XDECREF(self->args);
    Py_XDECREF(self->kwargs);
    Py_XDECREF(self->result);
    Py_XDECREF(self->exception);
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyTypeObject GILFreeThreadType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gilfree.GILFreeThread",
    .tp_doc = "Thread that executes Python code without GIL protection using various techniques",
    .tp_basicsize = sizeof(GILFreeThread),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)GILFreeThread_init,
    .tp_dealloc = (destructor)GILFreeThread_dealloc,
    .tp_methods = GILFreeThread_methods,
};

/*
 * Direct bytecode execution without GIL.
 * This function attempts to execute Python bytecode directly
 * in the current thread without GIL protection. It's essentially
 * performing interpreter surgery with a rusty scalpel.
 */
static PyObject* execute_bytecode_nogil(PyObject *self, PyObject *args) {
    PyObject *code_obj;
    PyObject *globals = NULL;
    PyObject *locals = NULL;
    
    if (!PyArg_ParseTuple(args, "O|OO", &code_obj, &globals, &locals)) {
        return NULL;
    }
    
    if (!PyCode_Check(code_obj)) {
        PyErr_SetString(PyExc_TypeError, 
            "First argument must be a code object. Even in our lawless realm, "
            "we maintain some standards.");
        return NULL;
    }
    
    if (globals == NULL) {
        globals = PyEval_GetGlobals();
        if (globals == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "No globals available");
            return NULL;
        }
    }
    
    if (locals == NULL) {
        locals = globals;
    }
    
    /*
     * Here's where we perform the actual GIL bypass for bytecode execution.
     * We're going to manually construct a frame and execute it without
     * any GIL protection. This is like performing open-heart surgery
     * while riding a motorcycle - technically possible, but the margin
     * for error is quite narrow.
     */
    
    PyThreadState *tstate = PyThreadState_Get();
    
    /* Create a new frame for our bytecode */
    PyFrameObject *frame = PyFrame_New(tstate, (PyCodeObject*)code_obj, globals, locals);
    if (!frame) {
        return NULL;
    }
    
    /*
     * Install our catastrophic failure handler because we're about to
     * do something that has a non-zero probability of causing a segfault.
     */
    signal(SIGSEGV, catastrophic_failure_handler);
    
    PyObject *result;
    
    /*
     * Execute the frame without GIL protection.
     * We're essentially calling PyEval_EvalFrame() in a context where
     * the GIL isn't held. This works until it doesn't, at which point
     * it fails spectacularly.
     */
    
    /* Memory barrier to ensure consistent view across all CPUs */
    memory_barrier();
    
    /* Mark that we're in a GIL bypass operation */
    tls_gil_bypass_depth++;
    
    /*
     * The actual evaluation. This is where the magic happens.
     * By "magic" I mean "undefined behavior that occasionally
     * produces the correct result."
     */
    result = PyEval_EvalFrame(frame);
    
    tls_gil_bypass_depth--;
    
    /* Another memory barrier because consistency is important */
    memory_barrier();
    
    /* Clean up our frame */
    Py_DECREF(frame);
    
    return result;
}

/*
 * Function to check if we're currently in a GIL bypass operation.
 * This is useful for debugging, or for implementing additional
 * safety checks that we'll promptly ignore.
 */
static PyObject* get_gil_bypass_depth(PyObject *self, PyObject *args) {
    return PyLong_FromLong(tls_gil_bypass_depth);
}

/*
 * Function to force a memory barrier from Python code.
 * Because sometimes you need to ensure memory consistency
 * in your race condition.
 */
static PyObject* force_memory_barrier(PyObject *self, PyObject *args) {
    memory_barrier();
    Py_RETURN_NONE;
}

/*
 * Create multiple sub-interpreters and run them concurrently.
 * This is actually a relatively safe approach to Python parallelism,
 * which makes it somewhat disappointing for our purposes.
 */
static PyObject* create_concurrent_interpreters(PyObject *self, PyObject *args) {
    int num_interpreters = 2;
    
    if (!PyArg_ParseTuple(args, "|i", &num_interpreters)) {
        return NULL;
    }
    
    if (num_interpreters < 1 || num_interpreters > 16) {
        PyErr_SetString(PyExc_ValueError, 
            "Number of interpreters must be between 1 and 16. "
            "We have standards, even here.");
        return NULL;
    }
    
    PyObject *interpreters = PyList_New(0);
    PyThreadState *main_tstate = PyThreadState_Get();
    
    for (int i = 0; i < num_interpreters; i++) {
        /* Create a new sub-interpreter */
        PyThreadState *new_tstate = Py_NewInterpreter();
        if (!new_tstate) {
            PyErr_SetString(PyExc_RuntimeError, 
                "Failed to create sub-interpreter. Even Python has limits.");
            Py_DECREF(interpreters);
            return NULL;
        }
        
        /* Switch back to main interpreter */
        PyThreadState_Swap(main_tstate);
        
        /* Store the interpreter as a capsule */
        PyObject *capsule = PyCapsule_New(new_tstate, "SubInterpreter", NULL);
        PyList_Append(interpreters, capsule);
        Py_DECREF(capsule);
    }
    
    return interpreters;
}

static PyMethodDef module_methods[] = {
    {"execute_bytecode_nogil", execute_bytecode_nogil, METH_VARARGS,
     "Execute bytecode without GIL protection (for the truly committed)"},
    {"get_gil_bypass_depth", get_gil_bypass_depth, METH_NOARGS,
     "Get current GIL bypass depth (for debugging your mistakes)"},
    {"force_memory_barrier", force_memory_barrier, METH_NOARGS,
     "Force a memory barrier (because consistency matters, even in chaos)"},
    {"create_concurrent_interpreters", create_concurrent_interpreters, METH_VARARGS,
     "Create multiple sub-interpreters for safer parallelism"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gilfree_module = {
    PyModuleDef_HEAD_INIT,
    "gilfree",
    "Extension for GIL-free Python thread execution using various techniques",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_gilfree(void) {
    PyObject *m;
    
    if (PyType_Ready(&GILFreeThreadType) < 0)
        return NULL;
    
    m = PyModule_Create(&gilfree_module);
    if (m == NULL)
        return NULL;
    
    Py_INCREF(&GILFreeThreadType);
    if (PyModule_AddObject(m, "GILFreeThread", 
                          (PyObject*)&GILFreeThreadType) < 0) {
        Py_DECREF(&GILFreeThreadType);
        Py_DECREF(m);
        return NULL;
    }
    
    /* Add some informative constants */
    PyModule_AddStringConstant(m, "__warning__", 
        "This module quibbles with CPython's thread safety model.");
    
    PyModule_AddStringConstant(m, "__approach__", 
        "Thread state interpolation with optional memory barriers and stack switching");
    
    PyModule_AddStringConstant(m, "__recommended_use__", 
        "Educational purposes. Performance benchmarking. "
        "Demonstrating why the GIL exists.");
        
    PyModule_AddIntConstant(m, "DEFAULT_STACK_SIZE", 1024 * 1024);
    
    return m;
}
