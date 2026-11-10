//! Process-wide switch for the progress messages printed while a model loads.
//!
//! A library must not write to the standard output of its host unasked, so the switch is **off by default**: loading
//! a model and generating text are silent unless the application calls [`set_verbose`] with `true` (the `quintrit`
//! command-line tool and the benchmarks do). Errors are never printed; they are returned.
//!
//! The switch is a single atomic flag shared by all threads. Use [`VerboseScope`] to enable it for one call and
//! restore the previous value afterwards; concurrent scopes on different threads overlap, so an application that
//! loads several models in parallel should set the flag once instead.

use std::sync::atomic::{AtomicBool, Ordering};

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Turns the progress messages on or off for the whole process. The default is off.
pub fn set_verbose(verbose: bool) {
    VERBOSE.store(verbose, Ordering::Relaxed);
}

/// Returns whether progress messages are currently printed.
pub fn is_verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

/// Sets the switch for the lifetime of the guard and restores the previous value when it is dropped.
#[must_use = "the previous verbosity is restored when the scope is dropped"]
pub struct VerboseScope {
    previous: bool,
}

impl VerboseScope {
    /// Sets the switch to `verbose` until the returned guard is dropped.
    pub fn new(verbose: bool) -> Self {
        Self { previous: VERBOSE.swap(verbose, Ordering::Relaxed) }
    }
}

impl Drop for VerboseScope {
    fn drop(&mut self) {
        VERBOSE.store(self.previous, Ordering::Relaxed);
    }
}

/// Prints a progress line to the standard output when [`is_verbose`] is true.
macro_rules! progress {
    ($($arg:tt)*) => {
        if $crate::verbosity::is_verbose() {
            println!($($arg)*);
        }
    };
}
pub(crate) use progress;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The flag is process-wide, so the tests that change it run one at a time.
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn default_is_silent() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!is_verbose(), "a library must be silent by default");
    }

    #[test]
    fn set_and_scope_restore_the_previous_value() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_verbose(true);
        assert!(is_verbose());
        {
            let _s = VerboseScope::new(false);
            assert!(!is_verbose());
        }
        assert!(is_verbose());
        set_verbose(false);
        {
            let _s = VerboseScope::new(true);
            assert!(is_verbose());
        }
        assert!(!is_verbose());
    }
}
