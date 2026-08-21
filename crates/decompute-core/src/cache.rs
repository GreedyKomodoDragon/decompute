use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration shared by the worker and the in-process inference runtime.
/// Durations are represented as milliseconds so this type remains transport
/// and runtime neutral.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCacheConfig {
    /// Do not retain prompts shorter than this many tokens.
    pub min_tokens: usize,
    /// Maximum number of cached sessions.
    pub max_entries: usize,
    /// Maximum reserved KV bytes across cached contexts. Zero means unlimited.
    pub max_bytes: usize,
    /// Idle expiry in milliseconds. Zero disables idle expiry.
    pub idle_ttl_ms: u64,
    /// Maximum time a same-session request waits for the session lease.
    pub slot_wait_timeout_ms: u64,
}

impl Default for SessionCacheConfig {
    fn default() -> Self {
        Self {
            min_tokens: 100,
            max_entries: 1,
            max_bytes: 0,
            idle_ttl_ms: 15 * 60 * 1_000,
            slot_wait_timeout_ms: 30 * 1_000,
        }
    }
}

impl SessionCacheConfig {
    pub fn idle_ttl(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.idle_ttl_ms)
    }

    pub fn slot_wait_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.slot_wait_timeout_ms)
    }
}

/// Process-local cumulative cache counters. The runtime updates these without
/// retaining session identifiers, prompts, or token contents.
#[derive(Debug, Default)]
pub struct SessionCacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub expirations: AtomicU64,
    pub invalidations: AtomicU64,
}

impl SessionCacheStats {
    pub fn increment_hits(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }
    pub fn increment_misses(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
    pub fn increment_evictions(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
    pub fn increment_expirations(&self) {
        self.expirations.fetch_add(1, Ordering::Relaxed);
    }
    pub fn increment_invalidations(&self) {
        self.invalidations.fetch_add(1, Ordering::Relaxed);
    }
    pub fn snapshot(&self) -> [u64; 5] {
        [
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
            self.expirations.load(Ordering::Relaxed),
            self.invalidations.load(Ordering::Relaxed),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::{SessionCacheConfig, SessionCacheMiss, SessionCacheStats};

    #[test]
    fn defaults_are_conservative_and_privacy_safe() {
        let config = SessionCacheConfig::default();
        assert_eq!(config.min_tokens, 100);
        assert_eq!(config.max_entries, 1);
        assert_eq!(config.idle_ttl().as_secs(), 900);
        assert_eq!(config.slot_wait_timeout().as_secs(), 30);
        assert_eq!(SessionCacheMiss::PrefixMismatch.as_str(), "prefix_mismatch");
    }

    #[test]
    fn stats_are_cumulative() {
        let stats = SessionCacheStats::default();
        stats.increment_hits();
        stats.increment_evictions();
        assert_eq!(stats.snapshot(), [1, 0, 1, 0, 0]);
    }
}

impl Clone for SessionCacheStats {
    fn clone(&self) -> Self {
        let snapshot = self.snapshot();
        Self {
            hits: AtomicU64::new(snapshot[0]),
            misses: AtomicU64::new(snapshot[1]),
            evictions: AtomicU64::new(snapshot[2]),
            expirations: AtomicU64::new(snapshot[3]),
            invalidations: AtomicU64::new(snapshot[4]),
        }
    }
}

impl SessionCacheStats {
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionCacheMiss {
    Disabled,
    TooShort,
    Missing,
    Expired,
    PrefixMismatch,
    Incompatible,
    Overflow,
    Contended,
    Invalidated,
}

impl SessionCacheMiss {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::TooShort => "too_short",
            Self::Missing => "missing",
            Self::Expired => "expired",
            Self::PrefixMismatch => "prefix_mismatch",
            Self::Incompatible => "incompatible",
            Self::Overflow => "overflow",
            Self::Contended => "contended",
            Self::Invalidated => "invalidated",
        }
    }
}
