use anyhow::{Context, Result};
use decompute_core::SessionCacheStats;
use opentelemetry::{KeyValue, metrics::MeterProvider};
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_otlp::{LogExporter, MetricExporter};
use opentelemetry_sdk::{
    Resource,
    logs::{SdkLoggerProvider, log_processor_with_async_runtime::BatchLogProcessor},
    metrics::{SdkMeterProvider, periodic_reader_with_async_runtime::PeriodicReader},
    runtime::TokioCurrentThread,
};
use std::{env, sync::Arc, time::Duration};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub struct Observability {
    logger_provider: Option<SdkLoggerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl Observability {
    pub fn init() -> Result<(Self, Arc<SessionCacheStats>)> {
        let stats = SessionCacheStats::shared();
        if env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_none() {
            tracing_subscriber::fmt().with_env_filter("info").init();
            return Ok((
                Self {
                    logger_provider: None,
                    meter_provider: None,
                },
                stats,
            ));
        }

        let service_name =
            env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "decompute-worker".into());
        let resource = Resource::builder()
            .with_service_name(service_name)
            .with_attributes([KeyValue::new("decompute.component", "worker")])
            .build();
        let log_exporter = LogExporter::builder()
            .with_http()
            .build()
            .context("build OTLP log exporter")?;
        let metric_exporter = MetricExporter::builder()
            .with_http()
            .build()
            .context("build OTLP metric exporter")?;
        let log_processor = BatchLogProcessor::builder(log_exporter, TokioCurrentThread).build();
        let logger_provider = SdkLoggerProvider::builder()
            .with_resource(resource.clone())
            .with_log_processor(log_processor)
            .build();
        let metric_reader = PeriodicReader::builder(metric_exporter, TokioCurrentThread).build();
        let meter_provider = SdkMeterProvider::builder()
            .with_resource(resource)
            .with_reader(metric_reader)
            .build();
        let meter = meter_provider.meter("decompute-worker");
        let stats_for_metrics = stats.clone();
        let _hits = meter
            .u64_observable_counter("decompute.session_cache.hits")
            .with_description("Session cache hits")
            .with_callback(move |observer| observer.observe(stats_for_metrics.snapshot()[0], &[]))
            .build();
        let stats_for_metrics = stats.clone();
        let _misses = meter
            .u64_observable_counter("decompute.session_cache.misses")
            .with_description("Session cache misses")
            .with_callback(move |observer| observer.observe(stats_for_metrics.snapshot()[1], &[]))
            .build();
        let stats_for_metrics = stats.clone();
        let _evictions = meter
            .u64_observable_counter("decompute.session_cache.evictions")
            .with_description("Session cache evictions")
            .with_callback(move |observer| observer.observe(stats_for_metrics.snapshot()[2], &[]))
            .build();
        let stats_for_metrics = stats.clone();
        let _expirations = meter
            .u64_observable_counter("decompute.session_cache.expirations")
            .with_description("Session cache expirations")
            .with_callback(move |observer| observer.observe(stats_for_metrics.snapshot()[3], &[]))
            .build();
        let stats_for_metrics = stats.clone();
        let _invalidations = meter
            .u64_observable_counter("decompute.session_cache.invalidations")
            .with_description("Session cache invalidations")
            .with_callback(move |observer| observer.observe(stats_for_metrics.snapshot()[4], &[]))
            .build();
        let otel_logs = OpenTelemetryTracingBridge::new(&logger_provider);
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(otel_logs)
            .with(tracing_subscriber::EnvFilter::from_default_env())
            .init();
        tracing::info!("OpenTelemetry OTLP logs and metrics enabled");
        Ok((
            Self {
                logger_provider: Some(logger_provider),
                meter_provider: Some(meter_provider),
            },
            stats,
        ))
    }

    pub fn shutdown(&self) {
        if let Some(provider) = &self.meter_provider {
            let _ = provider.shutdown_with_timeout(Duration::from_secs(5));
        }
        if let Some(provider) = &self.logger_provider {
            let _ = provider.shutdown_with_timeout(Duration::from_secs(5));
        }
    }
}

impl Drop for Observability {
    fn drop(&mut self) {
        self.shutdown();
    }
}
