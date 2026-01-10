"""
Configuration module for Proxima.

This module provides comprehensive configuration management including:
- Layered configuration loading (CLI > env > user > project > defaults)
- Pydantic-based settings validation
- Secure secrets handling
- Configuration migration between versions
- Export/import utilities
- File watching for auto-reload
- Schema documentation and introspection
"""

from proxima.config.defaults import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_RELATIVE_PATH,
    ENV_PREFIX,
    PROJECT_CONFIG_FILENAME,
    USER_CONFIG_PATH,
)

from proxima.config.settings import (
    # Settings classes
    Settings,
    GeneralSettings,
    BackendsSettings,
    LLMSettings,
    ResourcesSettings,
    ConsentSettings,
    # Config service
    ConfigService,
    config_service,
    # Convenience functions
    get_settings,
    reload_settings,
    FlatSettings,
)

from proxima.config.validation import (
    # Enums
    ValidationSeverity,
    # Data classes
    ValidationIssue,
    ValidationResult,
    # Main validation
    validate_settings,
    validate_config_file,
    # Individual validators
    validate_verbosity,
    validate_output_format,
    validate_backend,
    validate_timeout,
    validate_llm_provider,
    validate_url,
    validate_memory_threshold,
    validate_path,
    validate_storage_backend,
    validate_model_name,
    validate_env_var_name,
)

from proxima.config.export_import import (
    # Enums
    ExportFormat,
    # Data classes
    ExportOptions,
    ImportResult,
    BackupInfo,
    # Export functions
    export_config,
    # Import functions
    import_config,
    import_from_url,
    # Backup functions
    create_backup,
    list_backups,
    restore_backup,
    cleanup_old_backups,
    # Template functions
    generate_template,
)

from proxima.config.secrets import (
    # Enums
    SecretBackend,
    # Data classes
    SecretMetadata,
    SecretResult,
    # Storage backends
    SecretStorage,
    KeyringStorage,
    EncryptedFileStorage,
    EnvironmentStorage,
    MemoryStorage,
    # Main manager
    SecretManager,
    get_secret_manager,
    # Utilities
    generate_secret_key,
    mask_secret,
    validate_api_key_format,
)

from proxima.config.migration import (
    # Constants
    CURRENT_VERSION,
    # Enums
    MigrationDirection,
    # Data classes
    MigrationStep,
    MigrationResult,
    # Classes
    MigrationRegistry,
    ConfigMigrator,
    # Functions
    get_config_version,
    set_config_version,
    needs_migration,
    get_migrator,
    auto_migrate,
    check_migration_status,
    # Decorators
    migration,
    register_pending_migrations,
)

from proxima.config.schema import (
    # Enums
    FieldType,
    # Data classes
    FieldInfo,
    SectionInfo,
    # Introspection
    introspect_model,
    # Documentation generation
    generate_markdown_docs,
    generate_json_schema,
    generate_completion_data,
    # Convenience
    get_settings_schema,
    get_field_help,
    get_field_examples,
    list_all_settings,
    print_settings_tree,
)

from proxima.config.watcher import (
    # Enums
    WatchEvent,
    # Data classes
    FileChange,
    # Watchers
    FileWatcher,
    PollingWatcher,
    WatchdogWatcher,
    ConfigWatcher,
    WatchedConfigService,
    # Convenience
    create_config_watcher,
    watch_config_file,
)


__all__ = [
    # ==========================================================================
    # DEFAULTS
    # ==========================================================================
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "ENV_PREFIX",
    "PROJECT_CONFIG_FILENAME",
    "USER_CONFIG_PATH",
    
    # ==========================================================================
    # SETTINGS
    # ==========================================================================
    # Settings classes
    "Settings",
    "GeneralSettings",
    "BackendsSettings",
    "LLMSettings",
    "ResourcesSettings",
    "ConsentSettings",
    # Config service
    "ConfigService",
    "config_service",
    # Convenience functions
    "get_settings",
    "reload_settings",
    "FlatSettings",
    
    # ==========================================================================
    # VALIDATION
    # ==========================================================================
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "validate_settings",
    "validate_config_file",
    "validate_verbosity",
    "validate_output_format",
    "validate_backend",
    "validate_timeout",
    "validate_llm_provider",
    "validate_url",
    "validate_memory_threshold",
    "validate_path",
    "validate_storage_backend",
    "validate_model_name",
    "validate_env_var_name",
    
    # ==========================================================================
    # EXPORT/IMPORT
    # ==========================================================================
    "ExportFormat",
    "ExportOptions",
    "ImportResult",
    "BackupInfo",
    "export_config",
    "import_config",
    "import_from_url",
    "create_backup",
    "list_backups",
    "restore_backup",
    "cleanup_old_backups",
    "generate_template",
    
    # ==========================================================================
    # SECRETS
    # ==========================================================================
    "SecretBackend",
    "SecretMetadata",
    "SecretResult",
    "SecretStorage",
    "KeyringStorage",
    "EncryptedFileStorage",
    "EnvironmentStorage",
    "MemoryStorage",
    "SecretManager",
    "get_secret_manager",
    "generate_secret_key",
    "mask_secret",
    "validate_api_key_format",
    
    # ==========================================================================
    # MIGRATION
    # ==========================================================================
    "CURRENT_VERSION",
    "MigrationDirection",
    "MigrationStep",
    "MigrationResult",
    "MigrationRegistry",
    "ConfigMigrator",
    "get_config_version",
    "set_config_version",
    "needs_migration",
    "get_migrator",
    "auto_migrate",
    "check_migration_status",
    "migration",
    "register_pending_migrations",
    
    # ==========================================================================
    # SCHEMA
    # ==========================================================================
    "FieldType",
    "FieldInfo",
    "SectionInfo",
    "introspect_model",
    "generate_markdown_docs",
    "generate_json_schema",
    "generate_completion_data",
    "get_settings_schema",
    "get_field_help",
    "get_field_examples",
    "list_all_settings",
    "print_settings_tree",
    
    # ==========================================================================
    # WATCHER
    # ==========================================================================
    "WatchEvent",
    "FileChange",
    "FileWatcher",
    "PollingWatcher",
    "WatchdogWatcher",
    "ConfigWatcher",
    "WatchedConfigService",
    "create_config_watcher",
    "watch_config_file",
]
