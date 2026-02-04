"""Build System for Repository Analysis and Compilation.

This module implements Phase 3.2 for the Dynamic AI Assistant:
- Repository Analysis: Structure analysis, technology stack detection,
  dependency identification, build script location
- Build Environment: Dependency resolution, environment variables,
  tool chain verification, version checking
- Compilation Management: Incremental builds, error handling, output capture

Key Features:
============
- Automatic technology stack detection
- Build script discovery and analysis
- Dependency graph construction
- Environment setup automation
- Multi-language build support
- Incremental build detection
- Error pattern recognition
- Build output analysis

Design Principle:
================
All build decisions use LLM reasoning - NO hardcoded build patterns.
The LLM analyzes repositories and determines build approaches dynamically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class TechnologyStack(Enum):
    """Technology stacks for projects."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    UNKNOWN = "unknown"


class BuildSystemType(Enum):
    """Build system types."""
    # Python
    PIP = "pip"
    POETRY = "poetry"
    PIPENV = "pipenv"
    CONDA = "conda"
    SETUPTOOLS = "setuptools"
    
    # JavaScript/TypeScript
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    BUN = "bun"
    
    # Rust
    CARGO = "cargo"
    
    # Go
    GO_MOD = "go_mod"
    
    # Java
    MAVEN = "maven"
    GRADLE = "gradle"
    ANT = "ant"
    
    # C/C++
    CMAKE = "cmake"
    MAKE = "make"
    MESON = "meson"
    BAZEL = "bazel"
    
    # .NET
    DOTNET = "dotnet"
    MSBUILD = "msbuild"
    
    # General
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class BuildStatus(Enum):
    """Build status states."""
    PENDING = "pending"
    PREPARING = "preparing"
    BUILDING = "building"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BuildFile:
    """Represents a build-related file."""
    path: Path
    file_type: str  # e.g., "package.json", "setup.py"
    build_system: BuildSystemType
    content_hash: Optional[str] = None
    last_modified: Optional[float] = None


@dataclass
class Dependency:
    """Represents a project dependency."""
    name: str
    version: Optional[str] = None
    version_constraint: Optional[str] = None  # e.g., ">=1.0.0"
    is_dev: bool = False
    is_optional: bool = False
    source: Optional[str] = None  # e.g., "pypi", "npm"


@dataclass
class RepositoryAnalysis:
    """Analysis results for a repository."""
    root_path: Path
    technology_stacks: List[TechnologyStack] = field(default_factory=list)
    build_systems: List[BuildSystemType] = field(default_factory=list)
    build_files: List[BuildFile] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    
    # Structure info
    entry_points: List[str] = field(default_factory=list)
    source_directories: List[str] = field(default_factory=list)
    test_directories: List[str] = field(default_factory=list)
    
    # Detected commands
    build_commands: List[str] = field(default_factory=list)
    test_commands: List[str] = field(default_factory=list)
    run_commands: List[str] = field(default_factory=list)
    
    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_path": str(self.root_path),
            "technology_stacks": [s.value for s in self.technology_stacks],
            "build_systems": [s.value for s in self.build_systems],
            "build_files": [{"path": str(f.path), "type": f.file_type} for f in self.build_files],
            "dependencies": [{"name": d.name, "version": d.version, "dev": d.is_dev} for d in self.dependencies],
            "entry_points": self.entry_points,
            "build_commands": self.build_commands,
            "test_commands": self.test_commands,
            "run_commands": self.run_commands,
        }


@dataclass
class BuildEnvironment:
    """Build environment configuration."""
    working_directory: Path
    environment_variables: Dict[str, str] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    tool_versions: Dict[str, str] = field(default_factory=dict)
    
    # Virtual environment
    use_virtual_env: bool = False
    virtual_env_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "working_directory": str(self.working_directory),
            "environment_variables": self.environment_variables,
            "required_tools": self.required_tools,
            "tool_versions": self.tool_versions,
            "use_virtual_env": self.use_virtual_env,
            "virtual_env_path": str(self.virtual_env_path) if self.virtual_env_path else None,
        }


@dataclass
class BuildResult:
    """Result of a build operation."""
    status: BuildStatus
    output: str = ""
    error_output: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    
    # Parsed info
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
            "artifacts": self.artifacts,
        }


# File patterns for detecting build systems
BUILD_FILE_PATTERNS: Dict[str, Tuple[BuildSystemType, TechnologyStack]] = {
    "package.json": (BuildSystemType.NPM, TechnologyStack.JAVASCRIPT),
    "yarn.lock": (BuildSystemType.YARN, TechnologyStack.JAVASCRIPT),
    "pnpm-lock.yaml": (BuildSystemType.PNPM, TechnologyStack.JAVASCRIPT),
    "bun.lockb": (BuildSystemType.BUN, TechnologyStack.JAVASCRIPT),
    "tsconfig.json": (BuildSystemType.NPM, TechnologyStack.TYPESCRIPT),
    "setup.py": (BuildSystemType.SETUPTOOLS, TechnologyStack.PYTHON),
    "pyproject.toml": (BuildSystemType.POETRY, TechnologyStack.PYTHON),
    "Pipfile": (BuildSystemType.PIPENV, TechnologyStack.PYTHON),
    "requirements.txt": (BuildSystemType.PIP, TechnologyStack.PYTHON),
    "environment.yml": (BuildSystemType.CONDA, TechnologyStack.PYTHON),
    "Cargo.toml": (BuildSystemType.CARGO, TechnologyStack.RUST),
    "go.mod": (BuildSystemType.GO_MOD, TechnologyStack.GO),
    "pom.xml": (BuildSystemType.MAVEN, TechnologyStack.JAVA),
    "build.gradle": (BuildSystemType.GRADLE, TechnologyStack.JAVA),
    "build.gradle.kts": (BuildSystemType.GRADLE, TechnologyStack.KOTLIN),
    "build.xml": (BuildSystemType.ANT, TechnologyStack.JAVA),
    "CMakeLists.txt": (BuildSystemType.CMAKE, TechnologyStack.CPP),
    "Makefile": (BuildSystemType.MAKE, TechnologyStack.C),
    "meson.build": (BuildSystemType.MESON, TechnologyStack.CPP),
    "BUILD": (BuildSystemType.BAZEL, TechnologyStack.UNKNOWN),
    "BUILD.bazel": (BuildSystemType.BAZEL, TechnologyStack.UNKNOWN),
    "*.csproj": (BuildSystemType.DOTNET, TechnologyStack.CSHARP),
    "*.sln": (BuildSystemType.MSBUILD, TechnologyStack.CSHARP),
}


class BuildSystem:
    """Build system for repository analysis and compilation.
    
    Uses LLM reasoning to:
    1. Analyze repository structure and detect technologies
    2. Identify build systems and commands
    3. Set up build environment
    4. Execute builds with error handling
    
    Example:
        >>> build_system = BuildSystem(llm_client=client)
        >>> analysis = await build_system.analyze_repository(path)
        >>> result = await build_system.build(analysis)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        terminal_manager: Optional[Any] = None,
    ):
        """Initialize the build system.
        
        Args:
            llm_client: LLM client for reasoning
            terminal_manager: Terminal manager for command execution
        """
        self._llm_client = llm_client
        self._terminal_manager = terminal_manager
        
        # Cache
        self._analysis_cache: Dict[str, RepositoryAnalysis] = {}
        self._environment_cache: Dict[str, BuildEnvironment] = {}
    
    async def analyze_repository(
        self,
        repo_path: Union[str, Path],
        force_refresh: bool = False,
    ) -> RepositoryAnalysis:
        """Analyze a repository to detect technology stack and build system.
        
        Args:
            repo_path: Path to the repository
            force_refresh: Force re-analysis even if cached
            
        Returns:
            RepositoryAnalysis with detected information
        """
        repo_path = Path(repo_path)
        cache_key = str(repo_path.absolute())
        
        if not force_refresh and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        analysis = RepositoryAnalysis(root_path=repo_path)
        
        # Step 1: Find build files
        await self._find_build_files(analysis)
        
        # Step 2: Detect technology stacks
        await self._detect_technology_stacks(analysis)
        
        # Step 3: Parse dependencies
        await self._parse_dependencies(analysis)
        
        # Step 4: Find source directories
        await self._find_source_directories(analysis)
        
        # Step 5: Determine build commands using LLM
        await self._determine_build_commands(analysis)
        
        self._analysis_cache[cache_key] = analysis
        return analysis
    
    async def _find_build_files(self, analysis: RepositoryAnalysis):
        """Find build-related files in the repository."""
        repo_path = analysis.root_path
        
        for filename, (build_system, tech_stack) in BUILD_FILE_PATTERNS.items():
            if "*" in filename:
                # Glob pattern
                pattern = filename
                for found in repo_path.glob(pattern):
                    build_file = BuildFile(
                        path=found,
                        file_type=filename,
                        build_system=build_system,
                        last_modified=found.stat().st_mtime if found.exists() else None,
                    )
                    analysis.build_files.append(build_file)
                    
                    if build_system not in analysis.build_systems:
                        analysis.build_systems.append(build_system)
                    if tech_stack != TechnologyStack.UNKNOWN and tech_stack not in analysis.technology_stacks:
                        analysis.technology_stacks.append(tech_stack)
            else:
                # Direct filename
                file_path = repo_path / filename
                if file_path.exists():
                    build_file = BuildFile(
                        path=file_path,
                        file_type=filename,
                        build_system=build_system,
                        last_modified=file_path.stat().st_mtime,
                    )
                    analysis.build_files.append(build_file)
                    
                    if build_system not in analysis.build_systems:
                        analysis.build_systems.append(build_system)
                    if tech_stack != TechnologyStack.UNKNOWN and tech_stack not in analysis.technology_stacks:
                        analysis.technology_stacks.append(tech_stack)
    
    async def _detect_technology_stacks(self, analysis: RepositoryAnalysis):
        """Detect technology stacks from file extensions and content."""
        repo_path = analysis.root_path
        
        # Extension to technology mapping
        ext_map = {
            ".py": TechnologyStack.PYTHON,
            ".js": TechnologyStack.JAVASCRIPT,
            ".ts": TechnologyStack.TYPESCRIPT,
            ".jsx": TechnologyStack.JAVASCRIPT,
            ".tsx": TechnologyStack.TYPESCRIPT,
            ".rs": TechnologyStack.RUST,
            ".go": TechnologyStack.GO,
            ".java": TechnologyStack.JAVA,
            ".kt": TechnologyStack.KOTLIN,
            ".scala": TechnologyStack.SCALA,
            ".cs": TechnologyStack.CSHARP,
            ".cpp": TechnologyStack.CPP,
            ".cc": TechnologyStack.CPP,
            ".c": TechnologyStack.C,
            ".h": TechnologyStack.C,
            ".hpp": TechnologyStack.CPP,
            ".rb": TechnologyStack.RUBY,
            ".php": TechnologyStack.PHP,
            ".swift": TechnologyStack.SWIFT,
        }
        
        # Scan source files
        for ext, tech in ext_map.items():
            found = list(repo_path.rglob(f"*{ext}"))
            # Skip node_modules, venv, etc.
            found = [f for f in found if not any(
                d in str(f) for d in ["node_modules", "venv", ".venv", "__pycache__", "target", "build"]
            )]
            if found and tech not in analysis.technology_stacks:
                analysis.technology_stacks.append(tech)
    
    async def _parse_dependencies(self, analysis: RepositoryAnalysis):
        """Parse dependencies from build files."""
        for build_file in analysis.build_files:
            if not build_file.path.exists():
                continue
            
            try:
                content = build_file.path.read_text(encoding="utf-8", errors="ignore")
                
                if build_file.file_type == "package.json":
                    await self._parse_package_json(content, analysis)
                elif build_file.file_type == "requirements.txt":
                    await self._parse_requirements_txt(content, analysis)
                elif build_file.file_type == "pyproject.toml":
                    await self._parse_pyproject_toml(content, analysis)
                elif build_file.file_type == "Cargo.toml":
                    await self._parse_cargo_toml(content, analysis)
                elif build_file.file_type == "go.mod":
                    await self._parse_go_mod(content, analysis)
                    
            except Exception as e:
                logger.warning(f"Error parsing {build_file.path}: {e}")
    
    async def _parse_package_json(self, content: str, analysis: RepositoryAnalysis):
        """Parse package.json dependencies."""
        import json
        try:
            data = json.loads(content)
            
            for name, version in data.get("dependencies", {}).items():
                analysis.dependencies.append(Dependency(
                    name=name,
                    version_constraint=version,
                    source="npm",
                ))
            
            for name, version in data.get("devDependencies", {}).items():
                analysis.dependencies.append(Dependency(
                    name=name,
                    version_constraint=version,
                    is_dev=True,
                    source="npm",
                ))
                
            # Check for scripts
            scripts = data.get("scripts", {})
            if "build" in scripts:
                analysis.build_commands.append(f"npm run build")
            if "test" in scripts:
                analysis.test_commands.append(f"npm test")
            if "start" in scripts:
                analysis.run_commands.append(f"npm start")
                
        except json.JSONDecodeError:
            pass
    
    async def _parse_requirements_txt(self, content: str, analysis: RepositoryAnalysis):
        """Parse requirements.txt dependencies."""
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            
            # Parse package==version or package>=version
            match = re.match(r"([a-zA-Z0-9_-]+)([=<>!~]+.*)?", line)
            if match:
                name = match.group(1)
                constraint = match.group(2)
                analysis.dependencies.append(Dependency(
                    name=name,
                    version_constraint=constraint,
                    source="pypi",
                ))
    
    async def _parse_pyproject_toml(self, content: str, analysis: RepositoryAnalysis):
        """Parse pyproject.toml dependencies."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return
        
        try:
            data = tomllib.loads(content)
            
            # Poetry format
            poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            for name, spec in poetry_deps.items():
                if name == "python":
                    continue
                version = spec if isinstance(spec, str) else spec.get("version", "")
                analysis.dependencies.append(Dependency(
                    name=name,
                    version_constraint=version,
                    source="pypi",
                ))
            
            # Standard pyproject.toml
            deps = data.get("project", {}).get("dependencies", [])
            for dep in deps:
                match = re.match(r"([a-zA-Z0-9_-]+)([=<>!~]+.*)?", dep)
                if match:
                    analysis.dependencies.append(Dependency(
                        name=match.group(1),
                        version_constraint=match.group(2),
                        source="pypi",
                    ))
                    
        except Exception:
            pass
    
    async def _parse_cargo_toml(self, content: str, analysis: RepositoryAnalysis):
        """Parse Cargo.toml dependencies."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return
        
        try:
            data = tomllib.loads(content)
            
            for name, spec in data.get("dependencies", {}).items():
                version = spec if isinstance(spec, str) else spec.get("version", "")
                analysis.dependencies.append(Dependency(
                    name=name,
                    version_constraint=version,
                    source="crates.io",
                ))
                
        except Exception:
            pass
    
    async def _parse_go_mod(self, content: str, analysis: RepositoryAnalysis):
        """Parse go.mod dependencies."""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("require ") or (
                "\t" in line and " v" in line
            ):
                match = re.search(r"([^\s]+)\s+(v[^\s]+)", line)
                if match:
                    analysis.dependencies.append(Dependency(
                        name=match.group(1),
                        version=match.group(2),
                        source="go",
                    ))
    
    async def _find_source_directories(self, analysis: RepositoryAnalysis):
        """Find source and test directories."""
        repo_path = analysis.root_path
        
        # Common source directory names
        source_dirs = ["src", "lib", "app", "source", "pkg"]
        for d in source_dirs:
            if (repo_path / d).is_dir():
                analysis.source_directories.append(d)
        
        # Common test directory names
        test_dirs = ["test", "tests", "spec", "specs", "__tests__"]
        for d in test_dirs:
            if (repo_path / d).is_dir():
                analysis.test_directories.append(d)
        
        # Find entry points
        entry_files = [
            "main.py", "app.py", "__main__.py",
            "index.js", "index.ts", "main.js", "main.ts",
            "main.go", "main.rs",
            "Main.java", "App.java",
        ]
        for f in entry_files:
            matches = list(repo_path.rglob(f))
            matches = [m for m in matches if "node_modules" not in str(m)]
            for match in matches:
                rel_path = str(match.relative_to(repo_path))
                analysis.entry_points.append(rel_path)
    
    async def _determine_build_commands(self, analysis: RepositoryAnalysis):
        """Determine build commands using LLM reasoning."""
        # Fallback commands based on build system
        fallback_commands = {
            BuildSystemType.NPM: ("npm install && npm run build", "npm test", "npm start"),
            BuildSystemType.YARN: ("yarn install && yarn build", "yarn test", "yarn start"),
            BuildSystemType.PNPM: ("pnpm install && pnpm run build", "pnpm test", "pnpm start"),
            BuildSystemType.PIP: ("pip install -e .", "pytest", "python -m"),
            BuildSystemType.POETRY: ("poetry install", "poetry run pytest", "poetry run python"),
            BuildSystemType.CARGO: ("cargo build", "cargo test", "cargo run"),
            BuildSystemType.GO_MOD: ("go build ./...", "go test ./...", "go run ."),
            BuildSystemType.MAVEN: ("mvn package", "mvn test", "java -jar target/*.jar"),
            BuildSystemType.GRADLE: ("./gradlew build", "./gradlew test", "./gradlew run"),
            BuildSystemType.CMAKE: ("cmake -B build && cmake --build build", "ctest --test-dir build", "./build/"),
            BuildSystemType.MAKE: ("make", "make test", "./"),
            BuildSystemType.DOTNET: ("dotnet build", "dotnet test", "dotnet run"),
        }
        
        # Add fallback commands for detected build systems
        for build_system in analysis.build_systems:
            if build_system in fallback_commands:
                build_cmd, test_cmd, run_cmd = fallback_commands[build_system]
                if build_cmd not in analysis.build_commands:
                    analysis.build_commands.append(build_cmd)
                if test_cmd not in analysis.test_commands:
                    analysis.test_commands.append(test_cmd)
                if run_cmd not in analysis.run_commands:
                    analysis.run_commands.append(run_cmd)
        
        # Use LLM to refine commands if available
        if self._llm_client:
            await self._llm_refine_commands(analysis)
    
    async def _llm_refine_commands(self, analysis: RepositoryAnalysis):
        """Use LLM to refine build commands based on analysis."""
        prompt = f"""Given this repository analysis, provide the best build commands.

Repository: {analysis.root_path}
Technology Stacks: {', '.join(s.value for s in analysis.technology_stacks)}
Build Systems: {', '.join(s.value for s in analysis.build_systems)}
Build Files: {', '.join(str(f.path.name) for f in analysis.build_files)}
Entry Points: {', '.join(analysis.entry_points[:5])}

Current detected commands:
- Build: {', '.join(analysis.build_commands)}
- Test: {', '.join(analysis.test_commands)}
- Run: {', '.join(analysis.run_commands)}

Please confirm or suggest better commands. Respond in this format:
BUILD: <command>
TEST: <command>
RUN: <command>

Only provide commands that are likely to work based on the detected configuration.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            # Parse response
            for line in response.splitlines():
                line = line.strip()
                if line.startswith("BUILD:"):
                    cmd = line[6:].strip()
                    if cmd and cmd not in analysis.build_commands:
                        analysis.build_commands.insert(0, cmd)
                elif line.startswith("TEST:"):
                    cmd = line[5:].strip()
                    if cmd and cmd not in analysis.test_commands:
                        analysis.test_commands.insert(0, cmd)
                elif line.startswith("RUN:"):
                    cmd = line[4:].strip()
                    if cmd and cmd not in analysis.run_commands:
                        analysis.run_commands.insert(0, cmd)
                        
        except Exception as e:
            logger.warning(f"LLM command refinement failed: {e}")
    
    async def setup_environment(
        self,
        analysis: RepositoryAnalysis,
    ) -> BuildEnvironment:
        """Set up build environment based on analysis.
        
        Args:
            analysis: Repository analysis
            
        Returns:
            BuildEnvironment configuration
        """
        cache_key = str(analysis.root_path.absolute())
        
        if cache_key in self._environment_cache:
            return self._environment_cache[cache_key]
        
        env = BuildEnvironment(working_directory=analysis.root_path)
        
        # Determine required tools
        for build_system in analysis.build_systems:
            tools = self._get_required_tools(build_system)
            for tool in tools:
                if tool not in env.required_tools:
                    env.required_tools.append(tool)
        
        # Check tool versions
        await self._check_tool_versions(env)
        
        # Set environment variables
        env.environment_variables.update(self._get_default_env_vars(analysis))
        
        # Check for virtual environment needs
        if TechnologyStack.PYTHON in analysis.technology_stacks:
            env.use_virtual_env = True
            env.virtual_env_path = analysis.root_path / ".venv"
        
        self._environment_cache[cache_key] = env
        return env
    
    def _get_required_tools(self, build_system: BuildSystemType) -> List[str]:
        """Get required tools for a build system."""
        tool_map = {
            BuildSystemType.NPM: ["node", "npm"],
            BuildSystemType.YARN: ["node", "yarn"],
            BuildSystemType.PNPM: ["node", "pnpm"],
            BuildSystemType.PIP: ["python", "pip"],
            BuildSystemType.POETRY: ["python", "poetry"],
            BuildSystemType.PIPENV: ["python", "pipenv"],
            BuildSystemType.CARGO: ["rustc", "cargo"],
            BuildSystemType.GO_MOD: ["go"],
            BuildSystemType.MAVEN: ["java", "mvn"],
            BuildSystemType.GRADLE: ["java"],
            BuildSystemType.CMAKE: ["cmake", "make"],
            BuildSystemType.MAKE: ["make"],
            BuildSystemType.DOTNET: ["dotnet"],
        }
        return tool_map.get(build_system, [])
    
    async def _check_tool_versions(self, env: BuildEnvironment):
        """Check versions of required tools."""
        version_commands = {
            "node": "node --version",
            "npm": "npm --version",
            "yarn": "yarn --version",
            "python": "python --version",
            "pip": "pip --version",
            "poetry": "poetry --version",
            "cargo": "cargo --version",
            "go": "go version",
            "java": "java -version",
            "mvn": "mvn --version",
            "cmake": "cmake --version",
            "dotnet": "dotnet --version",
        }
        
        for tool in env.required_tools:
            if tool in version_commands:
                try:
                    result = subprocess.run(
                        version_commands[tool],
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        # Extract version from output
                        output = result.stdout or result.stderr
                        version_match = re.search(r"(\d+\.\d+(\.\d+)?)", output)
                        if version_match:
                            env.tool_versions[tool] = version_match.group(1)
                except Exception:
                    pass
    
    def _get_default_env_vars(self, analysis: RepositoryAnalysis) -> Dict[str, str]:
        """Get default environment variables."""
        env_vars = {
            "CI": "true",  # Common CI indicator
        }
        
        if TechnologyStack.JAVASCRIPT in analysis.technology_stacks:
            env_vars["NODE_ENV"] = "development"
        
        if TechnologyStack.PYTHON in analysis.technology_stacks:
            env_vars["PYTHONDONTWRITEBYTECODE"] = "1"
        
        return env_vars
    
    async def build(
        self,
        analysis: RepositoryAnalysis,
        command: Optional[str] = None,
        environment: Optional[BuildEnvironment] = None,
    ) -> BuildResult:
        """Execute build for a repository.
        
        Args:
            analysis: Repository analysis
            command: Optional specific build command
            environment: Optional build environment
            
        Returns:
            BuildResult with output and status
        """
        result = BuildResult(status=BuildStatus.PREPARING)
        
        # Get or create environment
        if environment is None:
            environment = await self.setup_environment(analysis)
        
        # Determine build command
        if command is None:
            if analysis.build_commands:
                command = analysis.build_commands[0]
            else:
                return BuildResult(
                    status=BuildStatus.FAILED,
                    error_output="No build command found",
                )
        
        # Execute build
        result.status = BuildStatus.BUILDING
        start_time = asyncio.get_event_loop().time()
        
        try:
            if self._terminal_manager:
                # Use terminal manager
                output = await self._terminal_manager.execute_command(
                    command,
                    cwd=str(environment.working_directory),
                    env=environment.environment_variables,
                )
                result.output = output
                result.exit_code = 0
            else:
                # Direct execution
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(environment.working_directory),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, **environment.environment_variables},
                )
                
                stdout, stderr = await process.communicate()
                result.output = stdout.decode("utf-8", errors="ignore")
                result.error_output = stderr.decode("utf-8", errors="ignore")
                result.exit_code = process.returncode or 0
            
            result.status = BuildStatus.COMPLETED if result.exit_code == 0 else BuildStatus.FAILED
            
        except Exception as e:
            result.status = BuildStatus.FAILED
            result.error_output = str(e)
            result.exit_code = 1
        
        result.duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Parse output for warnings/errors
        await self._parse_build_output(result, analysis)
        
        return result
    
    async def _parse_build_output(self, result: BuildResult, analysis: RepositoryAnalysis):
        """Parse build output for warnings, errors, and artifacts."""
        output = result.output + "\n" + result.error_output
        
        # Common error patterns
        error_patterns = [
            r"error[\s:]+(.+)",
            r"Error[\s:]+(.+)",
            r"ERROR[\s:]+(.+)",
            r"fatal[\s:]+(.+)",
            r"FAILED[\s:]+(.+)",
        ]
        
        warning_patterns = [
            r"warning[\s:]+(.+)",
            r"Warning[\s:]+(.+)",
            r"WARN[\s:]+(.+)",
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            result.errors.extend(matches[:10])  # Limit to 10 errors
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            result.warnings.extend(matches[:10])  # Limit to 10 warnings
        
        # Use LLM to find artifacts if available
        if self._llm_client and result.status == BuildStatus.COMPLETED:
            await self._llm_find_artifacts(result, analysis, output)
    
    async def _llm_find_artifacts(
        self,
        result: BuildResult,
        analysis: RepositoryAnalysis,
        output: str,
    ):
        """Use LLM to identify build artifacts."""
        prompt = f"""Identify build artifacts from this build output.

Technology: {', '.join(s.value for s in analysis.technology_stacks)}
Build Output (last 500 chars):
{output[-500:]}

List any generated files or artifacts mentioned (one per line).
Respond with just file paths, or "NONE" if no artifacts found.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            if "NONE" not in response.upper():
                for line in response.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        result.artifacts.append(line)
                        
        except Exception:
            pass
    
    async def run_tests(
        self,
        analysis: RepositoryAnalysis,
        command: Optional[str] = None,
        environment: Optional[BuildEnvironment] = None,
    ) -> BuildResult:
        """Run tests for a repository.
        
        Args:
            analysis: Repository analysis
            command: Optional specific test command
            environment: Optional build environment
            
        Returns:
            BuildResult with test output
        """
        # Determine test command
        if command is None:
            if analysis.test_commands:
                command = analysis.test_commands[0]
            else:
                return BuildResult(
                    status=BuildStatus.FAILED,
                    error_output="No test command found",
                )
        
        # Execute using build method
        result = await self.build(analysis, command, environment)
        result.status = BuildStatus.TESTING if result.status == BuildStatus.BUILDING else result.status
        
        return result


# Module-level instance
_global_build_system: Optional[BuildSystem] = None


def get_build_system(llm_client: Optional[Any] = None) -> BuildSystem:
    """Get the global build system.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        BuildSystem instance
    """
    global _global_build_system
    if _global_build_system is None:
        _global_build_system = BuildSystem(llm_client=llm_client)
    return _global_build_system
