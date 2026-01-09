# Homebrew Formula for Proxima Agent
# =============================================================================
# This formula is a template for Homebrew distribution.
# Submit to Homebrew/homebrew-core or create a custom tap.
# =============================================================================
#
# To use with a custom tap:
# 1. Create a GitHub repo: github.com/proxima-project/homebrew-proxima
# 2. Place this file as Formula/proxima.rb
# 3. Users install with: brew tap proxima-project/proxima && brew install proxima
#
# =============================================================================

class Proxima < Formula
  include Language::Python::Virtualenv

  desc "Intelligent Quantum Simulation Orchestration Framework"
  homepage "https://github.com/proxima-project/proxima"
  url "https://files.pythonhosted.org/packages/source/p/proxima-agent/proxima-agent-0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256_HASH"  # Update with actual hash
  license "MIT"
  head "https://github.com/proxima-project/proxima.git", branch: "main"

  # Bottle configuration (pre-built binaries)
  # bottle do
  #   sha256 cellar: :any_skip_relocation, arm64_sonoma: "HASH"
  #   sha256 cellar: :any_skip_relocation, arm64_ventura: "HASH"
  #   sha256 cellar: :any_skip_relocation, sonoma: "HASH"
  #   sha256 cellar: :any_skip_relocation, ventura: "HASH"
  #   sha256 cellar: :any_skip_relocation, x86_64_linux: "HASH"
  # end

  depends_on "python@3.11"

  # Core dependencies
  resource "typer" do
    url "https://files.pythonhosted.org/packages/source/t/typer/typer-0.9.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic/pydantic-2.5.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "pydantic-settings" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic-settings/pydantic_settings-2.1.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "structlog" do
    url "https://files.pythonhosted.org/packages/source/s/structlog/structlog-23.2.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "psutil" do
    url "https://files.pythonhosted.org/packages/source/p/psutil/psutil-5.9.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/source/h/httpx/httpx-0.25.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "pandas" do
    url "https://files.pythonhosted.org/packages/source/p/pandas/pandas-2.1.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "openpyxl" do
    url "https://files.pythonhosted.org/packages/source/o/openpyxl/openpyxl-3.1.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/source/p/pyyaml/PyYAML-6.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "transitions" do
    url "https://files.pythonhosted.org/packages/source/t/transitions/transitions-0.9.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "keyring" do
    url "https://files.pythonhosted.org/packages/source/k/keyring/keyring-24.0.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/source/r/rich/rich-13.0.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/source/a/anyio/anyio-4.0.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  # Quantum backends (optional, but included for full functionality)
  resource "cirq" do
    url "https://files.pythonhosted.org/packages/source/c/cirq/cirq-1.3.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "qiskit" do
    url "https://files.pythonhosted.org/packages/source/q/qiskit/qiskit-0.45.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "qiskit-aer" do
    url "https://files.pythonhosted.org/packages/source/q/qiskit-aer/qiskit-aer-0.13.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      Proxima Agent has been installed successfully!

      To get started:
        proxima init          # Initialize configuration
        proxima --help        # Show all commands

      For LLM features, you may want to:
        - Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables
        - Or install Ollama for local LLM inference

      Configuration file location:
        ~/.proxima/config.yaml

      Documentation:
        https://proxima.readthedocs.io
    EOS
  end

  test do
    # Test version command
    assert_match "proxima", shell_output("#{bin}/proxima version")
    
    # Test help command
    assert_match "Usage", shell_output("#{bin}/proxima --help")
    
    # Test backends list (should work without quantum packages)
    output = shell_output("#{bin}/proxima backends list 2>&1", 0)
    assert_match(/backend|available/i, output)
  end
end
