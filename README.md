# G Code

G Code is a minimal, high-performance CLI AI coding assistant designed as a lightweight alternative to Claude Code. It features a robust XML-based tool execution engine, streaming responses, auto-retry logic, and deep project structural awareness.

## Key Features

- **Real-time Streaming**: Watch the AI think and code in real-time.
- **XML-Based Tool Use**: Secure and structured execution of file operations, search, and bash commands.
- **Structural Awareness**: The agent understands your directory tree and OS environment out of the box.
- **@References**: Effortlessly attach files or folders to your prompt by typing `@filename` or `@directory/`.
- **Auto-Retry Logic**: If a command fails, the agent analyzes the error and self-corrects.
- **Rich Terminal UI**: Full tab-completion for file paths and syntax-highlighted output.

## Installation

The easiest way to use G Code globally is to install it directly from GitHub:

```bash
pip install git+https://github.com/bytarch/g_code.git
```

## Configuration

G Code requires an OpenAI-compatible API key. You can set it up in two ways:

### 1. Environment Variable (Recommended)

Add this to your `.zshrc`, `.bashrc`, or Windows Environment Variables:

```bash
set BYTARCH_API_KEY=bsk_***
```

### 2. .env File

Place a `.env` file in your working directory:

```bash
BYTARCH_API_KEY=bsk_***
MODEL=kwaipilot/kat-coder-pro
```

## Usage

Once installed, simply type `gcode` in any terminal to start the session:

```bash
gcode
```

## Interactive Commands

- `@path/to/file`: Read a specific file into the AI's context.
- `@directory/`: Provide the AI with a list of all files in a directory.
- `/c`: Clear the current conversation history.
- `/q` or `exit`: Safely exit the application.

## Integrated Tools

The agent can autonomously perform:

- **File Ops**: read, write, edit (search/replace), copy, move.
- **Search**: grep (content search) and glob (file finding).
- **Execution**: bash (run shell commands/tests).

Development

If you want to contribute or customize the logic:

Clone the repository: git clone https://github.com/bytarch/g_code.git

Install in editable mode: pip install -e .

License

MIT