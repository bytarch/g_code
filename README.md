
# G Code

G Code is a minimal, high-performance CLI AI coding assistant designed as a lightweight alternative to Claude Code. It features a robust XML-based tool execution engine, streaming responses, auto-retry logic, and deep project structural awareness.

## Key Features

- **Real-time Streaming**: Watch the AI think and code in real-time.
- **Project Context (`.gcode` Folder)**: Define project-specific rules, architecture, and prompts to ensure the AI writes code that matches your team's standards.
- **Smart File Reading**: Intelligently centers file reads around specific functions or classes, and automatically breaks large files into manageable chunks.
- **XML-Based Tool Use**: Secure and structured execution of file operations, search, and bash commands.
- **Structural Awareness**: The agent automatically loads directory listings when you reference files, helping it understand the project structure.
- **Auto-Retry & Anti-Loop**: If a command fails, the agent analyzes the error. It also detects repetitive loops to prevent getting stuck.
- **Rich Terminal UI**: Full tab-completion for file paths and syntax-highlighted output.
- **Interactive Config**: Set your API Key and Model directly from the chat interface; settings are persisted automatically.

## Installation

The easiest way to use G Code globally is to install it directly from GitHub:

```bash
pip install git+https://github.com/bytarch/g_code.git
```

## Configuration

G Code requires a BytArch-compatible API key. Configuration is handled via a local config folder.

### 1. Project Initialization (Recommended)

Initialize your project to create the `.gcode` folder structure. This allows you to guide the AI agent with your specific coding standards and context.

- **Initialize**: Run `/init` inside the G Code session.

This creates a `.gcode/` folder in your working directory with the following files:

- **`rules.md`**: Define coding standards (e.g., "Use logging instead of print", "Add type hints").
- **`context.md`**: Describe the project architecture and tech stack.
- **`prompts.md`**: Store reusable prompt snippets for common tasks.

By filling out these files, you ensure the AI agent generates code that fits your specific project style and structure.

### 2. Interactive Setup

You can also set your credentials directly within the application using slash commands.

- **Set API Key**: `/key sk-your-api-key-here`
- **Set Model**: `/model gpt-4` (or `kwaipilot/kat-coder-pro`, etc.)

### 3. Environment Variables (Fallback)

If no config file is found, G Code will fall back to environment variables or a `.env` file in your working directory.

```bash
# .env file
BYTARCH_API_KEY=bsk-***
MODEL=kwaipilot/kat-coder-pro
```

### Storage Location

- **Settings**: Saved to `~/.g_code/.config/settings.json` (or next to the script).
- **Project Rules**: Saved to `./.gcode/` (inside your project directory).

## Usage

Once installed, simply type `gcode` in any terminal to start the session:

```bash
gcode
```

## Interactive Commands

- `/init`: Initialize the `.gcode` folder with `rules.md`, `context.md`, and `prompts.md` templates.
- `@path/to/file`: Read a specific file into the AI's context.
- `@directory/`: Provide the AI with a list of all files in a directory.
- `/key <api_key>`: Update and save your API key.
- `/model <model_name>`: Update and save the active model.
- `/c`: Clear the current conversation history.
- `/q` or `exit`: Safely exit the application.

## Integrated Tools

The agent can autonomously perform:

- **File Ops**: read (with smart search/chunking), write (chunked creation), edit (search/replace), copy, move.
- **Search**: grep (content search) and glob (file finding).
- **Execution**: bash (run shell commands/tests).

## Development

If you want to contribute or customize the logic:

1. Clone the repository:
   ```bash
   git clone https://github.com/bytarch/g_code.git
   ```
2. Install in editable mode:
   ```bash
   pip install -e .
   ```

## License

MIT
