#!/usr/bin/env python3
"""G Code - minimal claude code alternative (OpenAI Compatible + XML + Streaming + Chunked Writing + Auto-Retry + Pre-Check + @References + Tab Completion + Structure Awareness + Interrupt Support + Copy/Move + Current Directory CWD + Smart Read + Forced Chunking + Anti-Loop + Project Init + False Positive Fix + Claude-Style Config + Sequential Execution)"""

import glob as globlib, json, os, re, subprocess, platform
import shutil # For copy and move operations
from dotenv import load_dotenv
from openai import OpenAI

# --- NEW IMPORT FOR TAB COMPLETION ---
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    print(f"{YELLOW}⚠ 'prompt_toolkit' not found. Tab completion disabled. Run: pip install prompt_toolkit{RESET}")

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION SETUP ---
# Determine the directory where this script is located
APP_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(APP_PATH, ".config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")

# Ensure the .config directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)

def load_user_config():
    """Loads API_KEY and MODEL from the config file if it exists."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"{YELLOW}⚠ Warning: Could not load config file: {e}{RESET}")
    return {}

def save_user_config(api_key, model):
    """Saves API_KEY and MODEL to the config file."""
    config = {
        "API_KEY": api_key,
        "MODEL": model
    }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"{RED}Error saving config: {e}{RESET}")

# Load configuration (Priority: Config File > Environment Variables > Defaults)
loaded_config = load_user_config()
API_KEY = loaded_config.get("API_KEY") or os.environ.get("BYTARCH_API_KEY")
API_URL = "https://api.bytarch.dpdns.org/openai/v1/chat/completions"
MODEL = loaded_config.get("MODEL") or os.environ.get("MODEL", "kwaipilot/kat-coder-pro")

# Context Configuration
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", "256000"))

# --- FEATURE SWITCH ---
ENABLE_FUNCTION_CALLING = False

# --- CHUNKING STRATEGY ---
DEFAULT_READ_LIMIT = 100 
# Reduced to prevent context spam when using @. Large files should be read explicitly.
REF_READ_LIMIT = 150 
# Hard cap for the read tool to force iterative reading
MAX_READ_LINES_PER_CALL = 100 

# ANSI colors
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED, MAGENTA = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[35m",
)


# --- Tab Completion Logic ---

if HAS_PROMPT_TOOLKIT:
    class PathCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            last_at_index = text.rfind('@')
            if last_at_index == -1:
                return
            
            if ' ' in text[last_at_index:]:
                return

            search_part = text[last_at_index + 1:]
            
            # --- NEW: Use GLOBAL SCRIPT_DIR (which is CWD) for completion ---
            # This ensures completion works from the directory you are currently in
            cwd = globals().get('SCRIPT_DIR', os.getcwd())
            
            if '/' in search_part:
                parts = search_part.rsplit('/', 1)
                dir_part = parts[0]
                file_prefix = parts[1]
                search_dir = os.path.join(cwd, dir_part)
            else:
                file_prefix = search_part
                search_dir = cwd
            
            try:
                entries = os.listdir(search_dir)
            except (FileNotFoundError, PermissionError):
                return

            for entry in sorted(entries):
                if entry.startswith(file_prefix):
                    full_path = os.path.join(search_dir, entry)
                    is_dir = os.path.isdir(full_path)
                    display_text = f"{entry}/" if is_dir else entry
                    
                    yield Completion(
                        display_text,
                        start_position=-len(file_prefix), 
                        display_meta='DIR' if is_dir else 'FILE'
                    )


# --- Helper Functions ---

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int((len(text) + 3) // 4)


def count_messages_tokens(messages: list) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("role", "")) + 5
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            total += estimate_tokens(json.dumps(content))
    return total


def resolve_references(text, base_dir):
    """
    Finds patterns like @file.py or @folder/ in user text,
    reads their content (truncated if large), and returns context.
    Updated: Uses base_dir (CWD) to resolve relative paths.
    """
    refs = re.findall(r'@([^\s]+)', text)
    if not refs:
        return text, []

    extra_context = []
    
    for ref in refs:
        # --- NEW: Handle Absolute vs Relative Paths based on Base Dir ---
        # If starts with / or DriveLetter: (Windows), it's absolute
        if os.path.isabs(ref):
            path = ref
        else:
            # Relative path: resolve against base_dir (where user ran g_code from)
            path = os.path.join(base_dir, ref)
        
        if not os.path.exists(path):
            continue

        if os.path.isfile(path):
            print(f"{CYAN}Loading context from file: {ref}{RESET}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Smart Truncation for @ references
                if len(content) > REF_READ_LIMIT:
                    content = content[:REF_READ_LIMIT] + f"\n... [CONTEXT TRUNCATED: {REF_READ_LIMIT} lines shown. File is large. Use the 'read' tool with 'offset' to inspect the rest.] ..."
                
                content = content.strip()
                extra_context.append(f"<file_context name=\"{ref}\">\n{content}\n</file_context>")
                
                parent_dir = os.path.dirname(path)
                if parent_dir and os.path.exists(parent_dir):
                    print(f"{CYAN}Analyzing structure of directory: {parent_dir}{RESET}")
                    try:
                        items = os.listdir(parent_dir)
                        items_str = "\n".join(items[:50])
                        if len(items) > 50:
                            items_str += "\n... [LIST TRUNCATED] ..."
                        extra_context.append(f"<dir_context name=\"{parent_dir}\">\n{items_str}\n</dir_context>")
                    except Exception as e:
                        pass
                        
            except Exception as e:
                print(f"{RED}Error reading {ref}: {e}{RESET}")
                
        elif os.path.isdir(path):
            print(f"{CYAN}Listing directory: {ref}{RESET}")
            try:
                items = os.listdir(path)
                items_str = "\n".join(items[:100])
                if len(items) > 100:
                    items_str += "\n... [LIST TRUNCATED] ..."
                extra_context.append(f"<dir_context name=\"{ref}\">\n{items_str}\n</dir_context>")
            except Exception as e:
                print(f"{RED}Error listing {ref}: {e}{RESET}")

    return text, extra_context


# --- Tool implementations ---

def read(args):
    # --- NEW: Resolve path relative to Base Dir ---
    base_dir = globals().get('SCRIPT_DIR')
    raw_path = args["path"]
    path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    
    try:
        # Load all lines into memory to allow for search/indexing
        lines = open(path, 'r', encoding='utf-8').readlines()
    except FileNotFoundError:
        return "error: file not found"
    except UnicodeDecodeError:
        return "error: file encoding issue (not utf-8)"

    # --- SMART SEARCH LOGIC ---
    search_term = args.get("search")
    
    if search_term:
        # Find the first occurrence of the search term
        found_line_idx = -1
        for idx, line in enumerate(lines):
            if search_term in line:
                found_line_idx = idx
                break
        
        if found_line_idx == -1:
            return f"error: search term '{search_term}' not found in file"
        
        # Center the read window around the found line
        # Default context window: 25 lines before, 25 lines after (50 total)
        context_buffer = 25 
        offset = max(0, found_line_idx - context_buffer)
        limit = 50 
    else:
        # Standard Manual Read
        offset = args.get("offset", 0)
        # Enforce Hard Cap to ensure chunking
        limit = args.get("limit", DEFAULT_READ_LIMIT)
        
    # STRICT CHUNKING: Never return more than MAX_READ_LINES_PER_CALL
    if limit > MAX_READ_LINES_PER_CALL:
        limit = MAX_READ_LINES_PER_CALL

    selected = lines[offset : offset + limit]
    
    hint = ""
    remaining_lines = len(lines) - (offset + limit)
    
    if remaining_lines > 0:
        next_offset = offset + limit
        hint = f"\n[... {remaining_lines} MORE LINES REMAINING ...]\n[Action Required: Use 'read' tool with 'offset={next_offset}' to read the next chunk.]"
        
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected)) + hint


def write(args):
    # --- NEW: Resolve path relative to Base Dir ---
    base_dir = globals().get('SCRIPT_DIR')
    raw_path = args["path"]
    path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        return f"error: Directory '{directory}' does not exist. Please create it using `bash mkdir -p {directory}` first."
    
    if "append_content" in args:
        chunk = args["append_content"]
        try:
            with open(path, "a", encoding='utf-8') as f:
                f.write(chunk)
            return "Appended content."
        except Exception as e:
            return f"error: {e}"
    else:
        content = args.get("content") or args.get("string")
        try:
            with open(path, "w", encoding='utf-8') as f:
                f.write(content)
            return "ok"
        except Exception as e:
            return f"error: {e}"


def copy_tool(args):
    # --- NEW: Resolve paths relative to Base Dir ---
    base_dir = globals().get('SCRIPT_DIR')
    raw_src = args.get("src")
    raw_dest = args.get("dest")
    
    src = raw_src if os.path.isabs(raw_src) else os.path.join(base_dir, raw_src)
    dest = raw_dest if os.path.isabs(raw_dest) else os.path.join(base_dir, raw_dest)
    
    if not src or not dest:
        return "error: missing 'src' or 'dest' parameters"
    
    if not os.path.exists(src):
        return f"error: source file '{src}' does not exist."
        
    dest_dir = os.path.dirname(dest)
    if dest_dir and not os.path.exists(dest_dir):
        return f"error: destination directory '{dest_dir}' does not exist. Please create it using `bash mkdir -p {dest_dir}` first."
        
    try:
        shutil.copy(src, dest)
        return f"Copied '{src}' to '{dest}'."
    except Exception as e:
        return f"error: {e}"


def move(args):
    # --- NEW: Resolve paths relative to Base Dir ---
    base_dir = globals().get('SCRIPT_DIR')
    raw_src = args.get("src")
    raw_dest = args.get("dest")
    
    src = raw_src if os.path.isabs(raw_src) else os.path.join(base_dir, raw_src)
    dest = raw_dest if os.path.isabs(raw_dest) else os.path.join(base_dir, raw_dest)
    
    if not src or not dest:
        return "error: missing 'src' or 'dest' parameters"
    
    if not os.path.exists(src):
        return f"error: source file '{src}' does not exist."
        
    dest_dir = os.path.dirname(dest)
    if dest_dir and not os.path.exists(dest_dir):
        return f"error: destination directory '{dest_dir}' does not exist. Please create it using `bash mkdir -p {dest_dir}` first."
        
    try:
        shutil.move(src, dest)
        return f"Moved '{src}' to '{dest}'."
    except Exception as e:
        return f"error: {e}"


def edit(args):
    # --- NEW: Resolve path relative to Base Dir ---
    base_dir = globals().get('SCRIPT_DIR')
    raw_path = args["path"]
    path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    
    try:
        text = open(path, 'r', encoding='utf-8').read()
    except FileNotFoundError:
        return "error: file not found"
    except UnicodeDecodeError:
        return "error: file encoding issue (not utf-8)"
        
    old = args.get("old")
    if old is None:
        old = args.get("old_string")
        
    new = args.get("new")
    if new is None:
        new = args.get("new_string")

    if old is None:
        return "error: missing 'old'/'old_string' parameter"
    if new is None: 
        return "error: missing 'new'/'new_string' parameter"
    
    if len(new) > 5000:
        print(f"{YELLOW}⚠ Warning: Large edit chunk ({len(new)} chars).{RESET}")

    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = (
        text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    )
    with open(path, "w", encoding='utf-8') as f:
        f.write(replacement)
    return "ok"


def glob(args):
    # --- NEW: Resolve path relative to Base Dir ---
    base_dir = globals().get('SCRIPT_DIR')
    raw_path = args.get("path", ".")
    path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    pattern = path + "/" + args["pattern"]
    
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"


def grep(args):
    # --- NEW: Resolve path relative to Base Dir ---
    base_dir = globals().get('SCRIPT_DIR')
    raw_path = args.get("path", ".")
    path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    pattern = re.compile(args["pattern"])
    hits = []
    for filepath in globlib.glob(path + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath, 'r', encoding='utf-8'), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except UnicodeDecodeError:
            pass # Skip binary files or non-utf8 files
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def bash(args):
    proc = subprocess.Popen(
        args["command"], shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    output_lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"  {DIM}│ {line.rstrip()}{RESET}", flush=True)
                output_lines.append(line)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        output_lines.append("\n(timed out after 30s)")
    return "".join(output_lines).strip() or "(empty)"


def finish(args):
    return "Task finished."


# --- Tool definitions: (description, schema, function) ---

TOOLS = {
    "read": (
        "Read file. Hard capped at 100 lines per call. "
        "Use 'offset' to paginate through the file. "
        "SMART READ: Use 'search' parameter to find a specific function/class (centers view around match).",
        {"path": "string", "offset": "number?", "limit": "number?", "search": "string?"},
        read,
    ),
    "write": (
        "Write content to file (Create/Overwrite). Use 'append_content' to build large files. "
        "CRITICAL: You MUST check if the file already exists. If it does, use 'edit' instead. "
        "You MUST also check if the directory exists. If not, use 'bash mkdir -p path'.",
        {"path": "string", "content": "string?", "append_content": "string?"},
        write,
    ),
    "edit": (
        "Replace 'old' with 'new'. "
        "IMPORTANT: Use keys 'old' and 'new' in JSON. To delete content, pass 'new' as an empty string \"\".",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "copy": (
        "Copy a file from 'src' to 'dest'. This is efficient for large files. "
        "CRITICAL: You MUST verify destination directory exists.",
        {"src": "string", "dest": "string"},
        copy_tool,
    ),
    "move": (
        "Move/Rename a file from 'src' to 'dest'. This is efficient for large files. "
        "CRITICAL: You MUST verify destination directory exists.",
        {"src": "string", "dest": "string"},
        move,
    ),
    "glob": (
        "Find files by pattern",
        {"pattern": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex",
        {"pattern": "string", "path": "string?"},
        grep,
    ),
    "bash": (
        "Run shell command",
        {"command": "string"},
        bash,
    ),
    "finish": (
        "MUST be called when task is COMPLETE. Stops agent.",
        {"summary": "string"},
        finish,
    ),
}


def run_tool(name, args):
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def make_schema():
    """Converts internal tool definitions to OpenAI function calling format."""
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        properties = {}
        required = []
        for param_name, param_type in params.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {
                "type": "integer" if base_type == "number" else base_type
            }
            if not is_optional:
                required.append(param_name)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return result


def call_api_stream(messages, system_prompt):
    """Handles API calls using OpenAI SDK with streaming support."""
    
    openai_messages = []
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            if any(block.get("type") == "tool_result" for block in content):
                for block in content:
                    if block["type"] == "tool_result":
                        tool_content = block.get("content", "")
                        if tool_content is None:
                            tool_content = ""
                        openai_messages.append({
                            "role": "user",
                            "content": tool_content
                        })
            else:
                text_parts = []
                tool_calls = []
                for block in content:
                    if block["type"] == "text":
                        text_parts.append(block["text"])
                    elif block["type"] == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"])
                            }
                        })
                
                content_str = "\n".join(text_parts)
                openai_messages.append({
                    "role": "assistant",
                    "content": content_str if content_str else "", 
                    "tool_calls": tool_calls if tool_calls else None
                })

    base_url = API_URL.replace("/chat/completions", "")
    
    client = OpenAI(api_key=API_KEY, base_url=base_url)
    
    payload_args = {
        "model": MODEL,
        "messages": openai_messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }
    
    if ENABLE_FUNCTION_CALLING:
        payload_args["tools"] = make_schema()
        payload_args["tool_choice"] = "auto"
    
    usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    accumulated_tool_calls = {} 
    accumulated_content = ""

    try:
        stream = client.chat.completions.create(**payload_args)
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                text_chunk = delta.content
                print(text_chunk, end="", flush=True)
                accumulated_content += text_chunk
            
            if ENABLE_FUNCTION_CALLING and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": None,
                            "arguments": ""
                        }
                    
                    tc = accumulated_tool_calls[idx]
                    
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                        
                    if hasattr(tc_delta, 'function') and tc_delta.function:
                        if tc_delta.function.name:
                            tc["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["arguments"] += tc_delta.function.arguments
            
            if chunk.choices[0].finish_reason is not None:
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_stats = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens
                    }
                break
                
    except Exception as e:
        # If user Ctrl+C's here (during API stream), we handle it in main loop
        # But we still need to print to clean up stream
        print() 
        raise

    content_blocks = []
    
    if accumulated_content:
        content_blocks.append({"type": "text", "text": accumulated_content})
        
    if ENABLE_FUNCTION_CALLING and accumulated_tool_calls:
        for idx in sorted(accumulated_tool_calls.keys()):
            tc = accumulated_tool_calls[idx]
            try:
                args = json.loads(tc["arguments"])
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": args
                })
            except json.JSONDecodeError:
                pass
    elif not ENABLE_FUNCTION_CALLING and accumulated_content:
        tag_pattern = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)
        matches = tag_pattern.findall(accumulated_content)
        for tool_name, args_str in matches:
            try:
                tool_args = json.loads(args_str.strip())
                synthetic_id = f"xml_call_{abs(hash(args_str))}"
                content_blocks.append({
                    "type": "tool_use",
                    "id": synthetic_id,
                    "name": tool_name,
                    "input": tool_args
                })
            except json.JSONDecodeError:
                pass

    return {"content": content_blocks, "usage": usage_stats}


def separator():
    return f"{DIM}{'─' * min(os.get_terminal_size().columns, 80)}{RESET}"


def main():
    # Allow modifying global variables inside this function
    global API_KEY, MODEL
    
    os_type = platform.system()
    fc_status = "Enabled" if ENABLE_FUNCTION_CALLING else "Disabled (XML Only)"
    
    # --- NEW: Set SCRIPT_DIR to Current Working Directory ---
    SCRIPT_DIR = os.getcwd()
    
    # Store globally for tools and completion to access
    globals()['SCRIPT_DIR'] = SCRIPT_DIR
    
    # Get Base Dir Name for display
    BASE_DIR_NAME = os.path.basename(SCRIPT_DIR)
    
    print(f"{BOLD}G Code{RESET} | {DIM}{MODEL} [BytArch] (FC: {fc_status}) | {BASE_DIR_NAME}{RESET} (CWD)\n")
    
    messages = []
    
    session = None
    if HAS_PROMPT_TOOLKIT:
        completer = PathCompleter()
        session = PromptSession(completer=completer)
    
    if ENABLE_FUNCTION_CALLING:
        tool_instruction = "Use JSON function calling."
        tool_examples = ""
    else:
        tool_instruction = (
            "You are in XML MODE. You MUST use XML tags to execute tools. "
            "Do not use JSON function calling. "
            "Use specific tool name as the tag."
        )
        tool_examples = (
            "\n"
            "XML Tool Call Examples:\n"
            "<bash>{\"command\": \"ls -la\"}</bash>\n"
            "<bash>{\"command\": \"mkdir -p new_folder\"}</bash>\n"
            "<read>{\"path\": \"file.py\"}</read>\n"
            "<read>{\"path\": \"file.py\", \"search\": \"def my_function\"}</read> (Smart Search)\n"
            "<edit>{\"path\": \"file.py\", \"old\": \"old text\", \"new\": \"new text\"}</edit>\n"
            "<edit>{\"path\": \"file.py\", \"old\": \"bad text\", \"new\": \"\"}</edit> (to delete)\n"
            "<write>{\"path\": \"newfile.html\", \"content\": \"<html>...</html>\"}</write>\n"
            "<write>{\"path\": \"newfile.html\", \"append_content\": \"<body>...</body>\"}</write>\n"
            "<copy>{\"src\": \"old_name.py\", \"dest\": \"new_name.py\"}</copy>\n"
            "<move>{\"src\": \"current_loc.py\", \"dest\": \"new_loc/current_loc.py\"}</move>\n"
            "<finish>{\"summary\": \"The file has been created successfully.\"}</finish>"
        )

    system_prompt = (
        f"Direct coding assistant. OS: {os_type}. CWD: {SCRIPT_DIR}.\n"
        "CRITICAL: CONSERVE TOKENS. BE EXTREMELY DIRECT AND BRIEF.\n"
        "1. NO PREAMBLE: Do not say 'Okay', 'Sure', 'I will', 'Here is code', etc.\n"
        "2. NO COMMENTARY: Do not explain what a tool did. Just proceed to next step.\n"
        "3. BE BRIEF: Keep any textual response to 1 sentence maximum.\n"
        "4. EXECUTE: If a tool is needed, call it immediately.\n"
        "5. STRUCTURAL AWARENESS (MANDATORY):\n"
        "   - When I mention a file (e.g., @youtube/lib.html), I have AUTOMATICALLY LOADED file content "
        "AND a directory listing of its parent folder.\n"
        "   - **YOU MUST EXAMINE THE PROVIDED DIRECTORY LISTING** to understand the project structure.\n"
        "   - Check which files exist in the folder before attempting 'write' or 'edit'.\n"
        "   - Use the directory structure to ensure you are targeting the correct file.\n"
        "6. BASE DIRECTORY (MANDATORY):\n"
        f"   - My Base Directory is: '{SCRIPT_DIR}'.\n"
        "   - This is the directory you ran this agent from (CWD).\n"
        "   - If you mention a file using a RELATIVE PATH (e.g., @youtube), I resolve it relative to Base Directory.\n"
        "   - If you mention an ABSOLUTE PATH (e.g., @C:/Users), I use it directly.\n"
        "   - To access a sibling folder, use '../sibling'.\n"
        "7. CHUNKED CREATION RULES (MANDATORY):\n"
        "   - NEVER write a large file (>50 lines) in a single call. IT WILL FAIL.\n"
        "   - CREATE PHASE 1: Use 'write' tool with 'content' parameter for the initial structure.\n"
        "   - BUILD PHASE 2: Use 'write' tool with 'append_content' parameter to add subsequent chunks.\n"
        "   - EXCEPTION: You can use <copy> or <move> to transfer large files efficiently.\n"
        "8. CHUNKED EDITING (MANDATORY):\n"
        "   - Use 'edit' on small sections (10-30 lines). Do not replace whole functions.\n"
        "9. INTELLIGENT READING (MANDATORY):\n"
        "   - **SMART READ**: When you need to inspect a specific function, class, or method, use the 'read' tool with 'search' parameter.\n"
        "   - Example: <read>{\"path\": \"app.py\", \"search\": \"def main\"}</read>\n"
        "   - The tool will automatically find the line and center the context (50 lines) around it.\n"
        "   - Use this instead of reading the whole file or guessing line numbers.\n"
        "10. PRE-WRITE/EDIT VERIFICATION (MANDATORY):\n"
        "   - BEFORE 'write' or 'edit': You MUST verify existence of the target file and directory.\n"
        "   - DIRECTORY CHECK: If the directory does not exist, create it immediately using: "
        "<bash>{\"command\": \"mkdir -p path/to/dir\"}</bash>\n"
        "   - FILE CHECK: Use <bash>{\"command\": \"ls filename\"}</bash> or similar to check if the file exists.\n"
        "   - IF THE FILE EXISTS: USE 'edit'. Do NOT use 'write' (it destroys data).\n"
        "   - IF THE FILE DOES NOT EXIST: USE 'write'.\n"
        "11. TOOL ARGUMENT NAMES (MANDATORY):\n"
        "   - For 'copy' and 'move': Use 'src' and 'dest'.\n"
        "   - For 'edit': You MUST use keys 'old' and 'new'. Do NOT use 'old_string' or 'new_string'.\n"
        "   - For 'write': You MUST use key 'content' (or 'append_content'). Do NOT use 'string'.\n"
        "12. ERROR HANDLING (MANDATORY):\n"
        "   - If a tool returns an 'error', STOP and ANALYZE.\n"
        "   - You MUST NOT blindly retry the exact same command.\n"
        "   - If 'old_string not found', use `read` with `search` to find the exact string.\n"
        "   - If 'directory not found', create it with mkdir.\n"
        "13. ANTI-LOOP PROTECTION (CRITICAL):\n"
        "   - **DO NOT REPEAT COMMANDS**. If you run the same tool with the same arguments twice in a row, the system will block it.\n"
        "   - Pay attention to tool outputs. If a tool succeeds, move to the next step. Do not re-run it to 'check'.\n"
        "14. .GCODE FOLDER & STANDARDS (CRITICAL):\n"
        "   - There is a `.gcode` folder in the project root containing `rules.md`, `context.md`, and `prompts.md`.\n"
        "   - It contains configuration files that define how you should write code.\n"
        "   - **STRICTLY ADHERE** to the coding standards defined in `.gcode/rules.md`.\n"
        "   - **UNDERSTAND** the project architecture from `.gcode/context.md`.\n"
        "   - **LEARN & UPDATE**: As you discover architectural details or patterns (e.g., 'this project uses MVC', 'this file connects to DB X'), "
        "YOU MUST UPDATE `.gcode/context.md` to document this knowledge so future sessions are consistent.\n"
        "15. SEQUENTIAL EXECUTION (MANDATORY):\n"
        "   - **CRITICAL**: Execute tools **ONE AT A TIME**.\n"
        "   - Wait for the tool result to complete before generating the next tool call.\n"
        "   - DO NOT output multiple <tool> tags in a single response unless you are 100% certain they are independent sequential steps (e.g., create folder -> write file).\n"
        "   - If you output multiple tools with the same name/arguments, the system will block the redundant ones.\n"
        "16. TASK COMPLETION (MANDATORY):\n"
        "   - YOU MUST CALL THE 'finish' TOOL WHEN THE TASK IS DONE.\n"
        "   - Do NOT stop the conversation by simply writing text. You MUST invoke 'finish'.\n"
        "----------------------------------------\n\n"
        f"{tool_instruction}\n"
        f"{tool_examples}"
    )

    # Helper to load project context from .gcode
    def load_gcode_context(base_dir):
        gcode_dir = os.path.join(base_dir, ".gcode")
        if not os.path.exists(gcode_dir):
            return None
        
        context_parts = []
        # We want to load rules first, then context, then prompts
        files_to_load = ['rules.md', 'context.md', 'prompts.md']
        
        for fname in files_to_load:
            fpath = os.path.join(gcode_dir, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content:
                            # Tag the content based on file name
                            tag = fname.replace('.md', '')
                            context_parts.append(f"<gcode_{tag}>\n{content}\n</gcode_{tag}>")
                except Exception as e:
                    pass
        
        if context_parts:
            return "\n\n".join(context_parts)
        return None

    # Load initial project context if available
    initial_context = load_gcode_context(SCRIPT_DIR)
    if initial_context:
        messages.append({"role": "system", "content": initial_context})

    while True:
        try:
            print(separator())
            if HAS_PROMPT_TOOLKIT and session:
                # FIX: Use plain text for prompt to avoid Windows ANSI errors
                user_input = session.prompt("❯ ")
            else:
                user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
                
            print(separator())
            if not user_input:
                continue
            if user_input in ("/q", "exit"):
                break
            if user_input == "/c":
                messages = []
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                # Reload context after clearing
                initial_context = load_gcode_context(SCRIPT_DIR)
                if initial_context:
                    messages.append({"role": "system", "content": initial_context})
                continue
            
            # --- NEW: /init COMMAND (Enhanced) ---
            if user_input == "/init":
                gcode_dir = os.path.join(SCRIPT_DIR, ".gcode")
                if not os.path.exists(gcode_dir):
                    os.makedirs(gcode_dir)
                    print(f"{GREEN}✔ Created .gcode folder{RESET}")
                else:
                    print(f"{YELLOW}✔ .gcode folder already exists{RESET}")

                # Create rules.md
                rules_file = os.path.join(gcode_dir, "rules.md")
                if not os.path.exists(rules_file):
                    with open(rules_file, "w", encoding="utf-8") as f:
                        f.write("""# Coding Rules
- Follow existing code style.
- Use logging instead of print. Include error context.
- Add type hints and docstrings for public functions.
- Keep functions small; prefer composition over long scripts.
- Write safe defaults. Handle timeouts and retries where external calls exist.
# Tests
- Provide a minimal test when adding new modules.
- Use fakes or fixtures; do not call real services.
# Security
- Never include secrets in code or examples.
- Use environment variables or placeholders like <API_KEY>.
""")
                    print(f"{GREEN}✔ Created .gcode/rules.md{RESET}")
                else:
                    print(f"{DIM}✔ .gcode/rules.md already exists.{RESET}")

                # Create context.md
                context_file = os.path.join(gcode_dir, "context.md")
                if not os.path.exists(context_file):
                    with open(context_file, "w", encoding="utf-8") as f:
                        f.write("""# Project Context
This is a [Project Type/Description].

## Main Components
- [Component 1]
- [Component 2]

## Tech Stack
- Runtime: [e.g., Python 3.11, Node 20]
- Dependencies: [List key packages]
- Tools: [Docker, Makefile, etc.]

## Conventions
- Config via environment variables.
- Error handling with structured logs.
- CI runs tests and lint on every PR.
""")
                    print(f"{GREEN}✔ Created .gcode/context.md{RESET}")
                else:
                    print(f"{DIM}✔ .gcode/context.md already exists.{RESET}")

                # Create prompts.md
                prompts_file = os.path.join(gcode_dir, "prompts.md")
                if not os.path.exists(prompts_file):
                    with open(prompts_file, "w", encoding="utf-8") as f:
                        f.write("""# Reusable Prompts
## Add a module
Create a new module that does X. Include:
- A clear, typed interface
- Error handling and logging
- A small unit test with a fake

## Improve performance
Review this function for bottlenecks. Propose changes.
Explain trade-offs in 3-5 bullet points.

## Write docs
Draft README instructions for running the project locally:
- Prerequisites
- Setup
- Common commands
- How to run tests
""")
                    print(f"{GREEN}✔ Created .gcode/prompts.md{RESET}")
                else:
                    print(f"{DIM}✔ .gcode/prompts.md already exists.{RESET}")
                
                print(f"{DIM}  Please fill out these files to guide the AI agent.{RESET}")
                continue
            
            # --- NEW: CONFIGURATION COMMANDS ---
            if user_input.startswith("/model "):
                parts = user_input.split(" ", 1)
                if len(parts) > 1:
                    MODEL = parts[1].strip()
                    save_user_config(API_KEY, MODEL)
                    print(f"{GREEN}✔ Model updated to: {MODEL}{RESET}")
                    print(f"{DIM}  Saved to {CONFIG_FILE}{RESET}")
                else:
                    print(f"{YELLOW}Usage: /model <model_name>{RESET}")
                continue
                
            if user_input.startswith("/key "):
                parts = user_input.split(" ", 1)
                if len(parts) > 1:
                    API_KEY = parts[1].strip()
                    save_user_config(API_KEY, MODEL)
                    print(f"{GREEN}✔ API Key updated.{RESET}")
                    print(f"{DIM}  Saved to {CONFIG_FILE}{RESET}")
                else:
                    print(f"{YELLOW}Usage: /key <api_key>{RESET}")
                continue

            # --- RESOLVE @ REFERENCES ---
            # Pass SCRIPT_DIR so it resolves relative paths correctly
            processed_input, context_blocks = resolve_references(user_input, SCRIPT_DIR)
                
            for block in context_blocks:
                messages.append({"role": "system", "content": block})
                
            user_input = processed_input

            messages.append({"role": "user", "content": user_input})

            # --- AGENT LOOP WITH INTERRUPT SUPPORT ---
            
            # Track consecutive failures to stop infinite loops
            consecutive_failures = 0
            # Track repetitive tool calls to stop "stuck" loops
            last_tool_signature = None
            repeat_counter = 0
            
            # Track the VERY LAST tool executed to prevent hallucinated duplicates in one stream
            last_executed_tool = None
            last_executed_args = None

            while True:
                try:
                    # Context Management
                    history_usage = count_messages_tokens(messages)
                    system_usage = estimate_tokens(system_prompt)
                    total_est = history_usage + system_usage
                    limit = CONTEXT_WINDOW - MAX_TOKENS - 1000
                    
                    if total_est > limit:
                        print(f"{YELLOW}⏺ Context window ({CONTEXT_WINDOW}) near limit ({total_est}). Pruning old messages...{RESET}")
                        while messages and (count_messages_tokens(messages) + system_usage) > limit:
                            messages.pop(0)

                    # API Call (Streaming)
                    response = call_api_stream(messages, system_prompt)
                    
                    content_blocks = response.get("content", [])
                    usage = response.get("usage", {})
                    
                    # Display Usage
                    prompt_t = usage.get("prompt_tokens", "?")
                    completion_t = usage.get("completion_tokens", "?")
                    total_t = usage.get("total_tokens", "?")
                    print(f"{DIM}\n  Tokens: {prompt_t} + {completion_t} = {total_t}{RESET}\n")
                    
                    tool_results = []
                    task_finished = False

                    for block in content_blocks:
                        if block["type"] == "text":
                            pass 

                        if block["type"] == "tool_use":
                            tool_name = block["name"]
                            tool_args = block["input"]
                            
                            if tool_name == "finish":
                                summary = tool_args.get("summary", "Task completed.")
                                print(f"{GREEN}✔ Task Complete:{RESET} {summary}\n")
                                task_finished = True
                                break

                            # --- NEW: STRICT SEQUENTIAL EXECUTION CHECK ---
                            # Block execution if the AI tries to run the EXACT same tool call 
                            # immediately after just running it (hallucinated urgency).
                            # We compare stringified args for easy comparison
                            current_sig_str = json.dumps(tool_args, sort_keys=True)
                            if (last_executed_tool == tool_name and 
                                last_executed_args == current_sig_str):
                                # We have a duplicate. Refuse execution.
                                print(f"{RED}✗ Redundant tool call blocked: {tool_name}{RESET}")
                                print(f"{YELLOW}  Reason: You just executed this command. Wait for the result before proceeding.{RESET}")
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block["id"],
                                    "content": "error: Redundant tool call detected. Wait for previous result before proceeding."
                                })
                                continue # Skip execution, but still loop to process next blocks

                            # Update last executed tracker
                            last_executed_tool = tool_name
                            last_executed_args = current_sig_str

                            # Normal Tool Handling
                            arg_preview = str(list(tool_args.values())[0])[:50]
                            
                            result = run_tool(tool_name, tool_args)
                            
                            # --- FIX: STRICT ERROR CHECKING ---
                            # We use .strip().startswith("error:") to avoid false positives 
                            # if the code content itself contains the word "error".
                            if result.strip().startswith("error:"):
                                consecutive_failures += 1
                                
                                print(f"{GREEN}⏺ {tool_name.capitalize()}{RESET}({DIM}{arg_preview}{RESET})")
                                print(f"{RED}  ✗ {result}{RESET}")
                                
                                # CHECK FAILURE COUNT
                                if consecutive_failures >= 3:
                                    print(f"{RED}\n⏺ AGENT STOPPED: Tool failed 3 times consecutively.{RESET}")
                                    print(f"{YELLOW}Reason: {result}{RESET}")
                                    print(f"{YELLOW}Action: Please correct the issue or provide new instructions.{RESET}")
                                    task_finished = True # Force stop
                                    break # Break tool loop
                                
                                # Injection for retry attempt (1st or 2nd time)
                                injection = (
                                    "\n\n[SYSTEM ALERT]: The previous tool call FAILED. "
                                    "You MUST NOT repeat the exact same command. "
                                    "Analyze the error above. "
                                    "If 'Directory not found', run `mkdir -p`. "
                                    "If 'old_string not found', use `read` with `search` to verify the EXACT string. "
                                    "If it was a context/length error, break the task into smaller chunks. "
                                    "You must try a DIFFERENT approach."
                                )
                                
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block["id"],
                                    "content": result + injection
                                })
                            else:
                                # Success: Reset failure counter
                                consecutive_failures = 0
                                
                                print(
                                    f"{GREEN}⏺ {tool_name.capitalize()}{RESET}({DIM}{arg_preview}{RESET})"
                                )
                                result_lines = result.split("\n")
                                preview = result_lines[0][:60]
                                if len(result_lines) > 1:
                                    preview += f" ... +{len(result_lines) - 1} lines"
                                elif len(result_lines[0]) > 60:
                                    preview += "..."
                                print(f"  {DIM}⎿  {preview}{RESET}")

                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block["id"],
                                    "content": result,
                                })

                    if task_finished:
                        break

                    messages.append({"role": "assistant", "content": content_blocks})

                    if not tool_results:
                        break
                    messages.append({"role": "user", "content": tool_results})

                except KeyboardInterrupt:
                    print(f"\n{YELLOW}[Interrupted]{RESET} User stopped the agent. Returning to main prompt...")
                    break
                
                # If task_finished was set due to 3 failures or loops, break agent loop
                if task_finished:
                    break

            print()

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")


if __name__ == "__main__":
    main()