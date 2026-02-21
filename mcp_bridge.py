import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uvicorn

# markitdown for OCR / document extraction
from markitdown import MarkItDown

# mcp client for tool usage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = FastAPI(title="ANE Utility Bridge (MCP & OCR)")

# MCP Configuration paths
CONFIG_PATH = Path("mcp_config.json")

# In-memory store of active MCP sessions
active_sessions: Dict[str, ClientSession] = {}
exit_stacks: Dict[str, Any] = {}

class MCPRequest(BaseModel):
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    # Load and initialize requested standard MCP servers if configured
    if not CONFIG_PATH.exists():
        # Auto-create default config
        default_config = {
            "mcpServers": {
                "brave-search": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": ""}
                }
            }
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(default_config, f, indent=2)
        print("Created default mcp_config.json")
    
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            
        mcp_servers = config.get("mcpServers", {})
        for name, details in mcp_servers.items():
            if ":" in name: continue # skip things mapping to urls or similar
            command = details.get("command")
            args = details.get("args", [])
            env = details.get("env", {})
            full_env = {**os.environ.copy(), **env}
            
            # Note: We won't block startup on MCPs booting. We will boot them on demand or asynchronously.
            # For simplicity, we'll initialize them gracefully when requested via the `/mcp/tools` endpoint.
    except Exception as e:
        print(f"Failed to load MCP config: {e}")

async def get_mcp_session(server_name: str) -> ClientSession:
    if server_name in active_sessions:
        return active_sessions[server_name]
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        
    mcp_servers = config.get("mcpServers", {})
    if server_name not in mcp_servers:
        raise HTTPException(status_code=404, detail=f"MCP server {server_name} not found in config")
        
    details = mcp_servers[server_name]
    env = {**os.environ.copy(), **details.get("env", {})}
    
    server_params = StdioServerParameters(
        command=details.get("command"),
        args=details.get("args", []),
        env=env
    )
    
    from contextlib import AsyncExitStack
    exit_stack = AsyncExitStack()
    
    try:
        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        
        active_sessions[server_name] = session
        exit_stacks[server_name] = exit_stack
        return session
    except Exception as e:
        await exit_stack.aclose()
        raise HTTPException(status_code=500, detail=f"Failed to initialize MCP {server_name}: {str(e)}")


@app.get("/mcp/list")
async def list_mcp_tools():
    """List all configured MCP servers and their available tools"""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        
    result = {}
    mcp_servers = config.get("mcpServers", {})
    
    for server_name in mcp_servers.keys():
        try:
            session = await get_mcp_session(server_name)
            try:
                tools_response = await session.list_tools()
                tools = [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools_response.tools]
                result[server_name] = tools
            except Exception as inner_e:
                print(f"Error listing tools for {server_name}: {inner_e}")
                result[server_name] = {"error": f"Tool list failed: {inner_e}"}
        except Exception as e:
            result[server_name] = {"error": str(e)}
            
    return result

@app.post("/mcp/call")
async def call_mcp_tool(req: MCPRequest):
    """Execute an MCP Tool"""
    session = await get_mcp_session(req.server_name)
    try:
        result = await session.call_tool(req.tool_name, arguments=req.arguments)
        
        # Serialize the CallToolResult to dict
        if hasattr(result, "content"):
            content = []
            for item in result.content:
                if item.type == "text":
                    content.append({"type": "text", "text": item.text})
                else:
                    content.append({"type": item.type})
            return {"content": content, "isError": result.isError}
            
        return {"error": "Unknown response format"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accepts a file, extracts OCR/Text via MarkItDown, and returns markdown content"""
    tmp_path = Path("/tmp") / file.filename
    try:
        code_content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(code_content)
            
        md = MarkItDown()
        result = md.convert(str(tmp_path))
        
        if not result or not result.text_content:
            return {"markdown": "", "error": "Could not extract text content"}
            
        return {"markdown": result.text_content}
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = 11437
    print(f"Starting ANE Utility Bridge on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
