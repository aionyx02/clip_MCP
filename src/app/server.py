from fastmcp import FastMCP

mcp = FastMCP("clip_server")

@mcp.tool()
def add_number(a: float, b: float) -> float:
    return a + b

if __name__ == "__main__":
    mcp.run()