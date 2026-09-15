/**
 * index.ts - the Cloudflare Worker in front of the Python agent.
 *
 * What it does, in order, for every request:
 *   1. Ask for a password (HTTP Basic auth) if APP_PASSWORD is set.
 *   2. Serve the static site (course + chat page) for every path except
 *      the three agent routes: /chat, /state, /reset.
 *   3. For those three, find this browser's sandbox (a cookie names it),
 *      make sure the Python server is running inside it, and forward the
 *      request to it unchanged. The response streams straight back.
 *
 * The Python code is NOT rewritten for the cloud. It is copied into the
 * sandbox from src/bundle.ts (see bundle.py) and started with the exact
 * command you use on your laptop:  python3 -m agent.server
 */

import { getSandbox, type Sandbox as SandboxType } from "@cloudflare/sandbox";
import { FILES } from "./bundle";

export { Sandbox } from "@cloudflare/sandbox";

type Env = {
  Sandbox: DurableObjectNamespace<SandboxType>;
  ASSETS: Fetcher;
  APP_PASSWORD?: string;
  LLM_MODEL?: string;
};

const AGENT_PORT = 8765;
const AGENT_ROUTES = new Set(["/chat", "/state", "/reset"]);
const CODE_DIR = "/workspace/blueprint";
const WORKSPACE_DIR = "/workspace/project";
const COOKIE = "agent_session";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // The course page is public. The chat page and the agent routes are
    // private, because a sandbox costs money and runs whatever the model asks.
    const isPrivate = AGENT_ROUTES.has(url.pathname) || url.pathname.startsWith("/app");
    if (isPrivate) {
      const denied = requireAuth(request, env);
      if (denied) return denied;
    }

    if (!AGENT_ROUTES.has(url.pathname)) {
      return env.ASSETS.fetch(request);
    }

    // One sandbox per browser. The cookie is the only thing that links them.
    let session = readCookie(request, COOKIE);
    let isNewSession = false;
    if (!session) {
      session = crypto.randomUUID();
      isNewSession = true;
    }
    const sandbox = getSandbox(env.Sandbox, `session-${session}`, { sleepAfter: "15m" });

    try {
      await ensureAgentRunning(sandbox, env);
    } catch (error) {
      return Response.json(
        { error: `The sandbox could not start the agent: ${(error as Error).message}` },
        { status: 503 },
      );
    }

    const upstream = await sandbox.containerFetch(request, AGENT_PORT);
    if (!isNewSession) return upstream;

    // First contact: hand the browser its session cookie.
    const response = new Response(upstream.body, upstream);
    response.headers.append(
      "Set-Cookie",
      `${COOKIE}=${session}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=86400`,
    );
    return response;
  },
};

/** Start the Python server inside the sandbox if it is not already answering. */
async function ensureAgentRunning(sandbox: SandboxType, env: Env): Promise<void> {
  if (await isAgentUp(sandbox)) return;

  // Copy the package in. Cheap, and idempotent: a wake-up after sleep does it again.
  const dirs = new Set<string>();
  for (const path of Object.keys(FILES)) {
    const folder = path.substring(0, path.lastIndexOf("/"));
    if (folder) dirs.add(`${CODE_DIR}/${folder}`);
  }
  for (const dir of dirs) await sandbox.mkdir(dir, { recursive: true });
  await sandbox.mkdir(WORKSPACE_DIR, { recursive: true });
  for (const [path, text] of Object.entries(FILES)) {
    await sandbox.writeFile(`${CODE_DIR}/${path}`, text);
  }

  await sandbox.startProcess("python3 -m agent.server", {
    cwd: CODE_DIR,
    processId: "agent-server",
    autoCleanup: false,
    env: {
      AGENT_HOST: "0.0.0.0",
      AGENT_PORT: String(AGENT_PORT),
      AGENT_WORKSPACE: WORKSPACE_DIR,
      AGENT_ALLOWED_ORIGINS: "",
      LLM_MODEL: env.LLM_MODEL ?? "openai/gpt-5-mini",
      PYTHONUNBUFFERED: "1",
    },
  });
  // Poll until the server answers (Python takes a second or two to start).
  const deadline = Date.now() + 45_000;
  while (Date.now() < deadline) {
    if (await isAgentUp(sandbox)) return;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("python3 -m agent.server did not start within 45 seconds");
}

async function isAgentUp(sandbox: SandboxType): Promise<boolean> {
  try {
    const probe = await sandbox.containerFetch(
      `http://localhost:${AGENT_PORT}/state`,
      { method: "GET" },
      AGENT_PORT,
    );
    return probe.ok;
  } catch {
    return false;
  }
}

/** HTTP Basic auth, only when APP_PASSWORD is configured. Username is free-form. */
function requireAuth(request: Request, env: Env): Response | null {
  if (!env.APP_PASSWORD) return null;
  const header = request.headers.get("Authorization") ?? "";
  const [scheme, encoded] = header.split(" ");
  if (scheme === "Basic" && encoded) {
    const decoded = atob(encoded);
    const password = decoded.substring(decoded.indexOf(":") + 1);
    if (timingSafeEqual(password, env.APP_PASSWORD)) return null;
  }
  return new Response("This page is private. Enter the password.", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="coding-agent-blueprint", charset="UTF-8"' },
  });
}

function timingSafeEqual(a: string, b: string): boolean {
  const encoder = new TextEncoder();
  const bytesA = encoder.encode(a);
  const bytesB = encoder.encode(b);
  if (bytesA.byteLength !== bytesB.byteLength) return false;
  return crypto.subtle.timingSafeEqual(bytesA, bytesB);
}

function readCookie(request: Request, name: string): string | null {
  const header = request.headers.get("Cookie") ?? "";
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return null;
}
