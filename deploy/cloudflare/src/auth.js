/**
 * auth.js - the small, testable pieces of the Worker's front door.
 *
 * Plain JavaScript on purpose: `node --test test/` runs these without a
 * build step. index.ts imports them.
 */

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * HTTP Basic auth. Returns null when the request may pass, otherwise the
 * Response to send instead. With no password configured the door stays
 * SHUT (503), never open.
 * @param {Request} request
 * @param {string | undefined} password
 * @returns {Response | null}
 */
export function requireAuth(request, password) {
  if (!password) {
    return Response.json(
      { error: "APP_PASSWORD is not configured. Run: wrangler secret put APP_PASSWORD" },
      { status: 503 },
    );
  }
  const header = request.headers.get("Authorization") ?? "";
  const [scheme, encoded] = header.split(" ");
  if (scheme === "Basic" && encoded) {
    let decoded = "";
    try {
      decoded = atob(encoded);
    } catch {
      decoded = "";
    }
    const supplied = decoded.substring(decoded.indexOf(":") + 1);
    if (decoded.includes(":") && constantTimeEqual(supplied, password)) return null;
  }
  return new Response("This page is private. Enter the password.", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="coding-agent-blueprint", charset="UTF-8"' },
  });
}

/**
 * Compare two strings without leaking, through timing, where they differ.
 * @param {string} a
 * @param {string} b
 */
export function constantTimeEqual(a, b) {
  const encoder = new TextEncoder();
  const bytesA = encoder.encode(a);
  const bytesB = encoder.encode(b);
  let difference = bytesA.length ^ bytesB.length;
  const length = Math.max(bytesA.length, bytesB.length);
  for (let i = 0; i < length; i++) {
    difference |= (bytesA[i] ?? 0) ^ (bytesB[i] ?? 0);
  }
  return difference === 0;
}

/**
 * A session id is only trusted if it looks like the UUID we issued.
 * @param {string | null} value
 */
export function isValidSessionId(value) {
  return typeof value === "string" && UUID.test(value);
}

/**
 * @param {Request} request
 * @param {string} name
 * @returns {string | null}
 */
export function readCookie(request, name) {
  const header = request.headers.get("Cookie") ?? "";
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return null;
}
