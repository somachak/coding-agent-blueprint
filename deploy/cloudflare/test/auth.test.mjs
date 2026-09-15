import { test } from "node:test";
import assert from "node:assert/strict";
import { constantTimeEqual, isValidSessionId, readCookie, requireAuth } from "../src/auth.js";

const basic = (user, pass) => "Basic " + Buffer.from(`${user}:${pass}`).toString("base64");
const req = (headers = {}) => new Request("https://example.test/app/", { headers });

test("no password configured fails closed with 503", () => {
  const response = requireAuth(req({ Authorization: basic("soma", "anything") }), undefined);
  assert.equal(response.status, 503);
});

test("missing header gets 401 with a Basic challenge", () => {
  const response = requireAuth(req(), "secret");
  assert.equal(response.status, 401);
  assert.match(response.headers.get("WWW-Authenticate"), /Basic/);
});

test("wrong password gets 401", () => {
  assert.equal(requireAuth(req({ Authorization: basic("x", "nope") }), "secret").status, 401);
});

test("right password passes, any username", () => {
  assert.equal(requireAuth(req({ Authorization: basic("whoever", "secret") }), "secret"), null);
});

test("malformed base64 and missing colon are refused, not thrown", () => {
  assert.equal(requireAuth(req({ Authorization: "Basic %%%not-base64%%%" }), "secret").status, 401);
  const noColon = "Basic " + Buffer.from("secret").toString("base64");
  assert.equal(requireAuth(req({ Authorization: noColon }), "secret").status, 401);
});

test("constantTimeEqual compares whole strings", () => {
  assert.equal(constantTimeEqual("abc", "abc"), true);
  assert.equal(constantTimeEqual("abc", "abd"), false);
  assert.equal(constantTimeEqual("abc", "ab"), false);
  assert.equal(constantTimeEqual("", ""), true);
});

test("session ids must be UUIDs", () => {
  assert.equal(isValidSessionId("3f2c1a9e-0b7d-4c5a-9e8f-1a2b3c4d5e6f"), true);
  assert.equal(isValidSessionId("../../etc"), false);
  assert.equal(isValidSessionId(""), false);
  assert.equal(isValidSessionId(null), false);
});

test("readCookie finds the named cookie only", () => {
  const request = req({ Cookie: "other=1; agent_session=abc=def; last=2" });
  assert.equal(readCookie(request, "agent_session"), "abc=def");
  assert.equal(readCookie(request, "missing"), null);
});
