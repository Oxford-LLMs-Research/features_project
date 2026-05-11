import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { Agent, Cursor, CursorAgentError } from "@cursor/sdk";

function parseArgs(argv) {
  const defaults = {
    rounds: 3,
    model: "composer-2",
    cwd: process.cwd(),
    currentStatePath: "paper/current_state.tex",
    researchDesignPath: "paper/research_design.tex",
    findingsPath: "analysis/prelim_findings.md",
    outputPath: "paper/prelim_paper.tex",
    reviewPath: "paper/review_notes.md",
  };

  const args = { ...defaults };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === "--rounds" && next) {
      args.rounds = Number.parseInt(next, 10);
      i += 1;
    } else if (arg === "--model" && next) {
      args.model = next;
      i += 1;
    } else if (arg === "--current-state" && next) {
      args.currentStatePath = next;
      i += 1;
    } else if (arg === "--research-design" && next) {
      args.researchDesignPath = next;
      i += 1;
    } else if (arg === "--findings" && next) {
      args.findingsPath = next;
      i += 1;
    } else if (arg === "--output" && next) {
      args.outputPath = next;
      i += 1;
    } else if (arg === "--review-notes" && next) {
      args.reviewPath = next;
      i += 1;
    }
  }
  return args;
}

async function readOptionalFile(filePath) {
  try {
    return await fs.readFile(filePath, "utf8");
  } catch {
    return "";
  }
}

async function resolveCursorApiKey(cwd) {
  if (process.env.CURSOR_API_KEY) {
    return process.env.CURSOR_API_KEY;
  }

  const envFilePath = path.resolve(cwd, ".env");
  const envRaw = await readOptionalFile(envFilePath);
  if (!envRaw) return undefined;

  for (const line of envRaw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const idx = trimmed.indexOf("=");
    if (idx < 0) continue;
    const key = trimmed.slice(0, idx).trim();
    if (key !== "CURSOR_API_KEY") continue;
    let value = trimmed.slice(idx + 1).trim();
    if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (value) return value;
  }
  return undefined;
}

function extractBetween(text, startMarker, endMarker) {
  const startIdx = text.indexOf(startMarker);
  const endIdx = text.indexOf(endMarker);
  if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) {
    return "";
  }
  const innerStart = startIdx + startMarker.length;
  return text.slice(innerStart, endIdx).trim();
}

function normalizeRoundCount(rounds) {
  if (!Number.isFinite(rounds) || rounds < 1) return 1;
  if (rounds > 10) return 10;
  return rounds;
}

async function disposeAgent(agent) {
  if (!agent) return;

  const asyncDisposer = agent[Symbol.asyncDispose];
  if (typeof asyncDisposer === "function") {
    await asyncDisposer.call(agent);
    return;
  }

  if (typeof agent.dispose === "function") {
    await agent.dispose();
    return;
  }

  if (typeof agent.close === "function") {
    await agent.close();
  }
}

async function main() {
  const config = parseArgs(process.argv);
  config.rounds = normalizeRoundCount(config.rounds);
  const apiKey = await resolveCursorApiKey(config.cwd);

  if (!apiKey) {
    console.error("Missing CURSOR_API_KEY. Set it in environment or .env before running.");
    process.exitCode = 1;
    return;
  }

  const currentStateFullPath = path.resolve(config.cwd, config.currentStatePath);
  const researchDesignFullPath = path.resolve(config.cwd, config.researchDesignPath);
  const findingsFullPath = path.resolve(config.cwd, config.findingsPath);
  const outputFullPath = path.resolve(config.cwd, config.outputPath);
  const reviewFullPath = path.resolve(config.cwd, config.reviewPath);
  const roundsDir = path.resolve(config.cwd, "outputs", "adversarial_rounds");

  const [currentState, researchDesign, findings] = await Promise.all([
    fs.readFile(currentStateFullPath, "utf8"),
    fs.readFile(researchDesignFullPath, "utf8"),
    readOptionalFile(findingsFullPath),
  ]);

  await fs.mkdir(roundsDir, { recursive: true });
  await fs.mkdir(path.dirname(outputFullPath), { recursive: true });
  await fs.mkdir(path.dirname(reviewFullPath), { recursive: true });

  const sharedOptions = {
    apiKey,
    model: { id: config.model },
    local: { cwd: config.cwd },
  };

  const writer = await Agent.create(sharedOptions);
  const critic = await Agent.create(sharedOptions);

  let workingDraft = currentState;
  const reviewLog = [];

  try {
    await Cursor.me({ apiKey });

    for (let round = 1; round <= config.rounds; round += 1) {
      const previousReview = round === 1 ? "No previous review." : reviewLog[reviewLog.length - 1];
      const writerPrompt = `
You are the Writer agent.
Task: produce an improved LaTeX draft for a preliminary paper by fusing:
1) current state report
2) research design
3) prelim findings (if present)

Round: ${round}/${config.rounds}

Hard constraints:
- Keep LaTeX compile-safe.
- Do not invent empirical results.
- Keep citations and labels intact when possible.
- Address prior critic feedback point-by-point.

Return EXACTLY in this format:
BEGIN_DRAFT
<full revised LaTeX draft>
END_DRAFT
BEGIN_CHANGELOG
<short bullet list of what you changed and why>
END_CHANGELOG

Previous critic review:
${previousReview}

Current draft input:
${workingDraft}

Research design source:
${researchDesign}

Prelim findings source (can be empty):
${findings}
`.trim();

      const writerRun = await writer.send(writerPrompt);
      const writerResult = await writerRun.wait();
      if (writerResult.status !== "finished") {
        throw new Error(`Writer failed in round ${round}: ${writerResult.status}`);
      }

      const writerText = writerResult.result ?? "";
      const nextDraft = extractBetween(writerText, "BEGIN_DRAFT", "END_DRAFT");
      const changelog = extractBetween(writerText, "BEGIN_CHANGELOG", "END_CHANGELOG");
      if (!nextDraft) {
        throw new Error(`Writer did not return draft markers in round ${round}.`);
      }

      workingDraft = nextDraft;
      const roundDraftPath = path.join(roundsDir, `prelim_paper.round_${round}.tex`);
      await fs.writeFile(roundDraftPath, workingDraft, "utf8");

      const criticPrompt = `
You are the Critic reviewer agent.
Evaluate the draft and provide high-signal review only.

Round: ${round}/${config.rounds}

Rubric (1-5):
- fidelity_to_research_design
- evidence_claim_alignment
- methodological_clarity
- internal_consistency
- writing_precision
- latex_structure

Output EXACTLY this schema:
BEGIN_REVIEW
overall_gate: PASS or FAIL
scores:
- fidelity_to_research_design: <1-5>
- evidence_claim_alignment: <1-5>
- methodological_clarity: <1-5>
- internal_consistency: <1-5>
- writing_precision: <1-5>
- latex_structure: <1-5>
top_issues:
- <severity:high|medium|low> <issue>
- <severity:high|medium|low> <issue>
- <severity:high|medium|low> <issue>
required_fixes:
- <actionable request>
- <actionable request>
- <actionable request>
END_REVIEW

Writer changelog:
${changelog || "No changelog provided."}

Draft to review:
${workingDraft}
`.trim();

      const criticRun = await critic.send(criticPrompt);
      const criticResult = await criticRun.wait();
      if (criticResult.status !== "finished") {
        throw new Error(`Critic failed in round ${round}: ${criticResult.status}`);
      }

      const criticText = criticResult.result ?? "";
      const review = extractBetween(criticText, "BEGIN_REVIEW", "END_REVIEW") || criticText;
      reviewLog.push(`## Round ${round}\n\n${review.trim()}\n`);

      await fs.writeFile(reviewFullPath, reviewLog.join("\n"), "utf8");
      const roundReviewPath = path.join(roundsDir, `review.round_${round}.md`);
      await fs.writeFile(roundReviewPath, review, "utf8");

      if (/\boverall_gate:\s*PASS\b/i.test(review)) {
        break;
      }
    }

    await fs.writeFile(outputFullPath, workingDraft, "utf8");
    console.log(`Finished adversarial loop. Final draft: ${outputFullPath}`);
    console.log(`Review notes: ${reviewFullPath}`);
    console.log(`Round artifacts: ${roundsDir}`);
  } catch (err) {
    if (err instanceof CursorAgentError) {
      console.error(`Cursor SDK startup error: ${err.message}`);
      process.exitCode = 1;
      return;
    }
    if (err && typeof err === "object" && "cause" in err) {
      const cause = err.cause;
      if (cause && typeof cause === "object" && "code" in cause && cause.code === "unauthenticated") {
        console.error("Cursor authentication failed. Verify CURSOR_API_KEY is valid for this account.");
        process.exitCode = 1;
        return;
      }
    }
    throw err;
  } finally {
    await Promise.allSettled([disposeAgent(writer), disposeAgent(critic)]);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
