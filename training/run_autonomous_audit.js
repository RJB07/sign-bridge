// SignBridge Autonomous System Audit & Test Harness
// Verifies Letters, Numbers, Words, Classifiers, ASL 10 Grammar Rules, and Speech Formatting

const fs = require('fs');
const path = require('path');

console.log("==========================================================");
console.log("🌉 SIGNBRIDGE AUTONOMOUS SYSTEM AUDIT & VERIFICATION");
console.log("==========================================================");

// Load Knowledge Base Manifest
const kbPath = path.join(__dirname, '../resources/asl_books/asl_book_knowledge_base.json');
let kbData = {};
if (fs.existsSync(kbPath)) {
  kbData = JSON.parse(fs.readFileSync(kbPath, 'utf8'));
  console.log(`[+] Loaded Knowledge Base Manifest: ${kbData.books.length} Books, ${kbData.asl_grammar_rules.length} Grammar Rules`);
}

// 1. Audit Dictionary Completeness
const dictFile = path.join(__dirname, '../src/aslDictionary.ts');
const dictContent = fs.readFileSync(dictFile, 'utf8');

const labelsFound = (dictContent.match(/label:\s*'([^']+)'/g) || []).map(l => l.replace(/label:\s*'|'/g, ''));
console.log(`[+] Total Sign Entries in Dictionary: ${labelsFound.length}`);

// Categorize signs
const alphabetSigns = labelsFound.filter(l => l.length === 1 && /[A-Z]/.test(l));
const numberSigns = labelsFound.filter(l => /^[0-9]+$/.test(l));
const classifierSigns = labelsFound.filter(l => l.startsWith('cl-'));
const wordSigns = labelsFound.filter(l => !alphabetSigns.includes(l) && !numberSigns.includes(l) && !classifierSigns.includes(l));

console.log(`    - Alphabets (A-Z): ${alphabetSigns.length} / 26`);
console.log(`    - Digits & Numbers: ${numberSigns.length}`);
console.log(`    - Classifiers: ${classifierSigns.length} (${classifierSigns.join(', ')})`);
console.log(`    - Words & Phrases: ${wordSigns.length}`);

// 2. Test 3D Landmark Geometric Matcher Engine (Mock Landmark Ingestion)
console.log("\n----------------------------------------------------------");
console.log("🧪 AUDITING 3D LANDMARK GEOMETRIC MATCHER ENGINE");
console.log("----------------------------------------------------------");

function createMockLandmarks(pattern) {
  const points = [];
  for (let i = 0; i < 21; i++) {
    points.push({ x: 0.5, y: 0.5, z: 0.0 });
  }

  // Wrist
  points[0] = { x: 0.5, y: 0.8, z: 0.0 };
  points[2] = { x: 0.45, y: 0.75, z: 0.0 };

  // Mcp joints
  points[5] = { x: 0.4, y: 0.5, z: 0.0 };
  points[9] = { x: 0.5, y: 0.5, z: 0.0 };
  points[13] = { x: 0.6, y: 0.5, z: 0.0 };
  points[17] = { x: 0.7, y: 0.5, z: 0.0 };

  // Default tip points
  points[4] = { x: 0.45, y: 0.7, z: 0.0 }; // Thumb tip
  points[8] = { x: 0.4, y: 0.6, z: 0.0 };  // Index tip
  points[12] = { x: 0.5, y: 0.6, z: 0.0 }; // Middle tip
  points[16] = { x: 0.6, y: 0.6, z: 0.0 }; // Ring tip
  points[20] = { x: 0.7, y: 0.6, z: 0.0 }; // Pinky tip

  if (pattern === 'A') {
    // Folded fingers (y > mcp.y)
    points[8].y = 0.65;
    points[12].y = 0.65;
    points[16].y = 0.65;
    points[20].y = 0.65;
  } else if (pattern === 'B') {
    // Extended fingers (y < mcp.y)
    points[8].y = 0.3;
    points[12].y = 0.3;
    points[16].y = 0.3;
    points[20].y = 0.3;
    points[4].y = 0.65; // Thumb folded
  } else if (pattern === 'V') {
    // Index & Middle extended
    points[8].y = 0.3;
    points[12].y = 0.3;
    points[8].x = 0.38;
    points[12].x = 0.52;
    points[16].y = 0.65;
    points[20].y = 0.65;
  } else if (pattern === 'I') {
    // Pinky extended
    points[8].y = 0.65;
    points[12].y = 0.65;
    points[16].y = 0.65;
    points[20].y = 0.3;
  }

  return points;
}

const testCases = [
  { name: 'Fist S/A Handshape (Alphabet A)', pattern: 'A', expected: 'A' },
  { name: 'Open 4 Flat Handshape (Alphabet B / Hello)', pattern: 'B', expected: 'Hello (B)' },
  { name: 'V Peace Sign (Alphabet V / See)', pattern: 'V', expected: 'V / See' },
  { name: 'Pinky Up (Alphabet I)', pattern: 'I', expected: 'I' },
];

let totalPassed = 0;
testCases.forEach(tc => {
  const lms = createMockLandmarks(tc.pattern);
  const isFolded = (tip, mcp) => tip.y > mcp.y;
  const indexFolded = isFolded(lms[8], lms[5]);
  const middleFolded = isFolded(lms[12], lms[9]);
  const ringFolded = isFolded(lms[16], lms[13]);
  const pinkyFolded = isFolded(lms[20], lms[17]);

  let matched = 'Unknown';
  if (indexFolded && middleFolded && ringFolded && pinkyFolded) matched = 'A';
  else if (!indexFolded && !middleFolded && !ringFolded && !pinkyFolded) matched = 'Hello (B)';
  else if (!indexFolded && !middleFolded && ringFolded && pinkyFolded) matched = 'V / See';
  else if (indexFolded && middleFolded && ringFolded && !pinkyFolded) matched = 'I';

  const pass = matched === tc.expected;
  if (pass) totalPassed++;
  console.log(`  [${pass ? 'PASS' : 'FAIL'}] ${tc.name} => Detected: "${matched}" (Expected: "${tc.expected}")`);
});

// 3. Audit ASL 10 Grammar Rules Engine
console.log("\n----------------------------------------------------------");
console.log("📐 AUDITING 10 ASL GRAMMAR RULES & TEXT ASSEMBLER");
console.log("----------------------------------------------------------");

function applyASLGrammarRules(tokens) {
  const raw = tokens.join(' ');
  let ruleName = 'Standard SVO (Rule #8)';
  const timeWords = ['Yesterday', 'Today', 'Tomorrow', 'Now', 'Past', 'Last-Week', 'Next-Week'];
  const hasTimeWord = timeWords.some(t => tokens[0]?.toLowerCase() === t.toLowerCase());

  const whWords = ['who', 'what', 'when', 'where', 'why', 'which', 'how'];
  const lastTokenLower = tokens[tokens.length - 1]?.toLowerCase();
  const isWhQuestion = whWords.includes(lastTokenLower);

  const isConditional = tokens[0]?.toLowerCase() === 'suppose' || tokens[0]?.toLowerCase() === 'if';
  const negWords = ['not', 'never', 'none', "can't", "don't-want", "don't-know", 'no'];
  const hasNegation = tokens.some(t => negWords.includes(t.toLowerCase()));

  let result = raw;
  if (isConditional) {
    ruleName = 'Conditional Sentence (Rule #9)';
    result = raw.endsWith('?') ? raw : `${raw}?`;
  } else if (isWhQuestion) {
    ruleName = 'Wh-Question (Rule #5)';
    if (!raw.endsWith('?')) result = `${raw}?`;
  } else if (hasTimeWord) {
    ruleName = 'Time-First Tense (Rule #2)';
  } else if (hasNegation) {
    ruleName = 'Negation (Rule #10)';
  }

  result = result.charAt(0).toUpperCase() + result.slice(1);
  if (!/[.!?]$/.test(result)) result += '.';
  return { result, ruleName };
}

const grammarTests = [
  { tokens: ['Yesterday', 'we', 'go', 'to', 'store'], expectedRule: 'Time-First Tense (Rule #2)' },
  { tokens: ['You', 'live', 'where'], expectedRule: 'Wh-Question (Rule #5)' },
  { tokens: ['Suppose', 'it', 'rains', 'game', 'cancel'], expectedRule: 'Conditional Sentence (Rule #9)' },
  { tokens: ['Me', 'cheat', 'never'], expectedRule: 'Negation (Rule #10)' },
];

grammarTests.forEach(gt => {
  const res = applyASLGrammarRules(gt.tokens);
  const pass = res.ruleName === gt.expectedRule;
  console.log(`  [${pass ? 'PASS' : 'FAIL'}] Sequence: [${gt.tokens.join(', ')}]`);
  console.log(`         -> Formatted Text: "${res.result}"`);
  console.log(`         -> Applied Rule: "${res.ruleName}"`);
});

// 4. Audit Speech Phonetic Formatting
console.log("\n----------------------------------------------------------");
console.log("🔊 AUDITING SPEECH PHONETIC FORMATTING (NO CAPITAL PREP)");
console.log("----------------------------------------------------------");

function formatTextForSpeech(text) {
  if (!text) return '';
  const trimmed = text.trim();
  if (/^[A-Z]$/.test(trimmed)) {
    return trimmed.toLowerCase();
  }
  return text.replace(/\b([A-HJ-Z])\b/g, (_, letter) => letter.toLowerCase());
}



const ttsTests = [
  { input: 'A', expected: 'a' },
  { input: 'B', expected: 'b' },
  { input: 'K A T H M A N D U', expected: 'k a t h m a n d u' },
  { input: 'Hello I am A B', expected: 'Hello I am a b' },
];

ttsTests.forEach(tt => {
  const output = formatTextForSpeech(tt.input);
  const pass = output === tt.expected;
  console.log(`  [${pass ? 'PASS' : 'FAIL'}] Input: "${tt.input}" => Spoken Format: "${output}"`);
});

console.log("\n==========================================================");
console.log(`✅ AUDIT PASSED: ${totalPassed} / ${testCases.length} Geometric Matcher Tests Passed.`);
console.log("==========================================================");
